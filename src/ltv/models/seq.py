"""Candidate C: small sequence models over each user's whole history, for CPU.

Encoders (see ltv.data.seq_inputs for the inputs)
    gru2res      C1.  A GRU over 4-week history blocks, whose final state starts a second GRU
                 over the recent weeks. The linear layer on each flattened block is a strided
                 temporal convolution (kernel 4, stride 4).
    episodes     C1-episodes. The same, with play-spell tokens in place of the blocks.
    transformer  C2.  A small Transformer over blocks (or tokens) plus recent weeks.

Static inputs: the trees' features and embeddings for platform and country.

Heads (multi-task)
    weekly activity   52 sigmoids; each output week sees its own projected calendar position.
                      Their sum is expected active weeks.
    any active        P(re-engage); a separate head because weeks are not independent.
    any spend         P(spend > 0).
    hours, spend      a distribution head for the 52-week total given it is positive
                      (ZILN by default; see heads.py), and a 5-way share head that splits the
                      total across the horizon buckets. Expected value = P(positive) x head
                      mean; cumulative horizons = total x cumulative shares, so they can never
                      decrease.
    high spender      optional auxiliary task (ablation).

State carry. With gru2res, the history GRU's state after a block depends only on blocks
already complete, so it can be stored per user and advanced one block every four weeks;
weekly scoring then reruns only the recent weeks (``encode_blocks`` / ``h_hist``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ltv.data import seq_inputs as si
from ltv.data.sequences import SequenceStore
from ltv.models.base import QUANTILES, Preds, increments
from ltv.models.heads import HEADS

LOSS_KEYS = ("weekly", "any_active", "any_spend", "hours", "spend", "share_hours", "share_spend", "whale")


@dataclass
class SeqConfig:
    encoder: str = "gru2res"          # gru2res | episodes | transformer
    d: int = 64
    layers: int = 1
    dropout: float = 0.1
    head: str = "ziln"                # ziln | mixture | ordinal
    whale_aux: bool = False
    use_tabular: bool = True          # the trees' features as static inputs (default; see DESIGN.md)
    tf_layers: int = 2
    tf_heads: int = 4
    horizon: int = 52
    static_mean: list = field(default_factory=list)
    static_std: list = field(default_factory=list)
    hours_center: float = 3.0
    spend_center: float = 3.0
    hours_edges: list = field(default_factory=list)
    spend_edges: list = field(default_factory=list)
    whale_threshold: float = 0.0
    loss_scale: dict = field(default_factory=dict)

    @property
    def uses_tokens(self) -> bool:
        return self.encoder == "episodes"


class SeqModel(nn.Module):
    def __init__(self, cfg: SeqConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        f_hist = si.F_TOKEN if cfg.uses_tokens else si.F_BLOCK
        self.hist_proj = nn.Sequential(nn.Linear(f_hist, d), nn.Tanh())
        self.week_proj = nn.Sequential(nn.Linear(si.F_WEEK, d), nn.Tanh())
        if cfg.encoder == "transformer":
            layer = nn.TransformerEncoderLayer(d, cfg.tf_heads, 4 * d, cfg.dropout, batch_first=True)
            self.tf = nn.TransformerEncoder(layer, cfg.tf_layers, enable_nested_tensor=False)
            self.pos = nn.Embedding(si.MAX_BLOCKS + si.MAX_WEEKLY + 2, d)    # position counted back from the origin
            self.kind = nn.Embedding(2, d)
            self.pool = nn.Linear(2 * d, d)
        else:
            self.hist_gru = nn.GRU(d, d, cfg.layers, batch_first=True)
            self.bridge = nn.Linear(d, d)
            self.week_gru = nn.GRU(d, d, 1, batch_first=True)
        self.emb_platform = nn.Embedding(si.VOCAB["platform_id"], 4)
        self.emb_country = nn.Embedding(si.VOCAB["country_id"], 6)
        n_static = (si.N_STATIC if cfg.use_tabular else 0) + 10
        self.static_net = nn.Sequential(nn.Linear(n_static, d), nn.ReLU(), nn.Dropout(cfg.dropout))
        self.trunk = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(d, d), nn.ReLU())
        mean = torch.tensor(cfg.static_mean or [0.0] * si.N_STATIC, dtype=torch.float32)
        std = torch.tensor(cfg.static_std or [1.0] * si.N_STATIC, dtype=torch.float32)
        self.register_buffer("s_mean", mean)
        self.register_buffer("s_std", std)
        dz = 32
        self.week_emb = nn.Parameter(0.01 * torch.randn(cfg.horizon, dz))
        self.cal_proj = nn.Linear(si.F_FUTURE, dz)
        self.z_proj = nn.Linear(d, dz)
        self.act_out = nn.Linear(dz, 1)
        self.act_bias = nn.Linear(d, 1)
        self.any_active = nn.Linear(d, 1)
        self.any_spend = nn.Linear(d, 1)
        head = HEADS[cfg.head]
        self.hours_head = head(d, center=cfg.hours_center, edges=cfg.hours_edges or None)
        self.spend_head = head(d, center=cfg.spend_center, edges=cfg.spend_edges or None)
        self.share_hours = nn.Linear(d, 5)
        self.share_spend = nn.Linear(d, 5)
        self.whale = nn.Linear(d, 1) if cfg.whale_aux else None

    # -- encoders -----------------------------------------------------------------
    def encode_blocks(self, hist: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        """Final state of the history GRU, [layers, n, d]. Zero for users with no history blocks."""
        x = self.hist_proj(hist)
        packed = pack_padded_sequence(x, length.clamp_min(1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.hist_gru(packed)
        return h * (length > 0).to(h.dtype).view(1, -1, 1)

    def _encode_gru(self, b: dict, h_hist: torch.Tensor | None) -> torch.Tensor:
        if h_hist is None:
            key = "tokens" if self.cfg.uses_tokens else "blocks"
            h_hist = self.encode_blocks(b[key], b["token_len" if self.cfg.uses_tokens else "block_len"])
        h0 = torch.tanh(self.bridge(h_hist[-1])).unsqueeze(0)
        packed = pack_padded_sequence(self.week_proj(b["weekly"]), b["weekly_len"].clamp_min(1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, h = self.week_gru(packed, h0)
        return h[-1]

    def _encode_tf(self, b: dict) -> torch.Tensor:
        hist, hl = b["blocks"], b["block_len"]
        wk, wl = b["weekly"], b["weekly_len"]
        n, nh, nw = hist.shape[0], hist.shape[1], wk.shape[1]
        x = torch.cat([self.hist_proj(hist) + self.kind.weight[0], self.week_proj(wk) + self.kind.weight[1]], dim=1)
        ar_h, ar_w = torch.arange(nh).unsqueeze(0), torch.arange(nw).unsqueeze(0)
        valid = torch.cat([ar_h < hl.unsqueeze(1), ar_w < wl.unsqueeze(1)], dim=1)
        # Position = how far back from the origin, so "last week" is always position 0.
        back = torch.cat([(hl.unsqueeze(1) - 1 - ar_h) + wl.unsqueeze(1), wl.unsqueeze(1) - 1 - ar_w], dim=1)
        x = x + self.pos(back.clamp(0, self.pos.num_embeddings - 1))
        out = self.tf(x, src_key_padding_mask=~valid)
        v = valid.unsqueeze(-1).to(out.dtype)
        mean = (out * v).sum(1) / v.sum(1).clamp_min(1.0)
        last = out[torch.arange(n), nh + wl.clamp_min(1) - 1]
        return torch.tanh(self.pool(torch.cat([mean, last], dim=-1)))

    def forward(self, b: dict, h_hist: torch.Tensor | None = None) -> dict:
        h = self._encode_tf(b) if self.cfg.encoder == "transformer" else self._encode_gru(b, h_hist)
        cat = torch.cat([self.emb_platform(b["cat"][:, 0]), self.emb_country(b["cat"][:, 1])], dim=-1)
        if self.cfg.use_tabular:
            cat = torch.cat([(b["static"] - self.s_mean) / self.s_std, cat], dim=-1)
        z = self.trunk(torch.cat([h, self.static_net(cat)], dim=-1))
        wk = torch.tanh(self.z_proj(z).unsqueeze(1) + (self.cal_proj(b["future"]) + self.week_emb).unsqueeze(0))
        out = {"weekly": self.act_out(wk).squeeze(-1) + self.act_bias(z),
               "any_active": self.any_active(z).squeeze(-1), "any_spend": self.any_spend(z).squeeze(-1),
               "hours": self.hours_head(z), "spend": self.spend_head(z),
               "share_hours": torch.log_softmax(self.share_hours(z), dim=-1),
               "share_spend": torch.log_softmax(self.share_spend(z), dim=-1), "z": z}
        if self.whale is not None:
            out["whale"] = self.whale(z).squeeze(-1)
        return out

    # -- losses -------------------------------------------------------------------
    def losses(self, out: dict, y: dict) -> dict[str, torch.Tensor]:
        bce = nn.functional.binary_cross_entropy_with_logits
        n = float(len(y["hours_52"]))
        L = {"weekly": bce(out["weekly"], y["weekly_active"]),
             "any_active": bce(out["any_active"], y["any_active_52"]),
             "any_spend": bce(out["any_spend"], y["any_spend_52"])}
        for tgt, head in (("hours", self.hours_head), ("spend", self.spend_head)):
            total = y[f"{tgt}_52"]
            pos = total > 0
            if bool(pos.any()):
                prm = {k: v[pos] for k, v in out[tgt].items()}
                L[tgt] = head.nll(prm, total[pos]).sum() / n                 # per user, not per positive
                share = (y[f"inc_{tgt}"][pos] / total[pos].unsqueeze(1)).clamp_min(0)
                L[f"share_{tgt}"] = -(share * out[f"share_{tgt}"][pos]).sum() / n
            else:
                L[tgt] = L[f"share_{tgt}"] = out["any_spend"].sum() * 0.0
        if self.whale is not None:
            L["whale"] = bce(out["whale"], (y["spend_52"] >= self.cfg.whale_threshold).float())
        return L

    def total_loss(self, L: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum of head losses, each divided by its size at initialisation so they start equal."""
        s = self.cfg.loss_scale
        return sum(v / s.get(k, 1.0) for k, v in L.items())

    def state(self) -> dict:
        return {"cfg": asdict(self.cfg), "weights": {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()}}

    @classmethod
    def from_state(cls, st: dict) -> "SeqModel":
        m = cls(SeqConfig(**st["cfg"]))
        m.load_state_dict({k: torch.tensor(v) for k, v in st["weights"].items()})
        return m.eval()


# ---------------------------------------------------------------------------
# Tensors and prediction
# ---------------------------------------------------------------------------

def to_tensors(inp: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in inp.items()}


def label_tensors(y: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    t = {k: torch.from_numpy(np.ascontiguousarray(y[k])) for k in
         ("weekly_active", "any_active_52", "any_spend_52", "hours_52", "spend_52")}
    t["inc_hours"] = torch.from_numpy(np.maximum(increments(y["cum_hours"]), 0).astype(np.float32))
    t["inc_spend"] = torch.from_numpy(np.maximum(increments(y["cum_spend"]), 0).astype(np.float32))
    return t


@torch.no_grad()
def predict_seq(model: SeqModel, store: SequenceStore, rows: np.ndarray, origin: int,
                known_window="current_season", batch: int = 4096) -> Preds:
    model.eval()
    parts = {k: [] for k in ("p_re", "aw", "weekly", "eh", "ch", "p_sp", "es", "cs", "qh", "qs", "z")}
    for a in range(0, len(rows), batch):
        inp = si.build_inputs(store, rows[a:a + batch], origin, known_window, tokens=model.cfg.uses_tokens)
        out = model(to_tensors(inp))
        p_re, p_sp = torch.sigmoid(out["any_active"]), torch.sigmoid(out["any_spend"])
        weekly = torch.sigmoid(out["weekly"])
        eh = p_re * model.hours_head.mean(out["hours"])
        es = p_sp * model.spend_head.mean(out["spend"])
        parts["p_re"].append(p_re.numpy()); parts["p_sp"].append(p_sp.numpy())
        parts["weekly"].append(weekly.numpy()); parts["aw"].append(weekly.sum(1).numpy())
        parts["eh"].append(eh.numpy()); parts["es"].append(es.numpy())
        parts["ch"].append((eh.unsqueeze(1) * torch.cumsum(torch.exp(out["share_hours"]), dim=1)).numpy())
        parts["cs"].append((es.unsqueeze(1) * torch.cumsum(torch.exp(out["share_spend"]), dim=1)).numpy())
        hp = {k: v.numpy().astype(np.float64) for k, v in out["hours"].items()}
        sp = {k: v.numpy().astype(np.float64) for k, v in out["spend"].items()}
        parts["qh"].append(model.hours_head.quantiles(hp, p_re.numpy().astype(np.float64), QUANTILES))
        parts["qs"].append(model.spend_head.quantiles(sp, p_sp.numpy().astype(np.float64), QUANTILES))
        parts["z"].append(out["z"].numpy())
    c = {k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()}
    return Preds(store.user_ids[rows], origin, c["p_re"], c["aw"], c["eh"], c["p_sp"], c["es"],
                 cum_hours=c["ch"], cum_spend=c["cs"], weekly_p_active=c["weekly"], q_hours=c["qh"],
                 q_spend=c["qs"], state=c["z"])
