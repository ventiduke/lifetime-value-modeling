# Seed papers on LTV prediction: four user-supplied papers and the references they build on

Provenance key used throughout these notes:

- **[FT]** = taken from the full text, which I read (arXiv HTML fetched with curl).
- **[ABS]** = taken from the arXiv abstract page only.
- **[SNIP]** = taken only from web-search result snippets; the full text was NOT read.
- **[CALC]** = a figure I computed from numbers in the paper's tables (not a number the authors print).

Access summary: Papers 2, 3 and 4 were read in full on arXiv. Paper 1 (ACM TKDD, DOI 10.1145/3441444) could not be read: dl.acm.org, doi.org, api.crossref.org and the author-hosted PDF host (web-ainf.aau.at) are all blocked in this session, and no arXiv version was found. Everything about Paper 1 below is from search snippets or from how Paper 4 describes it. I also read two heavily cited works on arXiv to characterise them: ZILN (arXiv 1912.07753, full text) and the Kuaishou ODMN/MDME paper (arXiv 2208.13358, full text).

Paper identities:

| # | Supplied link | Identified as |
|---|---|---|
| 1 | dl.acm.org/doi/pdf/10.1145/3441444 | Bauer and Jannach, "Improved Customer Lifetime Value Prediction With Sequence-To-Sequence Learning and Feature-Based Models", ACM TKDD 15(5), 2021 [SNIP] |
| 2 | arxiv.org/html/2412.20211v2 | Ma et al. (Fudan / Kuaishou), "Sequence Generation Modeling for Continuous Value Prediction" (v2 title); retitled "Generative Regression Based Watch Time Prediction for Short-Video Recommendation" in v3 [FT] |
| 3 | arxiv.org/html/2412.20295v1 | Chen, Ng, Steininger, Smyl (Meta / Uber), "Predicting Customer Lifetime Value Using Recurrent Neural Net", KDD 2022 Customer Journey Optimization Workshop [FT] |
| 4 | arxiv.org/pdf/2408.08585 | Weng et al. (Tencent FiT), "OptDist: Learning Optimal Distribution for Customer Lifetime Value Prediction", CIKM 2024 [FT] |

## What exactly does each paper propose, and on what data and horizon?

### Takeaway
Two of the four are output-distribution papers for a single scalar target (Paper 4 extends ZILN to several ZILN heads with a learned selector; Paper 2 replaces the regression head with an autoregressive token decoder and is primarily a watch-time paper), and two are sequence-model papers (Paper 3 a dilated RNN on user-level time series at 4 to 26 week horizons; Paper 1 an encoder-decoder RNN stacked with a GBM). Only Paper 3 uses data shaped like ours (sparse per-user weekly series, existing users, multi-horizon rolling forecasts); none of the four evaluates a 52-week rolling forecast on a weekly panel.

### Cited Findings

#### Paper 1: Bauer and Jannach 2021 (full text NOT read)
- Citation: Josef Bauer and Dietmar Jannach, "Improved Customer Lifetime Value Prediction With Sequence-To-Sequence Learning and Feature-Based Models", ACM Transactions on Knowledge Discovery from Data (TKDD) 15(5), 2021, pages 1-37, DOI 10.1145/3441444 [SNIP, and confirmed by the reference list of Paper 4, FT] — [DOI](https://doi.org/10.1145/3441444); [OptDist reference list](https://arxiv.org/html/2408.08585)
- Motivation per abstract fragments: standard methods based on purchase recency, frequency and past profit/revenue statistics "often have limited predictive power", so ML techniques are applied [SNIP] — [author-hosted PDF surfaced by search, not fetched](https://web-ainf.aau.at/pub/jannach/files/Journal_TKDD_2021.pdf)
- Method per abstract fragments: "a tailored deep learning approach based on encoder-decoder sequence-to-sequence recurrent neural networks (RNNs) with augmented temporal convolutions", combined with gradient boosting machines; embeddings of the recorded purchase logs are learned for both the GBM and the RNN; the stacked combination is called GBM-S2S ("Stacking of our GBM and Sequence-To-Sequence RNN") [SNIP] — [author-hosted PDF surfaced by search, not fetched](https://web-ainf.aau.at/pub/jannach/files/Journal_TKDD_2021.pdf)
- Data per abstract fragments: real-world data from "a larger e-commerce company" plus a public dataset from online retail; the abstract says the sequence-based model alone "leads to competitive performance results" [SNIP] — [author-hosted PDF surfaced by search, not fetched](https://web-ainf.aau.at/pub/jannach/files/Journal_TKDD_2021.pdf)
- How a later paper characterises it: Paper 4 groups it with works that "investigated various sequential models for behaviors in CLTV", describing it as "combining RNNs with gradient boosting machines (GBMs) to capture historical customer behavior", and calls this line of work "perpendicular" to distribution modelling and combinable with it [FT of Paper 4] — [OptDist](https://arxiv.org/html/2408.08585)

#### Paper 2: Generative Regression (GR), Ma et al.
- Citation: Hongxu Ma, Kai Tian, Tao Zhang, Xuefeng Zhang, Chunjie Chen, Han Li, Jihong Guan, Shuigeng Zhou (Fudan University, Kuaishou Technology, Tongji University). arXiv:2412.20211. v1 28 Dec 2024, v2 24 Jan 2025 titled "Sequence Generation Modeling for Continuous Value Prediction"; v3 12 Apr 2025 retitled "Generative Regression Based Watch Time Prediction for Short-Video Recommendation" and adds author Han Zhou. No venue is stated on arXiv (the comments field is a template placeholder) [FT, ABS] — [v2](https://arxiv.org/html/2412.20211v2); [abs page](https://arxiv.org/abs/2412.20211)
- Problem: continuous value prediction, chiefly watch time in short-video recommendation, where the target has a wide range and long tail; LTV is a secondary transfer experiment (Section 4.6) [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Core idea: decompose the scalar label y into a sequence of tokens whose values sum to y, then train an autoregressive decoder to emit that sequence; the prediction is the sum of the decoded tokens' values [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Architecture: a feed-forward encoder (2-layer MLP) over user and item features produces a hidden vector; a single-layer LSTM decoder generates tokens from a start token until an end token. The authors state watch-time prediction "does not inherently depend on history action sequences", so the encoder is not sequential [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Vocabulary construction (Algorithm 1): data-driven "dynamic percentile" procedure. Take the q-th percentile of the labels as a token, subtract it from every label at or above it, decay q (q_start 0.9999, decay 0.99, floor q_end 0.90) and repeat until the maximum relative residual is below 1e-6. Design principles: completeness, token-frequency balance, adaptability across datasets [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Label encoding: greedy decomposition, largest token first, constrained to be non-increasing and of minimal length; about 0.1% relative restoration error is tolerated [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Loss: token cross-entropy plus lambda times a Huber loss on the decoded sum, with lambda = 10 [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Training tricks: "Curriculum Learning with Embedding Mixup" (CLEM). Scheduled sampling moves from teacher forcing to free running with a non-linear decay; embedding mixup feeds the next step a logit-weighted mixture of embeddings in a window around the predicted token (window = half the vocabulary), which restores a gradient path through the argmax and reduces exposure bias [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Other settings: dropout 0.3, learning rate 5e-4, 20 epochs, batch 1024, Adam; GR has 4.18M parameters against 0.86M for the baselines (enlarged 4.34M baselines are also reported) [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Data: Kuaishou industrial logs (about 400M DAU, 4 days of interactions for training, next day for test); CIKM16 (310,302 sessions, 122,991 items); KuaiRec (7,176 users, 10,728 items, 12,530,806 impressions). LTV transfer experiment on Criteo-SSC and Kaggle Acquire Valued Shoppers "following the settings" of OptDist [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Horizon: per-impression watch time (seconds). For the LTV experiment the horizons are those of the OptDist setup: 30 days (Criteo-SSC) and one year (Kaggle) [FT of Papers 2 and 4] — [v2](https://arxiv.org/html/2412.20211v2); [OptDist](https://arxiv.org/html/2408.08585)
- Inference cost: the paper's own comparison table gives inference complexity O(m) for plain value regression, O(mn) for the ordinal methods TPM and CREAD, and O(mdt) for GR, where m is the last hidden size, n the number of discretisation segments, d the vocabulary size and t the generated sequence length [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Serving cost, v3 only: against the CREAD control, GR "exhibited a 10.2% reduction in average queries per second (QPS) during online serving"; the authors say ROI still met the bar for full deployment [FT of v3] — [v3 / latest](https://arxiv.org/html/2412.20211)
- Beam-search ensembling at inference "significantly boosts model performance" but "incurs more computational cost"; embedding mixup is offered as the cheaper substitute [FT] — [v2](https://arxiv.org/html/2412.20211v2)
- Stated limitations: the authors say of the vocabulary and label-encoding scheme "we do not contend that the aforementioned strategy is the most optimal one". No limitations section [FT] — [v2](https://arxiv.org/html/2412.20211v2)

#### Paper 3: Chen, Ng, Steininger, Smyl
- Citation: Huigang Chen (Meta), Edwin Ng (Amazon; work done at Uber), Gavin Steininger (Uber), Slawek Smyl (Meta), "Predicting Customer Lifetime Value Using Recurrent Neural Net", arXiv:2412.20295 [stat.AP], 28 Dec 2024; 4 pages; "first presented in Customer Journey Optimization Workshop in KDD 2022" [FT, ABS] — [paper](https://arxiv.org/html/2412.20295v1); [abs page](https://arxiv.org/abs/2412.20295)
- Setting: user value in services without fixed fees (the paper's examples include free-to-play gaming); two problems. "Acquisition LTV" (new users, Uber): predict the not-yet-realised remainder of value from current age-in-system up to a "graduation" age; the text describes the journey from 0 to 90 days age-in-system. "Rolling LTV" (existing users, Meta Quest): on date t0 forecast the sum of user value over the next T weeks for several horizons at once [FT] — [paper](https://arxiv.org/html/2412.20295v1)
- Three connected time dimensions are modelled: user cohort (join date), age-in-system, and calendar date (contemporaneous effects such as events and outages). The authors argue calendar effects must be modelled even though they cannot be forecast, so that cohort and age effects are estimated correctly [FT] — [paper](https://arxiv.org/html/2412.20295v1)
- Meta Quest data: gross revenue from app and in-app purchases at user level; horizons 1, 4, 13 and 26 weeks; user-level time series up to 4 years long; 100+ features (demographic, behavioural, seasonal); embeddings for large categorical variables. Number of users is not given [FT] — [paper](https://arxiv.org/html/2412.20295v1)
- Architecture: multi-layer RNN with LSTM-style cells arranged in blocks, dilated connections (a layer reads state from 2 or more steps back), ResNet-style shortcuts between blocks, and a final linear adaptor layer. The "dRNN" cell is derived from LSTM and GRU; it splits its output into a "real" output passed up the stack and a separate controlling state used for gating, and uses a fusion gate over the previous and the delayed cell state [FT] — [paper](https://arxiv.org/html/2412.20295v1)
- Sparsity handling: "most users purchase infrequently", so periods with no transactions ("zero-records") are subsampled; because this makes RNN steps uneven in time, calendar features such as week of year and "number of weeks since the last step" are added [FT] — [paper](https://arxiv.org/html/2412.20295v1)
- Evaluation metric proposed: adjusted SMAPE, aSMAPE = (2/N) * sum |A - F| / |max(A, a) + F|, with floor a = $1.00 in their experiments, because MAPE is meaningless when actuals are zero or near zero and SMAPE is bounded [FT] — [paper](https://arxiv.org/html/2412.20295v1)
- Criticisms of the alternatives (related work): BTYD assumes transaction amount, frequency and active duration are independent; it is hard to add features such as seasonality; it has too little data to discriminate new users; each line of business must be modelled separately. Feature-based supervised models (regularised regression, random forest, GBT) assume independent observations and need explicit lag features, which "may over inflate the feature space and run into runtime performance issues" [FT] — [paper](https://arxiv.org/html/2412.20295v1)
- Not stated anywhere in the paper: the loss function, any output distribution, dataset sizes, training or scoring cost, or limitations [FT] — [paper](https://arxiv.org/html/2412.20295v1)

#### Paper 4: OptDist, Weng et al.
- Citation: Yunpeng Weng, Xing Tang, Zhenhao Xu, Fuyuan Lyu, Dugang Liu, Zexu Sun, Xiuqiang He (FiT Tencent; McGill and MILA; Guangdong Laboratory of AI and Digital Economy; Renmin University), "OptDist: Learning Optimal Distribution for Customer Lifetime Value Prediction", CIKM 2024 (Boise, 21-25 Oct 2024), DOI 10.1145/3627673.3679712, arXiv:2408.08585 [FT] — [paper](https://arxiv.org/html/2408.08585)
- Problem: a single parametric distribution (ZILN) with one set of parameters per user cannot fit a CLTV distribution that differs sharply across user groups; hand-bucketed approaches (MDME) need rule-based bucketing and suffer class imbalance and uneven within-bucket distributions [FT] — [paper](https://arxiv.org/html/2408.08585)
- Method: shared embedding and bottom layers feed (a) a Distribution Learning Module of L sub-distribution networks (SDNs), each an MLP emitting ZILN parameters (p, mu, sigma), and (b) a Distribution Selection Module, an MLP with softmax over the L SDNs. During training a straight-through Gumbel-softmax gives near-one-hot weights, so each SDN trains mainly on the users routed to it. The per-user loss is the selection-weighted sum of the SDN ZILN negative log-likelihoods [FT] — [paper](https://arxiv.org/html/2408.08585)
- Alignment mechanism: the per-SDN losses for a user become supervision for the selector. Hard pseudo-label = the SDN with the lowest loss, trained with a focal-weighted cross-entropy; soft label = softmax of negative losses, trained with a KL term. Total loss = weighted ZILN NLL + CE + KL, optimised in one level (DARTS-style approximation) [FT] — [paper](https://arxiv.org/html/2408.08585)
- Inference: take the argmax SDN only and output p * exp(mu + sigma^2 / 2). The authors state the SDNs are simple MLPs that "can be parallelized, thus not increasing the time complexity of the model" [FT] — [paper](https://arxiv.org/html/2408.08585)
- Data and horizons: Criteo-SSC (15,995,633 click-level samples, 7.20% positive; label = revenue within 30 days of a click); Kaggle Acquire Valued Shoppers (805,753 samples, 90.12% positive; label = total value of a company's products bought by the user in the year after first purchase; top three companies kept); private Industrial set from a mutual-fund platform (4,535,675 samples, 6.35% positive; new users who have never invested; predict conversion and CLTV in the next 30 days). Random 7:1:2 split [FT] — [paper](https://arxiv.org/html/2408.08585)
- Tuning: best number of sub-distributions L is 4 on both public datasets and 5 on the industrial one (searched over 2 to 6); too high a Gumbel temperature hurts [FT] — [paper](https://arxiv.org/html/2408.08585)
- Compute: TensorFlow on one Nvidia Tesla V100 32GB, 128GB RAM, 8 CPU cores. No training or scoring time is reported [FT] — [paper](https://arxiv.org/html/2408.08585)
- Deployment: "deployed on a large-scale financial platform" for audience targeting in customer-acquisition campaigns [FT] — [paper](https://arxiv.org/html/2408.08585)
- Stated limitations: none; the authors note representation-learning methods (sequence encoders, missing-feature and cross-domain modules) were excluded from the baselines as orthogonal and pluggable into the shared bottom [FT] — [paper](https://arxiv.org/html/2408.08585)

### Inferences
- Paper 2's encoder ignores behaviour history by design, so it contributes nothing on the sequence side; its only transferable element for us is the output head.
- Paper 2's decoded value is a greedy argmax path, not a distributional mean, and the paper reports no calibration metric. Whether it yields calibrated expectations (which we need for expected hours and spend) is untested in the paper.
- Paper 3's zero-record subsampling plus a "weeks since last step" feature is a sequence-compression trick directly applicable to our sparse 104-week inputs; it should cut sequence length (and so scoring cost) for mostly-inactive users, though the paper reports no cost numbers.
- Paper 3's cohort / age / calendar-date framing maps onto our season calendar: season launch is a calendar-date effect that hits all cohorts at once.
- OptDist at inference is one selector MLP plus L small MLP heads, so the marginal scoring cost over a single ZILN head is small. This is my reading of the architecture; the paper gives no timing.
- OptDist's hard selection means the reported predictive distribution for a user is one ZILN component, not a mixture, so its uncertainty output is no richer than plain ZILN's.

### Gaps
- Paper 1 full text was unreachable. Not verified: dataset sizes, prediction horizon, zero-value share, the exact loss, whether the decoder emits a per-period sequence of future values, what "augmented temporal convolutions/features" consist of, whether known future covariates feed the decoder, and how the stack is wired (S2S outputs as GBM features versus a meta-learner).
- Paper 2: publication venue unknown; how zero labels are tokenised in the LTV experiment is not described.
- Paper 3: no user counts, no loss function, no serving information.

## Which baselines does each paper beat, by how much, on which metrics?

### Takeaway
OptDist improves on a single ZILN head by 1.6% to 5.5% in normalized Gini and 2% to 24% in MAE across three datasets; GR beats OptDist on MAE and Spearman on the two public LTV sets but reports neither Gini nor calibration; the Uber/Meta dilated RNN cuts aSMAPE by 16% to 26% relative to LightGBM and far more relative to BTYD, with the margin smallest at the longest (26-week) horizon. On the Uber acquisition task the RNN only beat XGBoost once categorical embeddings were added (by under 1%).

### Cited Findings

#### Paper 4 (OptDist), offline, Table 2 [FT]
| Dataset | Model | MAE | Norm-GINI | Spearman | Norm-GINI(+) | Spearman(+) |
|---|---|---|---|---|---|---|
| Criteo-SSC | Two-stage | 21.719 | 0.5278 | 0.2386 | 0.2204 | 0.2565 |
| Criteo-SSC | MTL-MSE | 21.190 | 0.6330 | 0.2478 | 0.4340 | 0.3663 |
| Criteo-SSC | ZILN | 20.880 | 0.6338 | 0.2434 | 0.4426 | 0.3874 |
| Criteo-SSC | MDME | 16.598 | 0.4383 | 0.2269 | 0.2297 | 0.2952 |
| Criteo-SSC | MDAN | 20.030 | 0.6209 | 0.2470 | 0.4128 | 0.3521 |
| Criteo-SSC | OptDist | 15.784 | 0.6437 | 0.2505 | 0.4428 | 0.3903 |
| Kaggle | Two-stage | 74.782 | 0.5498 | 0.4313 | 0.5505 | 0.4596 |
| Kaggle | MTL-MSE | 74.065 | 0.5503 | 0.4329 | 0.5349 | 0.4328 |
| Kaggle | ZILN | 72.528 | 0.6693 | 0.5239 | 0.6627 | 0.5303 |
| Kaggle | MDME | 72.900 | 0.6305 | 0.5163 | 0.6213 | 0.5289 |
| Kaggle | MDAN | 73.940 | 0.6648 | 0.4367 | 0.6680 | 0.4567 |
| Kaggle | OptDist | 70.929 | 0.6814 | 0.5249 | 0.6715 | 0.5346 |
| Industrial | Two-stage | 0.887 | 0.6670 | 0.0781 | 0.5588 | 0.4467 |
| Industrial | MTL-MSE | 0.548 | 0.7194 | 0.1161 | 0.5575 | 0.4274 |
| Industrial | ZILN | 0.389 | 0.7854 | 0.1208 | 0.5899 | 0.5401 |
| Industrial | MDME | 0.419 | 0.7277 | 0.1229 | 0.5609 | 0.5119 |
| Industrial | MDAN | 0.437 | 0.7629 | 0.1214 | 0.5816 | 0.5383 |
| Industrial | OptDist | 0.322 | 0.8283 | 0.1282 | 0.6271 | 0.5476 |

- Source for the table above; "(+)" columns are computed on positive-label samples only; the paper marks OptDist's MAE and Norm-GINI gains as significant at p < 0.05 on all three datasets — [OptDist](https://arxiv.org/html/2408.08585)
- OptDist versus plain ZILN, relative change [CALC]: Norm-GINI +1.6% (Criteo-SSC), +1.8% (Kaggle), +5.5% (Industrial); MAE -24.4%, -2.2%, -17.2% respectively — [OptDist Table 2](https://arxiv.org/html/2408.08585)
- Authors' reading of the baselines: ZILN beats two-stage and MTL-MSE overall; MDME is "unstable across the datasets" (low MAE but poor ranking on Criteo-SSC) because errors accumulate across sub-distribution, bucket and within-bucket predictions [FT] — [OptDist](https://arxiv.org/html/2408.08585)
- Ablation, Norm-GINI (Criteo-SSC / Kaggle / Industrial): full 0.6437 / 0.6814 / 0.8283; without Gumbel-softmax 0.6389 / 0.6628 / 0.8231; without KL 0.6382 / 0.6786 / 0.8158; without CE 0.6366 / 0.6761 / 0.8107; without both alignment terms 0.6361 / 0.6740 / 0.8023 [FT] — [OptDist](https://arxiv.org/html/2408.08585)
- Online A/B (50/50 traffic split per campaign, equal budgets, equal number of targeted users; users who had visited but never purchased): relative ROI lift over the unnamed production baseline at 7 / 14 / 30 days: Campaign A +8.96% / +17.31% / +21.90%; B +9.04% / +9.83% / +12.51%; C +6.47% / +8.68% / +11.45%; D +14.42% / +16.53% / +19.06% [FT] — [OptDist](https://arxiv.org/html/2408.08585)

#### Paper 2 (GR)
- LTV transfer experiment (Table 6 in v2), MAE / Spearman. Criteo-SSC: Two-stage 21.719 / 0.2386; MTL-MSE 21.190 / 0.2478; ZILN 20.880 / 0.2434; MDME 16.598 / 0.2269; MDAN 20.030 / 0.2470; OptDist 15.784 / 0.2505; GR 12.996 / 0.3026. Kaggle: Two-stage 74.782 / 0.4313; MTL-MSE 74.065 / 0.4329; ZILN 72.528 / 0.5239; MDME 72.900 / 0.5163; MDAN 73.940 / 0.4367; OptDist 70.929 / 0.5249; GR 67.035 / 0.5334 [FT] — [GR v2](https://arxiv.org/html/2412.20211v2)
- The baseline numbers in that table are identical to OptDist's Table 2; the authors say they follow OptDist's settings and baselines [FT] — [GR v2](https://arxiv.org/html/2412.20211v2); [OptDist](https://arxiv.org/html/2408.08585)
- Authors' headline: 17.66% MAE improvement and 20.79% Spearman improvement over OptDist on Criteo-SSC [FT] — [GR v2](https://arxiv.org/html/2412.20211v2)
- On Kaggle the gain over OptDist is smaller: MAE -5.5%, Spearman +1.6% [CALC] — [GR v2 Table 6](https://arxiv.org/html/2412.20211v2)
- Normalized Gini and any calibration metric are not reported for the LTV experiment [FT] — [GR v2](https://arxiv.org/html/2412.20211v2)
- Watch-time results, MAE / XAUC. KuaiRec watch time: VR 7.634 / 0.504; WLR 6.047 / 0.525; D2Q 5.426 / 0.565; TPM 3.456 / 0.571; CREAD 3.307 / 0.594; GR 3.196 / 0.614. CIKM16: VR 1.039 / 0.641; TPM 0.850 / 0.676; CREAD 0.865 / 0.678; GR 0.815 / 0.691. Industrial: VR 46.343 / 0.588; TPM 41.486 / 0.593; CREAD 39.979 / 0.597; GR 38.528 / 0.604. Each experiment repeated 5 times and averaged [FT] — [GR v2](https://arxiv.org/html/2412.20211v2)
- Parameter-matched check: CREAD-large (4.34M parameters) scores 3.293 / 0.599 on KuaiRec against GR (4.18M) 3.196 / 0.614, so the gain is not explained by size alone [FT] — [GR v2](https://arxiv.org/html/2412.20211v2)
- Ablation on KuaiRec: removing the whole CLEM strategy degrades MAE from 3.196 to 3.416 and XAUC from 0.614 to 0.584; curriculum without embedding mixup gives 3.359 / 0.588; mixup with pure teacher forcing gives 3.241 / 0.604 [FT] — [GR v2](https://arxiv.org/html/2412.20211v2)
- Online A/B on Kuaishou, v2: 2.5% of traffic (more than 10M users), six days; +0.055% app usage time, +0.072% average app usage per user, +0.042% app launches, +0.129% video consumption time [FT] — [GR v2](https://arxiv.org/html/2412.20211v2)
- Online A/B as restated in v3: 6% of traffic (more than 25M users), six days, CREAD as control; +0.112% app usage time (p = 0.01), +0.087% average app usage per user, +0.129% video consumption time. The two versions disagree on traffic share and on two of the lift figures [FT of v3] — [GR v3 / latest](https://arxiv.org/html/2412.20211)

#### Paper 3 (dilated RNN)
- Rolling LTV, Meta Quest, out-of-sample aSMAPE with floor $1.00 (lower is better) at 4 / 13 / 26 weeks: BTYD 53.6% / 70.5% / 78.3%; LightGBM 30.4% / 53.7% / 61.1%; dRNN 22.9% / 39.8% / 51.2% [FT] — [paper](https://arxiv.org/html/2412.20295v1)
- Relative aSMAPE reduction of dRNN versus LightGBM: 24.7% (4W), 25.9% (13W), 16.2% (26W) [CALC] — [paper Table 3](https://arxiv.org/html/2412.20295v1)
- Acquisition LTV, Uber, RMSE: xgboost 2,292; 2-cell RNN on conventional features 2,830; 2-cell RNN with embedding features (city encodings) 2,271. The paper states the plain RNN "is not found to fit the data as well as an xgboost model" [FT] — [paper](https://arxiv.org/html/2412.20295v1)
- The embedding RNN's edge over xgboost on the Uber task is 0.9% in RMSE, and the plain RNN is 23% worse than xgboost [CALC] — [paper Table 2](https://arxiv.org/html/2412.20295v1)
- Internal inconsistencies to be aware of: the abstract says the method improves "median absolute percent error", but the results use RMSE and aSMAPE; a 1-week horizon is mentioned but not tabulated; the text says "nRNN" where the table says "dRNN". The BTYD variant used is not specified [FT] — [paper](https://arxiv.org/html/2412.20295v1)

#### Paper 1
- The only result-level statement recovered is from the abstract: the sequence-based model alone is "competitive", and (per a later paper's summary surfaced by search) stacking it with the GBM improved overall performance because the two models capture different patterns [SNIP] — [author-hosted PDF surfaced by search, not fetched](https://web-ainf.aau.at/pub/jannach/files/Journal_TKDD_2021.pdf)

### Inferences
- The public LTV benchmarks used by Papers 2 and 4 look little like our problem: Criteo-SSC is click-level 30-day revenue, and the Kaggle set is 90% positive with features from a first purchase only. Gains measured there are evidence about output heads on cross-sectional features, not about rolling 52-week forecasts on a weekly panel.
- On the Kaggle set (the only one-year horizon among the public sets) every method lands within a narrow band (Norm-GINI 0.6693 for ZILN to 0.6814 for OptDist; MAE 72.5 to 67.0 including GR), so the choice of output head moved one-year results by a few percent at most.
- Paper 3 is the nearest evidence to our candidate comparison A versus C, and it favours the sequence model, but with a shrinking margin as horizon lengthens (26 weeks is their longest; we need 52) and with a result on the acquisition task that depended on embeddings. It supports keeping both A and C in the bake-off rather than presuming C wins.
- MDME's pattern in OptDist's table (best-but-one MAE, worst ranking on Criteo-SSC) is a warning that MAE and ranking metrics can disagree sharply for bucketed heads; a bake-off should score both.

### Gaps
- No numbers at all for Paper 1 (metrics, baselines, margins).
- None of the three papers read reports confidence intervals on offline metrics beyond OptDist's significance markers and GR's 5-run averaging; Paper 3 gives single numbers with no variance.
- No paper reports decile calibration or AUC for P(spend > 0); OptDist and GR report MAE and rank metrics only.

## Which prior works do several of them cite as the standard approaches?

### Takeaway
ZILN (Wang, Liu and Miao 2019) is the shared reference point: it is the building block of OptDist, a baseline in OptDist, GR and the Kuaishou paper, and the source of the normalized-Gini and decile-chart evaluation convention. The other recurring pillars are the BTYD family (Pareto/NBD, BG/NBD) as the classical baseline, the two-stage classify-then-regress approach from the games literature (Drachen et al. 2018), and Kuaishou's ODMN/MDME as the industrial multi-horizon solution.

### Cited Findings

Works cited by two or more of the papers I could read (Papers 2, 3, 4), with the two secondary papers I read (ZILN, Kuaishou) noted where they also cite the work. Reference lists: [GR](https://arxiv.org/html/2412.20211v2), [Chen et al.](https://arxiv.org/html/2412.20295v1), [OptDist](https://arxiv.org/html/2408.08585), [ZILN](https://arxiv.org/html/1912.07753), [Kuaishou](https://arxiv.org/html/2208.13358).

- Wang, Liu, Miao 2019, "A Deep Probabilistic Model for Customer Lifetime Value Prediction" (ZILN; arXiv 1912.07753). Cited by Papers 2 and 4 and by Kuaishou. Models LTV as a point mass at zero plus a lognormal; one network emits (p, mu, sigma); gives churn probability and value in one model and quantile-based uncertainty; recommends normalized Gini for discrimination and decile charts (decile-level MAPE) for calibration [FT] — [ZILN](https://arxiv.org/html/1912.07753)
- ZILN's own evidence: on Kaggle Acquire Valued Shoppers (311K customers, 12-month horizon after first purchase), ZILN versus MSE raised Spearman on average 23.9% for linear models and 48.0% for DNNs and cut decile-level MAPE 60.0% (linear) and 68.9% (DNN); normalized Gini improved over a rank-by-first-purchase baseline by 23.1% for DNN-ZILN against 10.6% for DNN-MSE. On KDD Cup 1998: Spearman 0.027 versus 0.020, normalized Gini 0.190 versus 0.184, decile MAPE 0.176 versus 0.210. The paper says practical horizons are 1, 2 or 3 years [FT] — [ZILN](https://arxiv.org/html/1912.07753)
- Li, Shao, Yang, Fang, Song 2022, "Billion-user Customer Lifetime Value Prediction: An Industrial-scale Solution from Kuaishou" (CIKM 2022; arXiv 2208.13358). Cited by Papers 2 and 4 (baseline "MDME" in both). ODMN predicts LTV at 30, 90, 180 and 365 days jointly with soft monotonic constraints across horizons; MDME splits the label distribution into sub-distributions and buckets and classifies into them; introduces the Mutual Gini metric [FT] — [Kuaishou](https://arxiv.org/html/2208.13358)
- Drachen et al. 2018, "To be or not to be... social: Incorporating simple social features in mobile game customer lifetime value predictions". Cited by Papers 2 and 4 and by Kuaishou as the "Two-stage" baseline: classify whether a user will pay (or churn), then regress value. It is a mobile-game paper [FT of citing papers] — [OptDist](https://arxiv.org/html/2408.08585)
- Ma et al. 2018, ESMM ("Entire space multi-task model", SIGIR 2018). Cited by Papers 2 and 4 as the "MTL-MSE" baseline: joint conversion and value estimation with MSE loss [FT of citing papers] — [OptDist](https://arxiv.org/html/2408.08585)
- Liu et al. 2024, MDAN ("Multi-distribution Adaptive Networks for LTV Prediction", PAKDD 2024). Cited by Papers 2 and 4 as a baseline: multi-class network over predefined LTV buckets plus per-bucket channel embeddings fused for regression [FT of citing papers] — [OptDist](https://arxiv.org/html/2408.08585)
- Fader, Hardie, Lee 2005, "RFM and CLV: Using Iso-Value Curves for Customer Base Analysis" (J. Marketing Research). Cited by Papers 3 and 4 as the RFM / BTYD foundation [FT of citing papers] — [Chen et al.](https://arxiv.org/html/2412.20295v1); [OptDist](https://arxiv.org/html/2408.08585)
- BTYD family more broadly: Schmittlein, Morrison, Colombo 1987 (Pareto/NBD, "Counting your customers"); Fader, Hardie, Lee 2005 (BG/NBD, "Counting your customers the easy way"); Fader and Hardie 2009 ("Probability models for customer-base analysis"); Colombo and Jiang 1999 (stochastic RFM); Bemmaor and Glady 2012 (flexible "sudden death" lifetime model). Cited by Paper 4 as "statistic-based" methods; Paper 3 cites the Fader line as BTYD and uses BTYD as a baseline [FT of citing papers] — [OptDist](https://arxiv.org/html/2408.08585); [Chen et al.](https://arxiv.org/html/2412.20295v1)
- Weng et al. 2024, OptDist (Paper 4) is itself cited by Paper 2, which adopts its LTV protocol and baselines [FT] — [GR v2](https://arxiv.org/html/2412.20211v2)
- Bauer and Jannach 2021 (Paper 1) is cited by Paper 4 as the RNN-plus-GBM sequential-behaviour approach [FT] — [OptDist](https://arxiv.org/html/2408.08585)
- Cho et al. 2014 (GRU / RNN encoder-decoder) is cited by both Papers 2 and 3 as the sequence-model building block [FT] — [GR v2](https://arxiv.org/html/2412.20211v2); [Chen et al.](https://arxiv.org/html/2412.20295v1)

Cited by one of the four but treated as foundational by that paper (and often also by ZILN or Kuaishou):

- Vanderveld et al. 2016, "An engagement-based customer lifetime value system for e-commerce" (KDD 2016, Groupon). Two-stage purchase-propensity then value model; cited by Paper 4 and by ZILN, which positions its single model as removing this two-stage complexity [FT of citing papers] — [OptDist](https://arxiv.org/html/2408.08585); [ZILN](https://arxiv.org/html/1912.07753)
- Chamberlain et al. 2017, "Customer Lifetime Value Prediction Using Embeddings" (KDD 2017, ASOS). Deployed system giving daily value estimates for every customer; handcrafted features plus learned customer embeddings; cited by Paper 4 and ZILN (which credits it for Spearman as a robust metric) [ABS; FT of citing papers] — [Chamberlain et al.](https://arxiv.org/abs/1703.02596)
- Xing et al. 2021, "Learning Reliable User Representations from Volatile and Sparse Data to Accurately Predict Customer Lifetime Value" (KDD 2021; "TSUR"). Wavelet transforms plus attention-based GRU on behaviour sequences; cited by Paper 4 and Kuaishou, both of which exclude it from baselines as a representation method compatible with their heads [FT of citing papers] — [OptDist](https://arxiv.org/html/2408.08585); [Kuaishou](https://arxiv.org/html/2208.13358)
- Chen, Guitart, del Rio, Perianez 2018, "Customer Lifetime Value in Video Games Using Deep Learning and Parametric Models" (IEEE Big Data 2018). Compares deep networks with Pareto/NBD on game data; finds CNNs on raw sequences most accurate and cheaper computationally; notes up to 50% of free-to-play revenue comes from about 2% of players. Cited by Paper 4 [ABS] — [Chen et al. 2018](https://arxiv.org/abs/1811.12799)
- Zhao et al. 2023, "perCLTV: A general system for personalized customer lifetime value prediction in online games" (ACM TOIS). Cited by Paper 4 as a value-based deep approach in games [FT of Paper 4, title only] — [OptDist](https://arxiv.org/html/2408.08585)
- Zhang et al. 2023, ExpLTV, "Out of the Box Thinking: Improving Customer Lifetime Value Modelling via Expert Routing and Game Whale Detection" (CIKM 2023). Multi-task framework for mobile games: a whale detector acts as the gating network over LTV experts, extending ZILN. Cited by Paper 4 [ABS; FT of Paper 4] — [ExpLTV](https://arxiv.org/abs/2308.12729)
- Yang et al. 2023, MarfNet (WSDM 2023; feature-missing-aware routing-and-fusion for advertising CLTV) and Su et al. 2023, CDAF (AAAI 2023; cross-domain adaptation for advertising CLTV). Cited by Paper 4 as representation modules pluggable under OptDist [FT of Paper 4] — [OptDist](https://arxiv.org/html/2408.08585)
- Chang et al. 2017, "Dilated Recurrent Neural Networks" (NeurIPS) and Smyl, Dudek, Pelka 2021, "ES-dRNN" (hybrid exponential smoothing and dilated RNN for load forecasting). The architectural basis of Paper 3; the dRNN cell is said to be more accurate than LSTM and GRU on forecasting tasks per the ES-dRNN paper [FT of Paper 3] — [Chen et al.](https://arxiv.org/html/2412.20295v1)
- Lin et al. 2023, TPM (tree-based progressive regression, KDD 2023) and Sun et al. 2024, CREAD (classification-restoration with error-adaptive discretisation, AAAI 2024). The ordinal-regression watch-time heads that Paper 2 positions itself against [FT of Paper 2] — [GR v2](https://arxiv.org/html/2412.20211v2)
- Jang et al. 2016 (Gumbel-softmax), Lin et al. 2017 (focal loss), Pham et al. 2021 (meta pseudo labels), Zhao et al. 2021 (AutoLoss), Liu et al. 2018 (DARTS). The machinery OptDist borrows for differentiable selection and alignment [FT of Paper 4] — [OptDist](https://arxiv.org/html/2408.08585)

### Inferences
- The standard baseline set in the 2022 to 2025 industrial LTV literature is: Two-stage (Drachen), MTL-MSE (ESMM), ZILN, MDME, MDAN, and now OptDist. Our plan already contains ZILN; a two-stage hurdle model is effectively what a separate P(spend > 0) head plus a conditional-amount head gives us.
- BTYD is the one "standard approach" named across these papers that our candidate list (R, A, B, C) does not include. In the one head-to-head I read (Paper 3) it was much the weakest (aSMAPE 53.6% versus 30.4% for LightGBM at 4 weeks).
- Games are well represented in the cited lineage (Drachen 2018, Chen 2018, perCLTV 2023, ExpLTV 2023), with whale handling as the recurring games-specific concern.

### Gaps
- Paper 1's reference list and related-work section were not available, so its contribution to the recurrence counts is missing.
- For Drachen 2018, ESMM, MDAN, perCLTV, TSUR, MarfNet and CDAF I rely on how the citing papers describe them; I did not read those papers.

## Does any paper survey or taxonomize LTV methods? If so, reproduce the taxonomy.

### Takeaway
None of the three papers I read is a survey, and I could not check Paper 1 (its 37-page length suggests a substantial related-work section, but that is unverified). OptDist gives the most complete implicit taxonomy of LTV methods; GR gives an explicit taxonomy of regression output heads.

### Cited Findings
- OptDist's implicit taxonomy of CLTV methods, assembled from its introduction and Section 2.1 [FT] — [OptDist](https://arxiv.org/html/2408.08585):
  1. Statistic-based (probabilistic) models: RFM, Pareto/NBD, BG/NBD and relatives. Criticised for using only history statistics, so users with the same recency and frequency get the same prediction, and for neglecting user attributes.
  2. Value-based ML/DL regression: networks that predict the CLTV value directly, mostly with MSE (examples named: Chen et al. 2018, perCLTV, TSUR, and ZILN is also listed in this sentence). Criticised as outlier-sensitive and unstable.
  3. Two-stage models: purchase likelihood then value (Vanderveld 2016, Drachen 2018).
  4. Representation-learning approaches: user embeddings (Chamberlain 2017, word2vec-style), sequential behaviour models (Bauer and Jannach 2021 RNN plus GBM; Xing 2021 wavelet plus attention GRU), missing-feature handling (MarfNet), cross-domain adaptation (CDAF). Declared orthogonal to distribution modelling.
  5. Distribution-modelling approaches, split in two: (a) prior parametric distribution with one set of parameters per sample: ZILN, and ExpLTV which extends ZILN with whale detection; (b) dividing samples into groups by label value with predefined buckets: MDME/ODMN, MDAN. OptDist adds (c) learned, adaptive sub-distribution selection.
- Paper 3's three-way split [FT] — [Chen et al.](https://arxiv.org/html/2412.20295v1):
  1. BTYD parametric generative models built on RFM (separate models for transaction frequency, amount and active duration with heterogeneous user-level parameters).
  2. Supervised feature-based ML with the value as target (regularised regression, random forest, gradient-boosted trees), needing explicit lag features.
  3. RNN sequence models (their proposal).
- GR's taxonomy of regression output paradigms (its Table 1) [FT] — [GR v2](https://arxiv.org/html/2412.20211v2):

| Method | Paradigm | Discretisation | Accuracy | Versatility | Inference complexity |
|---|---|---|---|---|---|
| VR (plain value regression) | Discriminative | none | Low | High | O(m) |
| TPM | Discriminative | Tree-based | Medium | Medium | O(mn) |
| CREAD | Discriminative | Linear | Medium | Medium | O(mn) |
| GR | Generative | Sequential | High | High | O(mdt) |

- In GR's related work, watch-time methods are further grouped as: value regression and debiasing variants (WLR, D2Q, CWM, D2Co); ordinal regression (CREAD, TPM); and sequence generation (their own) [FT] — [GR v2](https://arxiv.org/html/2412.20211v2)
- The Kuaishou paper describes existing approaches as those that "directly learn from posterior feature distributions" versus those that "leverage statistical models that make strong assumption on prior distributions" [ABS] — [Kuaishou](https://arxiv.org/abs/2208.13358)

### Inferences
- Merging these, the literature's working taxonomy has two independent axes: (i) the encoder / input representation (RFM statistics; engineered features with GBDT or MLP; learned embeddings; sequence models) and (ii) the output / loss (MSE; two-stage hurdle; parametric ZILN; mixtures or selections of parametric heads; bucketed or ordinal classification; autoregressive tokenised regression). The papers explicitly treat the two axes as combinable.
- Mapped to our plan: R and A sit at the feature end of axis (i), C at the sequence end, B is a latent-state generative model closest in spirit to the BTYD family; on axis (ii) the plan currently covers only ZILN and sigmoids.

### Gaps
- Whether Paper 1 contains a formal taxonomy could not be checked.
- I did not find, among these four, a dedicated survey paper on LTV modelling.

## Does any paper address long horizons (a year), zero-inflation and heavy tails, multi-task or multi-horizon outputs, or cost of scoring at very large scale?

### Takeaway
Zero-inflation and heavy tails are the central subject of Papers 2 and 4 (and of ZILN and MDME behind them). One-year horizons appear only through the Kaggle benchmark (Papers 2 and 4) and, more substantively, in the cited Kuaishou paper, which predicts 30/90/180/365-day active days jointly with monotonic constraints for 180M sampled new users. None of the four papers reports a scoring-throughput figure comparable to our 100M-in-30-minutes constraint; the only cost data points are GR's 10.2% QPS reduction and Kuaishou's statement that it scores all users daily in batch.

### Cited Findings
- Long horizon, Paper 4: the Kaggle task is total purchase value in the year after first purchase; the Industrial and Criteo-SSC tasks are 30 days [FT] — [OptDist](https://arxiv.org/html/2408.08585)
- Long horizon, Paper 3: longest horizon evaluated is 26 weeks; error rises steeply with horizon for every model (dRNN aSMAPE 22.9% at 4 weeks, 39.8% at 13, 51.2% at 26) [FT] — [Chen et al.](https://arxiv.org/html/2412.20295v1)
- Long horizon, cited work: ZILN says practical horizons are 1, 2 or 3 years and prefers a whole number of years "to avoid seasonal fluctuations" [FT] — [ZILN](https://arxiv.org/html/1912.07753)
- Long horizon and multi-horizon, cited work: Kuaishou's label is the number of active days of a new user in the next 30, 90, 180 and 365 days, predicted jointly from the first 7 days of behaviour, on a sample of 180 million new users. The business ordering ltv30 <= ltv90 <= ltv180 <= ltv365 is enforced softly by "Mono Units" that pass the shorter-horizon output distribution into the longer-horizon tower, plus an ordered-dependency calibration loss [FT] — [Kuaishou](https://arxiv.org/html/2208.13358)
- Kuaishou results at 365 days (NRMSE / NMAE / AMBE / Mutual Gini): Two-Stage 1.7507 / 1.1749 / 6.6276 / 0.0821; ZILN 1.2103 / 0.6833 / 8.6644 / 0.0558; ODMN 1.1538 / 0.5833 / 0.3123 / 0.0083. At 30 days: Two-Stage 0.8888 / 0.5768 / 0.3181 / 0.0273; ZILN 0.4831 / 0.2963 / 0.1336 / 0.0226; ODMN 0.4765 / 0.2683 / 0.0423 / 0.0125. ZILN's mean bias (AMBE) grows from 0.1336 at 30 days to 8.6644 at 365 days on this active-days label [FT] — [Kuaishou](https://arxiv.org/html/2208.13358)
- Kuaishou's authors report that short-horizon tasks help long-horizon ones ("compared with short-term LTV, long-term LTV has more space for improvement"), and that plain shared-bottom multi-task learning alone "has limited improvement" [FT] — [Kuaishou](https://arxiv.org/html/2208.13358)
- Kuaishou online A/B against ZILN (10% traffic each arm): ROI uplift 11.9% at 7 days, 12.8% at 14 days, 14.7% at 30 days [FT] — [Kuaishou](https://arxiv.org/html/2208.13358)
- Zero-inflation, Paper 4: positive rates are 7.20% (Criteo-SSC) and 6.35% (Industrial); every SDN is a ZILN head, so zero mass is handled by the per-head p; the selector's cross-entropy uses a focal weight because high-value users are rare [FT] — [OptDist](https://arxiv.org/html/2408.08585)
- Heavy tails, Paper 2: the vocabulary algorithm starts at the 99.99th percentile and peels the tail off first, which the authors say "expedites the reduction of tail values" and mitigates the long-tailed distribution; a Huber term adds robustness to outliers [FT] — [GR v2](https://arxiv.org/html/2412.20211v2)
- Sparsity, Paper 3: zero-transaction periods are subsampled and elapsed-time features added; the evaluation metric is floored at $1 to stop near-zero actuals dominating [FT] — [Chen et al.](https://arxiv.org/html/2412.20295v1)
- Multi-task / multi-horizon, Paper 3: one model is applied "for a variety of time horizons" (1, 4, 13, 26 weeks); the paper does not say whether horizons share one network with several outputs or are separate models [FT] — [Chen et al.](https://arxiv.org/html/2412.20295v1)
- Multi-task, Paper 4: single target (CLTV) with conversion probability as part of the ZILN head; no multi-horizon output [FT] — [OptDist](https://arxiv.org/html/2408.08585)
- Uncertainty, cited work: ZILN "provides a full prediction distribution"; uncertainty can be read from lognormal quantiles [FT] — [ZILN](https://arxiv.org/html/1912.07753)
- Calibration evaluation, cited work: ZILN recommends decile charts and decile-level MAPE; Kuaishou adds AMBE (mean bias) and Mutual Gini (area between the Lorenz curves of predictions and labels) and argues plain Gini "is not a good choice" for judging fit to an imbalanced distribution [FT] — [ZILN](https://arxiv.org/html/1912.07753); [Kuaishou](https://arxiv.org/html/2208.13358)
- Scoring cost, Paper 2: inference complexity O(mdt) versus O(m) for plain regression; 10.2% lower QPS than CREAD in production (v3) [FT] — [GR v2](https://arxiv.org/html/2412.20211v2); [GR v3 / latest](https://arxiv.org/html/2412.20211)
- Scoring cost, Paper 4: only the qualitative claim that SDNs are parallel simple MLPs and do not increase time complexity; experiments ran on one V100 [FT] — [OptDist](https://arxiv.org/html/2408.08585)
- Scoring cost, Paper 3: none reported; the only cost remark is that lag-feature engineering for tree models may "run into runtime performance issues" [FT] — [Chen et al.](https://arxiv.org/html/2412.20295v1)
- Scoring at scale, cited work: Kuaishou uses "day-level training and prediction of full volume users", then caches predictions for online lookup; no hardware or wall-clock figures [FT] — [Kuaishou](https://arxiv.org/html/2208.13358)
- Scoring at scale, cited work: ASOS's system "provides daily estimates of the future value of every customer" [ABS] — [Chamberlain et al.](https://arxiv.org/abs/1703.02596)

### Inferences
- The Kuaishou ODMN setup is the closest published analogue to our "expected active weeks over 52 weeks" target: an activity-count label at a one-year horizon, at hundreds of millions of users, with multi-horizon outputs. Its finding that ZILN's mean bias deteriorates badly at 365 days on a bounded count label is directly relevant: lognormal is a poor shape for a count bounded at 52, so ZILN should not be the head for active weeks (our plan's 52 weekly sigmoids already avoid this).
- Our planned 52 weekly activity sigmoids give a multi-horizon structure by construction (cumulative sums are monotone in horizon automatically). For hours and spend, where the plan has single ZILN heads, the ODMN idea suggests predicting cumulative value at several horizons (for example 4, 13, 26, 52 weeks) with a monotonicity constraint so short horizons assist the 52-week task.
- An autoregressive output decoder (GR) multiplies per-user head cost by the token sequence length, per target. With five targets and a hard 30-minute budget for 100M users this is the least throughput-friendly idea in the set; OptDist and ODMN-style heads are single-pass.
- No paper in this set addresses forecast revisions or lift measurement, interpretable journey states, or exogenous future covariates such as a season calendar (beyond Paper 3's general calendar-date dimension).

### Gaps
- No throughput numbers (users per second, hardware) for any method at 100M-user scale were found in these papers.
- No paper evaluates predictive-interval coverage or probability calibration (reliability curves, ECE) for P(active) or P(spend > 0).
- Whether Paper 1 addresses multi-step horizon outputs or runtime is unknown (full text not read).

## Does each paper propose something NOT already covered by candidates R, A, B, C?

### Takeaway
No paper introduces a whole new model family that the bake-off lacks; each contributes techniques that bolt onto candidates A or C. The items genuinely absent from the plan are: a GBM-plus-sequence stack (Paper 1), an autoregressive tokenised output head (Paper 2), dilated-RNN cells with zero-week subsampling and a floored SMAPE metric (Paper 3), and a multi-ZILN head with learned per-user selection (Paper 4), plus, from their shared references, a BTYD baseline and ODMN-style monotone multi-horizon heads.

### Cited Findings
- Paper 1 (Bauer and Jannach): the sequence model (encoder-decoder RNN) and the feature-based GBM are each covered by C and A. What is NOT in the plan is the hybrid: stacking the two (GBM-S2S), with purchase-log embeddings feeding both [SNIP] — [author-hosted PDF surfaced by search, not fetched](https://web-ainf.aau.at/pub/jannach/files/Journal_TKDD_2021.pdf)
- Paper 2 (GR): NOT covered. It is a new output head (tokenised autoregressive regression) that would replace a ZILN head; it beat ZILN and OptDist on MAE and Spearman on two public LTV sets, at an inference complexity of O(mdt) and a measured 10.2% QPS loss in production [FT] — [GR v2](https://arxiv.org/html/2412.20211v2); [GR v3 / latest](https://arxiv.org/html/2412.20211)
- Paper 3 (Chen et al.): the model family is covered by C (GRU on sequences). NOT in the plan: (a) dilated RNN with dRNN cells and residual blocks as an architecture variant; (b) subsampling zero-activity periods with a time-since-last-step feature; (c) aSMAPE as a user-level error metric robust to zeros; (d) BTYD as a baseline, which the paper includes and which loses to both LightGBM and the RNN [FT] — [Chen et al.](https://arxiv.org/html/2412.20295v1)
- Paper 4 (OptDist): ZILN is already in A and C. NOT in the plan: several ZILN heads with a Gumbel-softmax selector and loss-derived pseudo-label alignment. Reported gain over a single ZILN head is +1.6% to +5.5% normalized Gini and -2.2% to -24.4% MAE [FT; percentages CALC] — [OptDist](https://arxiv.org/html/2408.08585)
- From the shared references, NOT in the plan: ODMN's joint 30/90/180/365-day outputs with soft monotonic coupling, and the Mutual Gini and AMBE metrics [FT] — [Kuaishou](https://arxiv.org/html/2208.13358)
- From the shared references, NOT in the plan as far as the brief states: decile-level MAPE as the quantitative calibration measure alongside normalized Gini [FT] — [ZILN](https://arxiv.org/html/1912.07753)
- From the shared references, NOT in the plan: whale-aware expert routing (ExpLTV), where a high-spender detector gates a mixture of LTV experts [ABS] — [ExpLTV](https://arxiv.org/abs/2308.12729)

### Inferences
My assessment of what the decision-maker should do with each, given the 30-minute / 100M constraint and the five targets. These are judgements, not findings from the papers.

- Add as a cheap extra candidate: an A + C stack (C's outputs, or its pooled user embedding, as features to LightGBM, or a simple blend). Paper 1 and Paper 3 together suggest trees and sequence models capture different signal (the Uber RNN needed embeddings just to match XGBoost). Cost is one extra LightGBM pass. Confidence is limited because Paper 1's numbers were not read.
- Adopt as an ablation on C's spend and hours heads: OptDist-style multi-ZILN head (L of 4 or 5). It is single-pass and cheap at inference. Expect modest gains, and note it does not improve uncertainty output (one ZILN component is selected per user). It does not transfer directly to LightGBM (candidate A) because it relies on joint gradient training of selector and heads.
- Adopt as an ablation on C: multi-horizon cumulative heads with monotone coupling (ODMN idea) for hours and spend, since the plan currently has a single 52-week ZILN head for each and Kuaishou reports short horizons help the 365-day task.
- Adopt for throughput and input design: zero-week compression of the 104-week sequence with a gap feature (Paper 3). Worth testing mainly because it shortens sequences for the inactive majority; verify it does not hurt the reactivation predictions that depend on season timing.
- Treat as low priority: GR's generative head. It adds an autoregressive decode per target per user, has no calibration or uncertainty evidence, and its LTV evidence is on click-level and 90%-positive benchmarks. If tried at all, try it offline on the spend head only.
- Treat as low priority: dilated-RNN cell (Paper 3). It is an architecture variant inside candidate C; the paper does not isolate its benefit over a plain LSTM or GRU on LTV data.
- Add as a baseline in tier R, for credibility rather than expected performance: BTYD (BG/NBD or Pareto/NBD for active weeks, Gamma-Gamma for spend). It is the classical standard every one of these papers positions against, and it is a conceptual cousin of candidate B (latent alive/dead state without covariates), which makes it a useful reference point for the HMM. Expect it to lose.
- Adopt in the evaluation protocol: normalized Gini plus decile charts and decile-level MAPE (ZILN convention), Spearman on all users and on positives only (OptDist convention), mean bias (AMBE), and a zero-robust per-user error such as aSMAPE; report MAE and rank metrics together because bucketed heads can win one and lose the other.
- Nothing in these four papers helps with three of our stated wants: interpretable journey states (only candidate B addresses this), forecast revisions for lift measurement, and explicit handling of a known future season calendar.

### Gaps
- Because Paper 1's full text was unavailable, I cannot say how large the stacking gain was, nor whether its decoder design (possibly per-period outputs with temporal features) offers anything for our 52 weekly activity heads beyond what C already plans. This is the main unresolved item; it needs someone with ACM access or the author PDF (web-ainf.aau.at would have to be allowed in the session's network settings).
- No evidence in this set compares an HMM (candidate B) against any of these methods.
- No evidence in this set on self-supervised pretraining for LTV sequence models.
