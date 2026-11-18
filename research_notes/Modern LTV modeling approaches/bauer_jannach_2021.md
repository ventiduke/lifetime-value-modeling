# Bauer and Jannach 2021, read in full

Source: Josef Bauer and Dietmar Jannach, "Improved Customer Lifetime Value Prediction With
Sequence-To-Sequence Learning and Feature-Based Models", ACM Transactions on Knowledge Discovery
from Data 15(5), Article 80, April 2021, DOI 10.1145/3441444. 37 pages.

Provenance: read from the PDF supplied by Alex on 2026-11-18 (pages 1 to 32: all sections through
the conclusions; the appendix and reference list were not read). This replaces the snippet-only
account in `seed_papers.md`. The PDF itself is not committed, because it is a copyrighted ACM
article.

## What the paper does

- **Task.** Predict the profit a customer generates over a fixed future window, as a proxy for
  lifetime value. Churn is not observed and is modelled implicitly (a forecast of zero).
- **Data.**
  - *Children*: a European e-commerce company, 1,081,443 customers (1,073,872 after denoising),
    about 10.6M purchase records, more than three years, aggregated to weeks. Horizon 6 months
    (26 weekly outputs), input sequence 52 weeks. The last six months are the test period.
  - *UKRetail*: the public UCI online-retail set, 4,373 customers (3,284 after denoising), about one
    year, daily data, horizon 4 weeks.
  - Denoising removes wholesale customers: the top 0.1% and bottom 0.1% by sales on Children; the
    top 20% and bottom 1% on UKRetail.
- **Models.**
  - *GBM*: XGBoost on a wide set of time-dependent aggregate features (719 on Children), with one
    training row per customer per point in history, not one per customer.
  - *S2S*: two stacked temporal convolutions (monthly and quarterly smoothing), concatenated with
    the raw input and a pretrained customer embedding, feeding two GRU encoder layers and a GRU
    decoder with a time-distributed dense output. It predicts profit per future time step; the
    forecast is the sum. The aggregate features are also fed to this model as time series.
  - *GBM-S2S*: a second-level GBM that takes the original features plus the predictions of the
    first GBM and the S2S model. It is trained on predictions for the validation period and
    evaluated on the test period.
- **Training data.** A sliding window creates many input and output pairs per customer by stepping
  the window back in time. Tensors are generated on the fly. To fit in memory, the model trains on
  one disjoint subset of customers at a time and stores its weights between subsets (their
  Algorithm 1). They note that 1M customers, 52 input weeks, 26 output weeks, 10 sequence features
  and 128 training weeks would need over 100 GB if materialized.
- **Baselines.** Adjusted customer mean (past profit scaled to the horizon, which is a persistence
  baseline), ARMA variants, BG/NBD, a Markov-chain model on recency and frequency states, and an
  embeddings-plus-random-forest model.

## Results (their Table 5, single time split)

| Model | UKRetail RMSE | UKRetail MAE | Children RMSE | Children MAE |
|---|---|---|---|---|
| Adjusted customer mean (and ARMA) | 72.66 | 49.74 | 21.08 | 11.93 |
| BG/NBD | 69.17 | 41.54 | 20.14 | 8.09 |
| Markov chain | 68.84 | 45.13 | 18.09 | 7.69 |
| Embeddings-RF | 68.82 | 39.48 | 17.48 | 6.86 |
| GBM | 66.26 | 33.78 | 12.50 | 3.42 |
| S2S | 65.36 | 34.55 | 12.16 | 3.91 |
| GBM-S2S stack | 64.95 | 30.80 | 11.65 | 3.14 |

- On Children the stack beats both components with p < 1e-5 (bootstrap test). On UKRetail the
  stack's gain over its components is not significant (p < 0.2).
- GBM and S2S split the metrics: S2S has the lower RMSE, GBM the lower MAE.
- My arithmetic from the table, Children: the stack is 4.2% below the best single model on RMSE
  (11.65 against 12.16) and 8.4% below on MAE (3.14 against 3.42).
- A moving-window evaluation with three time folds (their Table 6) keeps the ordering of their
  three models, with higher errors.
- ARMA, ARIMA and SARIMA did not beat the adjusted customer mean.

## Other findings

- **Output granularity.** Predicting weekly, 4-weekly, 13-weekly or one 26-week total with the S2S
  model changed RMSE by less than 0.15 on Children (less than 0.2 on UKRetail).
- **Horizon.** Errors grow with horizon (4, 6, 8 months on Children) and the ordering of methods
  stays mostly the same.
- **Ablation of the S2S model** (their Table 9; 1.0 = full model, 0.0 = the embeddings-RF
  baseline, averaged over both datasets): without embeddings 0.84; simplified architecture (one GRU
  encoder, no temporal convolutions) 0.61; only basic recency, frequency and monetary features 0.42.
  The engineered features contribute most, the architecture second, the embeddings least.
- **Feature importance (GBM, Children).** Total profit, number of orders and total revenue lead,
  followed by days since last order and gap statistics between orders.
- **Not reported.** No ranking metric (Gini), no calibration, no intervals, no scoring time, and no
  hardware description beyond the memory argument.

## What it means for the bake-off

These are my judgements, not findings of the paper.

- **No new candidate.** The paper's parts are candidate A (trees on features) and candidate C (a
  GRU sequence model). Its baselines match R2 (persistence) and R3 (buy-till-you-die).
- **Stack stays a diagnostic, with a defined method.** The evidence for a gain is now full-text
  and significant on the large dataset. The plan fits a second-level model on validation-origin
  predictions and scores it at the test origins. It stays out of the candidate list because a
  production stack means maintaining two models.
- **Aggregate features go into the sequence models by default.** The ablation makes this the
  best-evidenced input change, and it costs nothing at scoring time.
- **Temporal convolutions fit the CPU constraint.** A strided convolution that pools weeks into
  4-week steps shortens the sequence fourfold, which is where CPU cost comes from. The paper uses
  convolutions for smoothing, not for shortening; using them to cut steps is my adaptation.
- **Cumulative-horizon heads should not cost accuracy**, given the granularity result.
- **Shard-by-shard training matches the run limits.** Their subset-at-a-time loop is the same
  pattern the plan uses for 16 GB memory and 1-hour commands.
- **Limits of transfer.** The target is profit only, the horizon is 26 weeks, the data is retail
  purchases with no seasons or engagement signal, and wholesale outliers were removed before
  evaluation. None of the five bake-off targets other than spend is tested.
