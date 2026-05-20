Deep Learning realized volatility; comparing TCN and HAR for realized volatility forecasts. 

This code backtests both the statistical HAR model (Corsi, 2009) and a deep learning TCN model (Bai et al, 2018) on the Oxford-Man realized volatility index data. 
Backtest evaluation is done using Quasi-Likelihood loss function (Patton, 2011), for three separate horizons: day ahead, week ahead, month ahead. For each horizon, a separate HAR model is trained, whereas a single TCN model will be trained on all three horizons. Additionally, the feature set is extended for the TCN model.  

The main hypothesis of this project is that the longer the horizon, the better Deep Learning's relative performance, due to having information on the horizon interactions.     

Work in progress. 




## Installation

_TODO_

## Data

- Source: Oxford-Man Realized Library.
- 129,209 daily observations across 31 symbols.
- Period: 2001-01-03 – 2018-06-26.

## Methodology

### Targets

For each $t$:

1. `rv5_ss` at $t+1$ (day ahead)
2. Mean of `rv5_ss` over $[t+1, \dots, t+5]$ (week ahead)
3. Mean of `rv5_ss` over $[t+1, \dots, t+22]$ (month ahead)

### Features

**HAR (per-horizon model):**

1. `rv5_ss` at $t$ — subsampled realized variance (5-min)
2. Mean of `rv5_ss` over $[t-4, \dots, t]$ — weekly
3. Mean of `rv5_ss` over $[t-21, \dots, t]$ — monthly

**TCN (single multi-horizon model):** HAR features plus

4. `rsv_ss` at $t$ — subsampled realized semi-variance (negative returns)
5. `jump` at $t$ — $\max(0,\ \text{rv5\_ss} - \text{bv\_ss})$, where `bv_ss` is subsampled bipower variation
6. `open_to_close` at $t$ — daily return
7. Symbol embedding

### Transformations

- Features (except returns) are log-transformed after aggregation; `jump` uses `log1p`.
- Features are z-normalized per symbol using a causal rolling window capped at 252 days. For samples with fewer than 252 prior observations, an expanding window is used.
- Targets are log-transformed.
- Targets are not symbol-normalized. This may introduce mild per-symbol bias but keeps the loss definition clean and scale-equivariant in variance.

### Models

- **HAR:** one OLS model per horizon.
- **TCN:** a single network with a 3-headed output layer producing un-normalized $\log \hat\sigma^2$ per horizon. The symbol embedding is added as a bias at the output, with the TCN network modelling per-symbol residuals.
- Some symbol asymmetry and agnosticism remains; this is deemed acceptable given the expected cross-sectional correlation of volatility.

### Train / validation / test split

| Split | Period      | Purpose                      |
| ----- | ----------- | ---------------------------- |
| Train | 2001 – 2013 | Model fitting                |
| Val   | 2014 – 2015 | Early stopping               |
| Test  | 2016 – 2018 | Backtesting                  |

Splitting is performed before normalization. A 22-day gap between splits prevents target leakage.

### Loss function

QLIKE in log-variance parameterization is used for both training and backtesting:

$$
L = \exp(y - \hat y) - (y - \hat y) - 1, \qquad y = \log \sigma^2
$$

Algebraically equivalent to Patton's QLIKE on variance: robust to noise in the volatility proxy, asymmetric (under-prediction penalized more than over-prediction), and using the same form for training and evaluation eliminates train/eval mismatch.

Per-head loss weighting may be required to prevent the higher-variance day-ahead head from dominating training; to be investigated. Backtesting is reported per horizon.

## References

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271.
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial Econometrics, 7(2), 174–196.
- Patton, A. J. (2011). *Volatility forecast comparison using imperfect volatility proxies.* Journal of Econometrics, 160(1), 246–256.
- Gerd Heber, Asger Lunde, Neil Shephard, and Kevin Sheppard (2009). *Oxford-Man Institute's Realized Library, version 0.3.* Oxford-Man Institute, University of Oxford.



## Methodology
The data consists of 129.209 rows, spanning 31 symbols over the period 2001-01-03 up to 2018-06-26. 

Targets:
    1. 'rv5_ss' at t + 1
    2. 'rv5_ss' mean over [t+1, .., t+5]
    3. 'rv5_ss' mean over [t+1, .., t+22]

Features for the HAR model are fixed to 
    1. 'rv5_ss' at t: subsampled realized volatility over 5 min
    2. 'rv5_ss' mean over [t-4, .., t]: weekly mean
    3. 'rv5_ss' mean over [t-21, .., t]: monthly mean
For the TCN model, these are complemented by
    3. 'rv5_ss' mean over [t-21, .., t]: monthly mean
    4. 'rsv_ss' at t: subsampled realized semi-variance (negative returns only) 
    5. jump at t: floor of rv5_ss - bv_ss (subsampled bipower variation)
    6. 'open_to_close' at t: returns
    7. Embedding layer for symbols

- All features are log-transformed (except returns), after taking the mean. Jump is log1p transformed instead. 
- All features are z-normalized after log-transforming, using rolling normalization per symbol, capped at 252 days. For samples with less than the full 252 ticks prior, expanding normalization is used instead (i.e., everything available is used).
- All targets are log-transformed
- Targets are not normalized symbol. This may introduce mild symbol-biases, but ensures the loss definition remains clean.

- The symbol layer is added as a bias to the output. Some symbol asymmetry and agnosticism remains. This is deemed acceptable due to cross-correlation of volatility. 
- For each horizon, a separate HAR model is trained. There is only a single TCN model  with 3-headed output layer
- train/val/test split: [2000-2014], [2014-2015], [2016-2018]. Splitting is performed before normalization. Gaps of 22 days between each set ensures no target leakage. The validation set is used for early-stopping to avoid overfitting. 
- QLIKE loss (in log-space) is used both for backtest evaluation and for DL-loss function: 
L = exp(y - y_hat) - (y - y_hat) - 1. 
Per-head weighting of the loss function might be needed in training to ensure the loss is not dominated by the higher variance of point-in-time volatility; to be investigated. Backtesting will be done on a per-horizon basis. 