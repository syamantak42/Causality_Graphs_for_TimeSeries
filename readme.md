## Sparse Causality Models for Time Series

This repository contains code and notebooks for learning and evaluating **sparse causality graphs** and **predictive models** on multivariate time series data. Several models with built-in sparsity constraints are included for inferring interpretable causal structure.

### Contents

* **`sparse_models_V2.py`**
  Generates and saves causality graphs from multivariate time series using a variety of models, including sparse regression-based methods.

* **`sparse_prediction_timeseries.py`**
  Trains and evaluates one-step-ahead time series predictors using learned sparse (causal) graph structures.

* **`sparse_reg_models/`**
  Contains implementations of various regression models used for learning sparse causality graphs.

* **Notebooks**
  Jupyter notebooks for:

  * Collecting and preprocessing real or synthetic time series data
  * Running experiments
  * Visualizing and analyzing results

### References

This implementation is based on the following works:

* **Inferring Causality in Networks of WSS Processes by Pairwise Estimation Methods**
  *Syamantak Datta Gupta, Ravi R Mazumdar*
  ITA Workshop, 2013.

* **A Frequency Domain Lasso Approach for Detecting Interdependence Relations among Time Series**
  *Syamantak Datta Gupta, Ravi Mazumdar*
  Proceedings of ITISE 2014.
