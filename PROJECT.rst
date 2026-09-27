Electric vehicle purchase prediction
************************************

This repository contains my solution for Kaggle Playground Series S6E9. The
training set has 668,665 rows and the test set has 286,571 rows. ROC-AUC is the
evaluation metric.

The main experiment is in ``notebooks/01_ev_purchase_project.ipynb``. Reusable
training code is under ``src/evpurchase``; saved fold scores and feature
importance tables are under ``reports``.

Result
------

The final rank ensemble reached 0.946314 out-of-fold ROC-AUC and 0.94641 on the
public leaderboard. The same-day Top-10% cutoff was about 0.94650.

The strongest individual model was a shallow 20-fold LightGBM model at
0.946186 OOF AUC. The ensemble also uses two 10-fold LightGBM runs, the original
5-fold LightGBM run, a feature-cross LightGBM model, and XGBoost. Exact weights
are saved in ``reports/ensemble_metrics.json``.

Implementation
--------------

The feature pipeline includes frequency encoding, numerical digit and bin
features, a small set of interaction keys, and smoothed target encodings. Target
encoding is fitted inside each outer fold; training-row values are produced by
an inner cross-validation split.

Adversarial validation produced an AUC of 0.50035. Submission validation checks
the row count, ID order, probability range, and SHA-256 checksum. Five small
tests cover target conversion and feature-generation behavior.

Reproduction
------------

The project uses Python 3.12. Data files are not included; their expected names
and sources are listed in ``data/DATA_SOURCES.rst``.

::

   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -e .
   python scripts/reproduce.py

The full run trains every recorded model and can take more than one hour on a
laptop. For a shorter inspection, open the executed notebook and the JSON/CSV
files in ``reports``.

Competition data, intermediate NumPy arrays, and the submitted prediction file
are excluded from Git. The published manifest records the submitted file's
checksum and public score. Data and method references are listed in
``SOURCES.rst``.
