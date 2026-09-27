Data sources
************

Competition files
-----------------

The canonical source is Kaggle Playground Series S6E9:

https://www.kaggle.com/competitions/playground-series-s6e9

For this run, the three competition files were retrieved from the public mirror
below after the Kaggle download endpoint rejected the active browser session:

https://github.com/dbwf923/Predicting-Electric-Vehicle-Purchases

The files were checked against the mirror's Git object hashes.

=======================  =======  =======  ========================================
File                       Rows   Columns  Git blob SHA-1
=======================  =======  =======  ========================================
train.csv                 668665       15  95c0ebe6e0fd2ab351675485c3cbe680ab7d7e14
test.csv                  286571       14  c8befb9b9b15e47afcf7a2c934db5b71f029e119
sample_submission.csv    286571        2  5646f411c5e94a32dd09d98ec362c956ec5f0ed6
=======================  =======  =======  ========================================

Local checks cover schemas, row counts, ID uniqueness, target values, missing
values, and test/sample-submission ID order. Raw files are excluded from Git.

To download the files from Kaggle after accepting the competition rules:

::

   kaggle competitions download -c playground-series-s6e9 -p data/raw

Source dataset
--------------

The target-mean lookup features use the 10,000-row ``EV Adoption Behavior and
Range Anxiety`` dataset:

https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety

Its CSV is stored locally as ``data/raw/original.csv``. The file used for this
run had SHA-256
``c271a380df51b18177d0a039d54525ca7c3d71500701dc7496714a091df16fae``.
Competition validation labels are not used to build these lookup values.
