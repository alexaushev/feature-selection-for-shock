# Experiments

# Overview

The repo has two scripts. The first one preprocessing.r takes the data.xlsx file and generates 100 test cases for three types of datasets: T1, T1+T2 and Full. The script performs data preprocessing and imputation steps.

The second script feature_selection.py is used to generate features of different size and to choose the best among them for all 100 generated test cases. It is required to run the preprocessing.r first so it generates the test cases.

# Requirements

To run the scripts it is necessary to put the data.xlsx file in the \experiments folder and to install the following:

# preprocessing.r

R (v3.3.3):
    
missForest >= 1.4
xlsx >= 0.6.1
plyr >= 1.8.4

# feature_selection.py

Python (v2.7):
    
pandas >= 0.23.4
pytz >= 2018.7
python_dateutil >= 2.7.5
numpy >= 1.15.3
six >= 1.11.0
scikit_learn >= 0.20.0
scipy >= 0.13.3
xlrd >= 1.1.0
