\# Hyperparameter Tuning Analysis



\## 1. Objective



The objective of this experiment was to develop a baseline machine learning

model and compare systematic hyperparameter tuning using Grid Search and

Random Search.



The models were evaluated using 5-fold cross-validation with macro F1 score

and test-set accuracy.



MLflow was used to track the tuning experiments and their results.



\---



\## 2. Baseline Model



The baseline model was a Decision Tree Classifier.



\### Results



| Metric | Result |

|---|---:|

| CV F1 Macro | 0.9663 |

| CV Standard Deviation | 0.0316 |

| Test Accuracy | 0.9000 |

| Total Fits | 5 |



The baseline model achieved a test accuracy of 90%.



This baseline provides a reference point for evaluating the tuned Random

Forest models.



\---



\## 3. Grid Search



Grid Search was performed using a Random Forest Classifier.



The following hyperparameters were searched:



\- n\_estimators: 50, 100, 200

\- max\_depth: 3, 5, 10, None

\- min\_samples\_split: 2, 5, 10

\- max\_features: sqrt, log2



The total number of combinations was:



3 × 4 × 3 × 2 = 72 combinations



With 5-fold cross-validation:



72 × 5 = 360 total fits



\### Best Parameters



```text

max\\\_depth = 3

max\\\_features = sqrt

min\\\_samples\\\_split = 2

n\\\_estimators = 50


