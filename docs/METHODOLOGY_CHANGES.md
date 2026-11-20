# Corrections and validation scope

Original notebook: `Machine Learning/Churn_Modeling.ipynb`. Cell ordinals below include all cells from one. Original files were not edited.

| Original behavior | Corrected publication behavior |
|---|---|
| Cell 3 already drops `#`, `CustomerId`, and `Surname` | Preserve identifier exclusion; do not claim a nonexistent repair of ID leakage |
| Cells 3, 5 and 16 learn imputation/scaling on the full frame before splitting | Split first; fit all learned preprocessing inside every GridSearchCV fold and only on training data for the final model |
| Conditional lowercase column checks do not match the original capitalized column names | Use explicit `Geography`/`Gender` contracts and a one-hot categorical pipeline |
| `NumOfProducts_int` is constructed after scaling and can enter a later frame | Select only the original ten predictors; retain raw units for any descriptive checks |
| Logistic grid contains invalid penalty/solver combinations; warnings are suppressed | Use a small valid liblinear/L2 grid and raise on fitting errors |
| A large unfinished tree grid includes obsolete `max_features='auto'` | Complete a bounded twelve-configuration comparison without that option |
| Hard-label average precision is called “AUC” | Compute average precision and ROC-AUC separately from positive-class probabilities |
| Accuracy is the main search metric despite minority-class recall concerns | Select by training-fold probability-based AP; report recall, precision, F1, balanced accuracy and a prior baseline at a fixed 0.5 threshold |

No held-out rows are used to fit preprocessing, choose a model configuration, or adjust a decision threshold. Both model results are reported; selecting a more favorable claim after inspecting the holdout would still be inappropriate. The current analysis is a corrected benchmark exercise, not an unbiased new external validation of the historical notebook.

Tests check training-only scaler behavior under extreme held-out values and unseen categories, distinguish probability AP from label AP, and enforce identifier/artifact exclusion. Historical saved numbers are retained only as clearly labeled context in the README.
