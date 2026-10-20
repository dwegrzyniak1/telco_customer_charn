# Telco customer churn

A learning project for exploratory analysis and, later, model experiments.

## Structure
### EDA
- `notebooks/01_eda.ipynb`: exploration, analysis steps, and conclusions.
- `src/eda/data_preparation.py`: loading, type conversion, and missing-charge handling.
- `src/eda/summaries.py`: reusable summary tables.
- `src/eda/visualization.py`: reusable plots.
- `data/`: input CSV.

## Analysis assumption

* The 11 missing `TotalCharges` values occur at zero tenure. The current EDA fills
these with zero, assuming new customers have no accumulated charges. This is an
explicit assumption, rather than a fact established by the dataset.
