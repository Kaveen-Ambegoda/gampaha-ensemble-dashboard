# Ensemble Learning Dashboard: Next-Day Temperature in Gampaha, Sri Lanka

Machine Learning & Pattern Recognition: hands-on project.
**Student:**224010J A.L.K.H Ambegoda

> **AI disclosure:** The notebook (`ensemble_gampaha_weather.ipynb`), the dashboard (`app.py`) and this README were **generated with Claude (Anthropic)**. [Add one line on how you ran and checked them.]

## What this project does
Forecasts **tomorrow's mean temperature in Gampaha** from daily weather (2015-2025) and compares the ensemble families from the lecture:

| Family | Models |
|---|---|
| Bagging | Random Forest, Bagged Trees |
| Boosting | Gradient Boosting, XGBoost, LightGBM, CatBoost |
| Stacking | RF + XGBoost + CatBoost + ANN, combined by a Ridge meta-model |
| Hybrid | ARIMA + Random Forest (on ARIMA's residuals), RF + ANN average |
| Baselines | "Tomorrow = today", ARIMA alone |

Models are trained on 2015-2023 and tested on 2024-2025 (a time-based split, so the future is never seen during training). Metrics: MAE, RMSE, R2.

**Results and interpretation:** see the notebook, sections 9-12. [Paste your headline result here, e.g. "Best model: X, RMSE 0.xx C, y% better than the baseline."]

## Files
| File | Purpose |
|---|---|
| `ensemble_gampaha_weather.ipynb` | Data download, features, all models, comparison, interpretation; saves `artifacts/` |
| `app.py` | Streamlit dashboard (front end) that loads the pickled models |
| `artifacts/` | `models.joblib` (pickled models), `metrics.csv`, `features.csv`, `test_predictions.csv`, `importances.csv` |
| `requirements.txt` | Python packages for the dashboard |

## Run it yourself
1. Open the notebook in Google Colab (or VS Code) and choose **Run all**. It downloads the data and takes a few minutes.
2. Put the generated `artifacts/` folder next to `app.py`.
3. Locally: `pip install -r requirements.txt` then `streamlit run app.py`.

## Deploy (Streamlit Community Cloud)
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, the notebook and the `artifacts/` folder.
2. Replace `requirements.txt` with `artifacts/requirements.txt` (the notebook writes it with the exact package versions used for training, so the pickled models load correctly).
3. Go to share.streamlit.io, sign in with GitHub, choose the repository, set the main file to `app.py` and deploy.

## Data and credits
Daily weather from the [Open-Meteo Historical Weather API](https://open-meteo.com/) (ERA5 reanalysis, CC BY 4.0). Reanalysis is a model-based estimate, not a weather-station record.

## Limitations
One location; two years of test data, so differences of a few hundredths of a degree between top models are not meaningful; hyperparameters were fixed rather than tuned.
