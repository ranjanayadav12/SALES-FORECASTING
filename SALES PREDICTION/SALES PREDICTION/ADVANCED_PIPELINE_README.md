# Advanced Forecasting Pipeline

## 1) Install dependencies

```bash
pip install pandas numpy scikit-learn
pip install statsmodels prophet
```

`statsmodels` enables ARIMA and `prophet` enables Prophet model comparison.
If these are not installed, the pipeline automatically falls back to the ML model.

## 2) Run from terminal

```bash
python advanced_forecasting_pipeline.py --input "c:\Users\Ranjana Yadav\OneDrive\Documents\Data Science\train.csv" --output-dir outputs --horizon 6 --top-products 20
```

## 3) Output files

- `outputs/advanced_forecast_results.json`
- `outputs/model_leaderboard.csv`
- `outputs/forecast_output.csv`

## 4) Run from Flask API

`POST /api/advanced_forecast/run`

Example JSON body:

```json
{
  "input_path": "c:\\Users\\Ranjana Yadav\\OneDrive\\Documents\\Data Science\\train.csv",
  "horizon": 6,
  "top_products": 20,
  "output_dir": "outputs"
}
```

Requires user login session.
