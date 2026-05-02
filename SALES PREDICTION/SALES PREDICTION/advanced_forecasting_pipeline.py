import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")


def try_import_prophet():
    try:
        from prophet import Prophet  # type: ignore
        return Prophet
    except Exception:
        return None


def try_import_arima():
    try:
        from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        return ARIMA
    except Exception:
        return None


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return (series < low) | (series > high)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["month"] = out["date"].dt.month
    out["quarter"] = out["date"].dt.quarter
    out["year"] = out["date"].dt.year
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_lag_features(df: pd.DataFrame, lags: List[int], windows: List[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out["sales"].shift(lag)
    for w in windows:
        out[f"roll_mean_{w}"] = out["sales"].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out["sales"].shift(1).rolling(w).std()
    out["trend_idx"] = np.arange(len(out))
    return out


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


@dataclass
class EvalResult:
    name: str
    mae: float
    rmse: float
    r2: float
    mape: float
    smape: float


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> EvalResult:
    eps = 1e-9
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)
    s_mape = smape(y_true, y_pred)
    return EvalResult(name=model_name, mae=mae, rmse=rmse, r2=r2, mape=mape, smape=s_mape)


def confidence_interval_from_residuals(pred: np.ndarray, residuals: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray]:
    sigma = float(np.std(residuals)) if len(residuals) > 1 else 0.0
    lower = pred - z * sigma
    upper = pred + z * sigma
    return lower, upper


class AdvancedSalesForecaster:
    def __init__(self, csv_path: Path, output_dir: Path, horizon: int = 6, top_products: int = 20):
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.horizon = horizon
        self.top_products = top_products
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.prophet_cls = try_import_prophet()
        self.arima_cls = try_import_arima()

    def load_and_validate(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        required = ["Order Date", "Product ID", "Category", "Sales"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        out = df[["Order Date", "Product ID", "Category", "Sales", "Region"]].copy() if "Region" in df.columns else df[["Order Date", "Product ID", "Category", "Sales"]].copy()
        out = out.rename(
            columns={
                "Order Date": "date",
                "Product ID": "product_id",
                "Category": "category",
                "Sales": "sales",
                "Region": "region",
            }
        )
        out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
        out["sales"] = pd.to_numeric(out["sales"], errors="coerce")
        out = out.dropna(subset=["date", "sales", "product_id", "category"])
        out = out[out["sales"] >= 0]

        out["month"] = out["date"].dt.to_period("M").dt.to_timestamp()
        return out

    def preprocess_monthly(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        monthly = (
            df.groupby([group_col, "month"], as_index=False)["sales"]
            .sum()
            .sort_values([group_col, "month"])
        )

        filled = []
        for key, grp in monthly.groupby(group_col):
            idx = pd.date_range(grp["month"].min(), grp["month"].max(), freq="MS")
            g = grp.set_index("month").reindex(idx).rename_axis("month").reset_index()
            g[group_col] = key
            g["sales"] = g["sales"].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill").fillna(0.0)
            outliers = detect_outliers_iqr(g["sales"])
            g.loc[outliers, "sales"] = g["sales"].median()
            filled.append(g[[group_col, "month", "sales"]])
        return pd.concat(filled, ignore_index=True)

    def _evaluate_ml_cv(
        self, grp: pd.DataFrame, model_name: str = "lstm_fallback_rf"
    ) -> Optional[Tuple[EvalResult, RandomForestRegressor, pd.DataFrame, List[str]]]:
        g = grp.rename(columns={"month": "date"}).copy().sort_values("date")
        g = add_time_features(g)
        g = add_lag_features(g, lags=[1, 2, 3, 6, 12], windows=[3, 6])
        g = g.dropna().reset_index(drop=True)
        if len(g) < 18:
            return None

        feature_cols = [
            c for c in g.columns if c not in {"date", "sales"} and pd.api.types.is_numeric_dtype(g[c])
        ]
        X = g[feature_cols]
        y = g["sales"].values

        splits = min(4, max(2, len(g) // 8))
        tscv = TimeSeriesSplit(n_splits=splits)

        y_true_all: List[float] = []
        y_pred_all: List[float] = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = RandomForestRegressor(n_estimators=300, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(preds.tolist())

        final_model = RandomForestRegressor(n_estimators=300, random_state=42)
        final_model.fit(X, y)
        metrics = compute_metrics(np.array(y_true_all), np.array(y_pred_all), model_name)
        return metrics, final_model, g, feature_cols

    def _evaluate_arima_cv(self, grp: pd.DataFrame) -> Optional[Tuple[EvalResult, object]]:
        if self.arima_cls is None:
            return None
        series = grp.sort_values("month")["sales"].values
        if len(series) < 18:
            return None
        preds = []
        trues = []
        split_points = [int(len(series) * 0.7), int(len(series) * 0.8), int(len(series) * 0.9)]
        split_points = [p for p in split_points if p >= 12 and p < len(series)]
        if not split_points:
            return None
        for p in split_points:
            train = series[:p]
            test = series[p]
            try:
                m = self.arima_cls(train, order=(1, 1, 1)).fit()
                pred = float(m.forecast(1)[0])
                preds.append(pred)
                trues.append(float(test))
            except Exception:
                return None
        fitted = self.arima_cls(series, order=(1, 1, 1)).fit()
        metrics = compute_metrics(np.array(trues), np.array(preds), "arima")
        return metrics, fitted

    def _evaluate_prophet_cv(self, grp: pd.DataFrame) -> Optional[Tuple[EvalResult, object]]:
        if self.prophet_cls is None:
            return None
        g = grp.rename(columns={"month": "ds", "sales": "y"}).sort_values("ds")
        if len(g) < 24:
            return None
        preds = []
        trues = []
        split_points = [int(len(g) * 0.7), int(len(g) * 0.8), int(len(g) * 0.9)]
        split_points = [p for p in split_points if p >= 18 and p < len(g)]
        if not split_points:
            return None
        for p in split_points:
            train = g.iloc[:p]
            test = g.iloc[p : p + 1]
            try:
                model = self.prophet_cls(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                model.fit(train)
                fc = model.predict(test[["ds"]])
                preds.append(float(fc["yhat"].iloc[0]))
                trues.append(float(test["y"].iloc[0]))
            except Exception:
                return None
        final_model = self.prophet_cls(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        final_model.fit(g)
        metrics = compute_metrics(np.array(trues), np.array(preds), "prophet")
        return metrics, final_model

    def _choose_best(self, metrics: List[EvalResult]) -> EvalResult:
        # Weighted score: lower is better
        def score(m: EvalResult) -> float:
            return 0.45 * m.rmse + 0.30 * m.mae + 0.25 * m.mape

        return min(metrics, key=score)

    def _forecast_ml_recursive(self, trained: RandomForestRegressor, history_df: pd.DataFrame, horizon: int) -> np.ndarray:
        g = history_df.copy().sort_values("date")
        preds = []
        for _ in range(horizon):
            next_date = g["date"].max() + pd.offsets.MonthBegin(1)
            temp = pd.concat(
                [g[["date", "sales"]], pd.DataFrame([{"date": next_date, "sales": np.nan}])],
                ignore_index=True,
            )
            temp = add_time_features(temp)
            temp = add_lag_features(temp, lags=[1, 2, 3, 6, 12], windows=[3, 6])
            row = temp.iloc[-1:]
            feature_cols = [c for c in row.columns if c not in {"date", "sales"}]
            pred = float(trained.predict(row[feature_cols])[0])
            preds.append(pred)
            g = pd.concat([g, pd.DataFrame([{"date": next_date, "sales": pred}])], ignore_index=True)
        return np.array(preds, dtype=float)

    def _forecast_for_group(self, grp: pd.DataFrame, group_name: str) -> Dict:
        candidates = []
        models = {}

        arima_eval = self._evaluate_arima_cv(grp)
        if arima_eval:
            candidates.append(arima_eval[0])
            models["arima"] = arima_eval[1]

        prophet_eval = self._evaluate_prophet_cv(grp)
        if prophet_eval:
            candidates.append(prophet_eval[0])
            models["prophet"] = prophet_eval[1]

        ml_eval = self._evaluate_ml_cv(grp)
        if ml_eval:
            candidates.append(ml_eval[0])
            models["lstm_fallback_rf"] = (ml_eval[1], ml_eval[2], ml_eval[3])

        if not candidates:
            # Very sparse fallback
            s = grp.sort_values("month")["sales"].values
            x = np.arange(len(s)).reshape(-1, 1)
            lr = LinearRegression()
            lr.fit(x, s)
            fut_x = np.arange(len(s), len(s) + self.horizon).reshape(-1, 1)
            preds = lr.predict(fut_x)
            residuals = s - lr.predict(x)
            lower, upper = confidence_interval_from_residuals(preds, residuals)
            return {
                "group": group_name,
                "selected_model": "linear_fallback",
                "metrics": {},
                "forecast": preds.round(2).tolist(),
                "confidence_interval": {
                    "lower": np.maximum(0, lower).round(2).tolist(),
                    "upper": np.maximum(0, upper).round(2).tolist(),
                },
            }

        best = self._choose_best(candidates)
        metrics_payload = {
            m.name: {
                "mae": round(m.mae, 3),
                "rmse": round(m.rmse, 3),
                "r2": round(m.r2, 3),
                "mape": round(m.mape, 3),
                "smape": round(m.smape, 3),
            }
            for m in candidates
        }

        if best.name == "arima":
            model = models["arima"]
            preds = np.array(model.forecast(self.horizon), dtype=float)
            residuals = np.array(model.resid, dtype=float)
        elif best.name == "prophet":
            model = models["prophet"]
            last_ds = grp["month"].max()
            fut = pd.DataFrame(
                {
                    "ds": pd.date_range(
                        last_ds + pd.offsets.MonthBegin(1),
                        periods=self.horizon,
                        freq="MS",
                    )
                }
            )
            fc = model.predict(fut)
            preds = fc["yhat"].values.astype(float)
            residuals = (model.history["y"].values - model.predict(model.history[["ds"]])["yhat"].values).astype(float)
        else:
            model, history_df, feature_cols = models["lstm_fallback_rf"]
            preds = self._forecast_ml_recursive(model, history_df[["date", "sales"]], self.horizon)
            insample = model.predict(history_df[feature_cols])
            residuals = history_df["sales"].values - insample

        lower, upper = confidence_interval_from_residuals(preds, residuals)
        return {
            "group": group_name,
            "selected_model": best.name,
            "metrics": metrics_payload,
            "forecast": np.maximum(0, preds).round(2).tolist(),
            "confidence_interval": {
                "lower": np.maximum(0, lower).round(2).tolist(),
                "upper": np.maximum(0, upper).round(2).tolist(),
            },
        }

    def run(self) -> Dict:
        df = self.load_and_validate()

        # Category-level forecasting (stable and dense)
        cat_monthly = self.preprocess_monthly(df, "category")
        category_results = []
        for category, grp in cat_monthly.groupby("category"):
            category_results.append(self._forecast_for_group(grp, str(category)))

        # Product-level forecasting for top products by revenue
        product_revenue = df.groupby("product_id")["sales"].sum().sort_values(ascending=False)
        top_products = product_revenue.head(self.top_products).index.tolist()
        prod_df = df[df["product_id"].isin(top_products)]
        prod_monthly = self.preprocess_monthly(prod_df, "product_id")

        product_results = []
        for product_id, grp in prod_monthly.groupby("product_id"):
            product_results.append(self._forecast_for_group(grp, str(product_id)))

        summary = {
            "input_file": str(self.csv_path),
            "horizon_months": self.horizon,
            "category_count": int(df["category"].nunique()),
            "product_count": int(df["product_id"].nunique()),
            "top_products_modeled": len(top_products),
            "models_available": {
                "arima": self.arima_cls is not None,
                "prophet": self.prophet_cls is not None,
                "lstm_fallback_rf": True,
            },
            "category_forecasts": category_results,
            "product_forecasts": product_results,
        }

        output_file = self.output_dir / "advanced_forecast_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        leaderboard_rows = []
        for res in category_results:
            for model_name, vals in res.get("metrics", {}).items():
                leaderboard_rows.append(
                    {
                        "level": "category",
                        "group": res["group"],
                        "model": model_name,
                        **vals,
                    }
                )
        for res in product_results:
            for model_name, vals in res.get("metrics", {}).items():
                leaderboard_rows.append(
                    {
                        "level": "product",
                        "group": res["group"],
                        "model": model_name,
                        **vals,
                    }
                )
        if leaderboard_rows:
            pd.DataFrame(leaderboard_rows).to_csv(self.output_dir / "model_leaderboard.csv", index=False)

        forecast_rows = []
        for level_name, items in [("category", category_results), ("product", product_results)]:
            for res in items:
                start = pd.Timestamp.today().to_period("M").to_timestamp() + pd.offsets.MonthBegin(1)
                months = pd.date_range(start, periods=self.horizon, freq="MS")
                for i, month in enumerate(months):
                    forecast_rows.append(
                        {
                            "level": level_name,
                            "group": res["group"],
                            "month": month.strftime("%Y-%m-%d"),
                            "forecast": res["forecast"][i],
                            "lower_95": res["confidence_interval"]["lower"][i],
                            "upper_95": res["confidence_interval"]["upper"][i],
                            "selected_model": res["selected_model"],
                        }
                    )
        pd.DataFrame(forecast_rows).to_csv(self.output_dir / "forecast_output.csv", index=False)
        return summary


def main():
    parser = argparse.ArgumentParser(description="Advanced sales forecasting pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to train.csv (Superstore-like format)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon in months (3 to 6 recommended)")
    parser.add_argument("--top-products", type=int, default=20, help="How many top products to model individually")
    args = parser.parse_args()

    if args.horizon < 1:
        raise ValueError("horizon must be >= 1")

    forecaster = AdvancedSalesForecaster(
        csv_path=Path(args.input),
        output_dir=Path(args.output_dir),
        horizon=args.horizon,
        top_products=args.top_products,
    )
    result = forecaster.run()
    print(json.dumps({"status": "ok", "output_dir": args.output_dir, "summary_keys": list(result.keys())}, indent=2))


if __name__ == "__main__":
    main()
