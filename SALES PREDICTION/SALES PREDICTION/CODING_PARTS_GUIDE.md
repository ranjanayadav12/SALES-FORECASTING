# Coding Parts Guide

Use this list when you need to quickly explain the project structure.

## Core Backend
- `app.py`: Main Flask app, user routes, prediction APIs, and admin routes.
- `database.py`: Database config and SQLAlchemy models (`User`, `Prediction`, `SalesRecord`).
- `feature_utils.py`: Feature engineering, trend/anomaly logic, and forecasting helpers.
- `advanced_forecasting_pipeline.py`: Advanced pipeline for explainable and seasonal forecasting.
- `scenario_service.py`: Scenario simulation service.
- `ai_insight_service.py`: AI-based business insight helper.

## UI Templates
- `templates/base.html`: Shared base layout (fonts, global style hook).
- `templates/login.html`: User login page.
- `templates/register.html`: User registration page.
- `templates/dashboard.html`: Main user dashboard.
- `templates/index.html`: Single prediction form page.
- `templates/admin_login.html`: Separate admin login page.
- `templates/admin_dashboard.html`: Separate admin dashboard page.

## Styling
- `static/css/style.css`: Global handcrafted UI theme used by all pages.

## Data and Models
- `model.pkl`: Trained prediction model.
- `scaler.pkl`: Input scaler used before model prediction.
- `sales_forecasting.db`: SQLite database file.
- `training_data.csv`: Training dataset.

## Useful Scripts
- `train_model.py`: Train baseline model.
- `retrain_model.py`: Retrain model from data.
- `test_prediction.py`: Prediction checks.
