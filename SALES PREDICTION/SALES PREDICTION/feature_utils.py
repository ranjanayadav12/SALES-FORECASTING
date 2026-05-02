import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
from scipy import stats
import pickle
import os

def create_features(input_data):
    """
    Create features from raw input for the sales prediction model.
    
    Args:
        input_data: numpy array or list with [rate, sales_first_month, sales_second_month]
    
    Returns:
        DataFrame with all 22 engineered features (matching retrain_model.py)
    """
    # Convert to numpy array if needed
    if isinstance(input_data, list):
        input_data = np.array([input_data])
    
    # Create DataFrame with column names
    if input_data.shape[0] == 1:
        df = pd.DataFrame(input_data, columns=['rate', 'sales_in_first_month', 'sales_in_second_month'])
    else:
        df = pd.DataFrame(input_data, columns=['rate', 'sales_in_first_month', 'sales_in_second_month'])
    
    # 1. Basic interaction features
    df['rate_x_sales_first'] = df['rate'] * df['sales_in_first_month']
    df['rate_x_sales_second'] = df['rate'] * df['sales_in_second_month']
    df['sales_first_x_sales_second'] = df['sales_in_first_month'] * df['sales_in_second_month']
    
    # 2. Ratio features (with smoothing to avoid division by zero)
    df['sales_ratio_first_second'] = df['sales_in_first_month'] / (df['sales_in_second_month'] + 1)
    df['sales_ratio_second_first'] = df['sales_in_second_month'] / (df['sales_in_first_month'] + 1)
    
    # 3. Difference and trend features
    df['sales_diff'] = df['sales_in_first_month'] - df['sales_in_second_month']
    df['sales_trend'] = (df['sales_in_second_month'] - df['sales_in_first_month']) / (df['sales_in_first_month'] + 1)
    
    # 4. Total and average
    df['total_sales'] = df['sales_in_first_month'] + df['sales_in_second_month']
    df['avg_sales'] = (df['sales_in_first_month'] + df['sales_in_second_month']) / 2
    
    # 5. Rate features
    df['rate_squared'] = df['rate'] ** 2
    df['rate_log'] = np.log1p(df['rate'])
    
    # 6. Sales polynomial features (Missing in previous version, added to match retrain_model.py)
    df['sales_first_squared'] = df['sales_in_first_month'] ** 2
    df['sales_second_squared'] = df['sales_in_second_month'] ** 2
    df['sales_first_log'] = np.log1p(df['sales_in_first_month'])
    df['sales_second_log'] = np.log1p(df['sales_in_second_month'])
    
    # 7. Normalized features
    df['sales_first_norm'] = df['sales_in_first_month'] / (df['total_sales'] + 1)
    df['sales_second_norm'] = df['sales_in_second_month'] / (df['total_sales'] + 1)
    
    # 8. Rate impact features
    df['rate_per_sales_first'] = df['rate'] / (df['sales_in_first_month'] + 1) * 100000
    df['rate_per_sales_second'] = df['rate'] / (df['sales_in_second_month'] + 1) * 100000
    
    return df


def get_feature_columns():
    """
    Return the list of 22 feature columns used in training.
    """
    return [
        'rate', 
        'sales_in_first_month', 
        'sales_in_second_month',
        'rate_x_sales_first',
        'rate_x_sales_second',
        'sales_first_x_sales_second',
        'sales_ratio_first_second',
        'sales_ratio_second_first',
        'sales_diff',
        'sales_trend',
        'total_sales',
        'avg_sales',
        'rate_squared',
        'rate_log',
        'sales_first_norm',
        'sales_second_norm',
        'rate_per_sales_first',
        'rate_per_sales_second'
    ]


def prepare_input(raw_input, scaler=None):
    """
    Prepare raw input for model prediction.
    
    Args:
        raw_input: list or array with [rate, sales_first, sales_second]
        scaler: StandardScaler object (optional)
    
    Returns:
        Scaled feature array ready for prediction
    """
    # Create features
    features_df = create_features(raw_input)
    
    # Get feature columns in correct order (18 features matching the trained model)
    feature_cols = get_feature_columns()
    X = features_df[feature_cols]
    
    # Scale if scaler provided — pass DataFrame with names to avoid sklearn warnings
    if scaler is not None:
        X = scaler.transform(X)
    else:
        X = X.values
    
    return X


# ============================================
# Time Series Analysis Functions
# ============================================

def detect_trend(sales_data):
    """
    Detect trend in sales data using linear regression.
    
    Args:
        sales_data: List or array of sales values ordered by time
    
    Returns:
        Dictionary with trend information
    """
    if len(sales_data) < 3:
        return {
            'trend': 'insufficient_data',
            'direction': 'stable',
            'slope': 0,
            'strength': 0
        }
    
    sales_array = np.array(sales_data)
    x = np.arange(len(sales_array))
    
    # Linear regression for trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, sales_array)
    
    # Determine direction and strength
    if abs(slope) < 1000:  # Small absolute slope
        direction = 'stable'
        strength = 'weak'
    elif slope > 0:
        direction = 'upward'
        strength = 'strong' if abs(r_value) > 0.7 else 'moderate'
    else:
        direction = 'downward'
        strength = 'strong' if abs(r_value) > 0.7 else 'moderate'
    
    return {
        'trend': 'detected',
        'direction': direction,
        'slope': round(slope, 2),
        'strength': strength,
        'r_squared': round(r_value**2, 4),
        'p_value': round(p_value, 4)
    }


def detect_seasonality(sales_data, period=None):
    """
    Detect seasonality in sales data.
    
    Args:
        sales_data: List or array of sales values
        period: Expected seasonal period (e.g., 12 for monthly, 7 for weekly)
    
    Returns:
        Dictionary with seasonality information
    """
    if len(sales_data) < 4:
        return {
            'seasonality': 'insufficient_data',
            'has_seasonality': False,
            'period': None
        }
    
    sales_array = np.array(sales_data)
    
    # Auto-detect period if not provided
    if period is None:
        # Use autocorrelation to find period
        n = len(sales_array)
        max_lag = min(n // 2, 12)
        
        best_corr = 0
        best_period = 1
        
        for lag in range(2, max_lag + 1):
            if lag >= n:
                break
            corr = np.corrcoef(sales_array[:-lag], sales_array[lag:])[0, 1]
            if not np.isnan(corr) and abs(corr) > abs(best_corr):
                best_corr = corr
                best_period = lag
        
        period = best_period
    
    # Calculate seasonal strength
    # Using coefficient of variation of seasonal means
    n_periods = len(sales_array) // period
    if n_periods < 2:
        return {
            'seasonality': 'insufficient_data',
            'has_seasonality': False,
            'period': period
        }
    
    # Reshape data by periods
    seasonal_means = []
    for i in range(n_periods):
        start = i * period
        end = min((i + 1) * period, len(sales_array))
        seasonal_means.append(np.mean(sales_array[start:end]))
    
    overall_mean = np.mean(sales_array)
    if overall_mean > 0:
        seasonal_variation = np.std(seasonal_means) / overall_mean
        has_seasonality = seasonal_variation > 0.1
    else:
        seasonal_variation = 0
        has_seasonality = False
    
    return {
        'seasonality': 'detected' if has_seasonality else 'none',
        'has_seasonality': has_seasonality,
        'period': period,
        'seasonal_strength': round(seasonal_variation, 4),
        'peak_season': seasonal_means.index(max(seasonal_means)) + 1 if seasonal_means else None,
        'low_season': seasonal_means.index(min(seasonal_means)) + 1 if seasonal_means else None
    }


def calculate_moving_average(sales_data, window=3):
    """
    Calculate moving average for smoothing.
    
    Args:
        sales_data: List of sales values
        window: Window size for moving average
    
    Returns:
        List of moving average values
    """
    if len(sales_data) < window:
        return sales_data
    
    sales_array = np.array(sales_data)
    ma = np.convolve(sales_array, np.ones(window)/window, mode='valid')
    return ma.tolist()


def forecast_next_period(sales_data, model=None, scaler=None, rate=None, periods=1):
    """
    Forecast next period sales using linear regression formula: y = a + bx
    
    Formula:
    b = (n∑xy - ∑x∑y) / (n∑x² - (∑x)²)
    a = (∑y - b∑x) / n
    
    For Month n: Sales = a + b*n
    
    Args:
        sales_data: Historical sales data
        model: Trained ML model (optional)
        scaler: Feature scaler (optional)
        rate: Current rate value (required for ML model)
        periods: Number of periods to forecast
    
    Returns:
        Dictionary with forecast information
    """
    if len(sales_data) < 2:
        return {
            'forecast': 'insufficient_data',
            'predicted_value': None,
            'confidence': None
        }
    
    sales_array = np.array(sales_data)
    n = len(sales_array)
    
    # Calculate linear regression using the provided formula
    # x values are time periods (1, 2, 3, ..., n)
    x = np.arange(1, n + 1)
    y = sales_array
    
    # Calculate required sums
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    # Calculate b using formula: b = (n∑xy - ∑x∑y) / (n∑x² - (∑x)²)
    numerator_b = n * sum_xy - sum_x * sum_y
    denominator_b = n * sum_x2 - sum_x * sum_x
    
    if denominator_b == 0:
        # Avoid division by zero
        b = 0
    else:
        b = numerator_b / denominator_b
    
    # Calculate a using formula: a = (∑y - b∑x) / n
    a = (sum_y - b * sum_x) / n
    
    # Calculate R-squared for confidence
    y_pred = a + b * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Forecast for next period (period = n + 1)
    next_period = n + 1
    predicted = a + b * next_period
    
    # Calculate confidence level based on R-squared
    confidence = max(0.3, min(0.95, r_squared))
    
    return {
        'method': 'linear_regression',
        'formula': f'y = {a:.2f} + {b:.2f} * x',
        'a': round(a, 2),
        'b': round(b, 2),
        'r_squared': round(r_squared, 4),
        'predicted_value': round(predicted, 2),
        'confidence_level': round(confidence * 100, 1),
        'confidence_label': 'High' if confidence > 0.8 else 'Medium' if confidence > 0.5 else 'Low',
        'next_period': next_period
    }


def forecast_with_linear_regression(sales_data, target_month):
    """
    Forecast sales for a specific month using linear regression formula.
    
    Formula:
    b = (n∑xy - ∑x∑y) / (n∑x² - (∑x)²)
    a = (∑y - b∑x) / n
    
    For Month m: Sales = a + b * m
    
    Args:
        sales_data: Historical sales data
        target_month: The month number to forecast (e.g., 3 for Month 3)
    
    Returns:
        Dictionary with forecast information
    """
    if len(sales_data) < 2:
        return {
            'error': 'Insufficient data',
            'predicted_value': None
        }
    
    sales_array = np.array(sales_data)
    n = len(sales_array)
    
    # Calculate linear regression using the formula
    x = np.arange(1, n + 1)
    y = sales_array
    
    # Required sums
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    # Calculate b = (n∑xy - ∑x∑y) / (n∑x² - (∑x)²)
    numerator_b = n * sum_xy - sum_x * sum_y
    denominator_b = n * sum_x2 - sum_x * sum_x
    
    if denominator_b == 0:
        b = 0
    else:
        b = numerator_b / denominator_b
    
    # Calculate a = (∑y - b∑x) / n
    a = (sum_y - b * sum_x) / n
    
    # Forecast for target month: Sales = a + b * target_month
    predicted = a + b * target_month
    
    # Calculate R-squared
    y_pred = a + b * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'formula': f'Sales = {a:.2f} + {b:.2f} × {target_month}',
        'a': round(a, 2),
        'b': round(b, 2),
        'target_month': target_month,
        'predicted_value': round(predicted, 2),
        'r_squared': round(r_squared, 4),
        'method': 'linear_regression_formula'
    }


def analyze_sales_data(sales_data, rate=None, model=None, scaler=None):
    """
    Comprehensive time-series analysis of sales data.
    
    Args:
        sales_data: List of sales values in chronological order
        rate: Current rate (required for ML prediction)
        model: Trained ML model
        scaler: Feature scaler
    
    Returns:
        Dictionary with all analysis results
    """
    result = {
        'data_summary': {},
        'trend': {},
        'seasonality': {},
        'forecast': {},
        'insights': []
    }
    
    if len(sales_data) < 2:
        result['error'] = 'Insufficient data for analysis. Need at least 2 data points.'
        return result
    
    sales_array = np.array(sales_data)
    
    # Data summary
    result['data_summary'] = {
        'total_periods': len(sales_data),
        'mean': round(np.mean(sales_array), 2),
        'median': round(np.median(sales_array), 2),
        'std_dev': round(np.std(sales_array), 2),
        'min': round(np.min(sales_array), 2),
        'max': round(np.max(sales_array), 2),
        'range': round(np.max(sales_array) - np.min(sales_array), 2)
    }
    
    # Trend analysis
    result['trend'] = detect_trend(sales_data)
    
    # Seasonality analysis
    result['seasonality'] = detect_seasonality(sales_data)
    
    # Moving averages
    result['moving_averages'] = {
        'ma_3': calculate_moving_average(sales_data, 3)[-1] if len(sales_data) >= 3 else None,
        'ma_5': calculate_moving_average(sales_data, 5)[-1] if len(sales_data) >= 5 else None
    }
    
    # Forecast
    if rate is not None and model is not None and scaler is not None:
        result['forecast'] = forecast_next_period(sales_data, model, scaler, rate)
    else:
        result['forecast'] = forecast_next_period(sales_data)
    
    # Generate insights
    if result['trend']['direction'] == 'upward':
        result['insights'].append({
            'type': 'positive',
            'message': '📈 Sales show an upward trend, indicating business growth.'
        })
    elif result['trend']['direction'] == 'downward':
        result['insights'].append({
            'type': 'warning',
            'message': '📉 Sales show a downward trend. Consider reviewing your strategy.'
        })
    
    if result['seasonality'].get('has_seasonality'):
        result['insights'].append({
            'type': 'info',
            'message': f'🔄 Seasonal patterns detected with period of {result["seasonality"]["period"]} months.'
        })
    
    if result['forecast'].get('predicted_value'):
        forecast_val = result['forecast']['predicted_value']
        last_val = sales_data[-1]
        change_pct = ((forecast_val - last_val) / last_val) * 100 if last_val > 0 else 0
        
        if change_pct > 10:
            result['insights'].append({
                'type': 'positive',
                'message': f'🚀 Predicted sales for next period: {forecast_val:,.0f} (+{change_pct:.1f}% growth).'
            })
        elif change_pct < -10:
            result['insights'].append({
                'type': 'warning',
                'message': f'⚠️ Predicted sales for next period: {forecast_val:,.0f} ({change_pct:.1f}% decline).'
            })
        else:
            result['insights'].append({
                'type': 'neutral',
                'message': f'➡️ Predicted sales for next period: {forecast_val:,.0f} (stable).'
            })
    
    return result


# ============================================
# Existing Functions (unchanged)
# ============================================

def calculate_confidence_interval(model, X, y, scaler=None, n_bootstraps=100, confidence=0.95):
    """Calculate prediction confidence intervals using bootstrap sampling."""
    from sklearn.utils import resample
    
    predictions = []
    
    try:
        if hasattr(X, 'values'):
            X_array = X.values
        else:
            X_array = np.array(X)
        
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(1, -1)
        
        if scaler is not None:
            X_scaled = scaler.transform(X_array)
        else:
            X_scaled = X_array
        
        # Ensure y is an array
        y_array = np.array(y) if not hasattr(y, '__iter__') or isinstance(y, (list, tuple)) else y
        
        for i in range(n_bootstraps):
            try:
                X_boot, y_boot = resample(X_scaled, y_array, random_state=i)
                if len(X_boot) < 2 or len(y_boot) < 2:
                    continue
                boot_model = RandomForestRegressor(n_estimators=10, random_state=i)
                boot_model.fit(X_boot, y_boot)
                pred = boot_model.predict(X_scaled[:1])[0]
                predictions.append(pred)
            except Exception:
                continue
        
        if len(predictions) < 10:
            return {'lower_bound': None, 'upper_bound': None, 'confidence_level': confidence}
        
        predictions = np.array(predictions)
        
        # Check for NaN or Inf values
        if not np.all(np.isfinite(predictions)):
            return {'lower_bound': None, 'upper_bound': None, 'confidence_level': confidence}
        
        alpha = 1 - confidence
        
        lower_bound = np.percentile(predictions, (alpha/2) * 100)
        upper_bound = np.percentile(predictions, (1 - alpha/2) * 100)
        
        return {
            'lower_bound': round(lower_bound, 2),
            'upper_bound': round(upper_bound, 2),
            'confidence_level': confidence,
            'std_dev': round(np.std(predictions), 2)
        }
    except Exception as e:
        # Return None values on any error
        return {'lower_bound': None, 'upper_bound': None, 'confidence_level': confidence}


def detect_anomalies(predictions_list, threshold=2.0):
    """Detect anomalies in prediction history."""
    if len(predictions_list) < 3:
        return [{'value': p, 'is_anomaly': False, 'z_score': 0} for p in predictions_list]
    
    predictions_array = np.array(predictions_list)
    mean = np.mean(predictions_array)
    std = np.std(predictions_array)
    
    if std == 0:
        return [{'value': p, 'is_anomaly': False, 'z_score': 0} for p in predictions_list]
    
    results = []
    for pred in predictions_list:
        z_score = (pred - mean) / std
        is_anomaly = abs(z_score) > threshold
        
        results.append({
            'value': pred,
            'is_anomaly': is_anomaly,
            'z_score': round(z_score, 2),
            'deviation': round(abs(pred - mean), 2)
        })
    
    return results


def calculate_trend_analysis(predictions_list, timestamps=None):
    """Calculate trend analysis with moving averages."""
    if len(predictions_list) < 2:
        return {
            'trend': 'insufficient_data',
            'direction': 'stable',
            'ma_7': None,
            'ma_30': None,
            'change_percentage': 0
        }
    
    predictions_array = np.array(predictions_list)
    
    ma_7 = None
    ma_30 = None
    
    if len(predictions_array) >= 7:
        ma_7 = round(np.mean(predictions_array[-7:]), 2)
    
    if len(predictions_array) >= 30:
        ma_30 = round(np.mean(predictions_array[-30:]), 2)
    
    if len(predictions_array) >= 2:
        recent_avg = np.mean(predictions_array[-min(5, len(predictions_array)):])
        older_avg = np.mean(predictions_array[:min(5, len(predictions_array))])
        
        if older_avg > 0:
            change_pct = ((recent_avg - older_avg) / older_avg) * 100
        else:
            change_pct = 0
        
        if change_pct > 5:
            direction = 'up'
        elif change_pct < -5:
            direction = 'down'
        else:
            direction = 'stable'
    else:
        change_pct = 0
        direction = 'stable'
    
    return {
        'trend': 'analyzed',
        'direction': direction,
        'ma_7': ma_7,
        'ma_30': ma_30,
        'change_percentage': round(change_pct, 2),
        'current_value': round(predictions_array[-1], 2) if len(predictions_array) > 0 else None
    }


def generate_business_insights(predictions_list):
    """Generate business insights from prediction history."""
    insights = []
    
    if len(predictions_list) < 3:
        insights.append({
            'type': 'info',
            'message': 'Make more predictions to get personalized business insights.'
        })
        return insights
    
    predictions_array = np.array(predictions_list)
    
    avg_sales = np.mean(predictions_array)
    insights.append({
        'type': 'stat',
        'title': 'Average Sales',
        'message': f'Your average predicted sales is ₹{avg_sales:,.2f}'
    })
    
    if len(predictions_array) >= 5:
        recent_avg = np.mean(predictions_array[-3:])
        older_avg = np.mean(predictions_array[:3])
        
        if older_avg > 0:
            change = ((recent_avg - older_avg) / older_avg) * 100
            
            if change > 10:
                insights.append({
                    'type': 'success',
                    'title': 'Growth Trend',
                    'message': f'Your sales are up by {change:.1f}% recently! Keep up the good work.'
                })
            elif change < -10:
                insights.append({
                    'type': 'warning',
                    'title': 'Declining Trend',
                    'message': f'Your sales have decreased by {abs(change):.1f}%. Consider reviewing your strategy.'
                })
    
    if len(predictions_array) >= 5:
        cv = np.std(predictions_array) / np.mean(predictions_array) * 100 if np.mean(predictions_array) > 0 else 0
        
        if cv < 20:
            insights.append({
                'type': 'success',
                'title': 'Consistent Performance',
                'message': 'Your predictions are very consistent with low variance.'
            })
        elif cv > 50:
            insights.append({
                'type': 'info',
                'title': 'Variable Performance',
                'message': 'Your sales show high variation. Consider analyzing factors causing this.'
            })
    
    max_sales = np.max(predictions_array)
    min_sales = np.min(predictions_array)
    insights.append({
        'type': 'stat',
        'title': 'Sales Range',
        'message': f'Your sales range from ₹{min_sales:,.2f} to ₹{max_sales:,.2f}'
    })
    
    return insights


def train_comparison_models(X, y):
    """Train multiple models for comparison."""
    models = {}
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    models['Random Forest'] = rf
    
    et = ExtraTreesRegressor(n_estimators=100, random_state=42)
    et.fit(X, y)
    models['Extra Trees'] = et
    
    return models


def get_model_comparison(models, X, y=None, X_train=None):
    """Get comparison metrics for multiple models."""
    comparison = {}

    for name, model in models.items():
        pred = model.predict(X)
        comparison[name] = {
            'prediction': round(float(pred[0]), 2),
            'model': model
        }

        if y is not None and X_train is not None:
            try:
                cv_folds = min(5, len(y))
                if cv_folds < 2 or len(X_train) != len(y):
                    comparison[name]['cv_score'] = None
                else:
                    cv_scores = cross_val_score(model, X_train, y, cv=cv_folds, scoring='r2')
                    comparison[name]['cv_score'] = round(np.mean(cv_scores), 3)
            except Exception:
                comparison[name]['cv_score'] = None

    return comparison


def goal_seek_prediction(target_sales, rate, sales_first, sales_second, scaler=None, model=None, linear_model=None):
    """Perform goal seeking to find the required parameter."""
    if sales_second is not None:
        # Intelligent Unit Handling for Goal Seek (Lakhs vs Rupees)
        if (sales_first > 10000 or sales_second > 10000):
             sales_first = sales_first * 0.00001
             sales_second = sales_second * 0.00001
             target_sales = target_sales * 0.00001
             print(f"Goal Seek Auto-Scaling: Input {sales_first*100000}/{sales_second*100000} -> Lakhs")

        if linear_model:
            X_lin = np.array([[rate, sales_first, sales_second]])
            predicted = float(linear_model.predict(X_lin)[0])
        else:
            input_features = create_features([rate, sales_first, sales_second])
            feature_cols = get_feature_columns()
            input_df = input_features[feature_cols]

            if scaler:
                input_data = scaler.transform(input_df)
            else:
                input_data = input_df.values

            predicted = float(model.predict(input_data)[0])
        
        # Dynamic Growth logic for Goal Seek
        recent_avg = (sales_first + sales_second) / 2
        growth_floor = recent_avg * 1.10
        predicted = max(predicted, growth_floor, 0.1)

        return {
            'target_sales': target_sales,
            'predicted_sales': round(predicted, 2),
            'achieved': abs(predicted - target_sales) < target_sales * 0.05,
            'message': f'With sales_second={sales_second}, predicted sales is Rs {predicted:,.2f}'
        }

    best_guess = 0.0
    best_predicted = 0.0
    best_diff = float('inf')
    upper_bound = max(sales_first * 2, target_sales * 1.2, 1.0)

    for guess in np.linspace(0, upper_bound, 140):
        # Intelligent Unit Handling for guessing loop
        local_s1 = sales_first
        local_guess = guess
        local_target = target_sales
        if (local_s1 > 10000 or local_guess > 10000):
             local_s1 = local_s1 * 0.00001
             local_guess = local_guess * 0.00001
             local_target = local_target * 0.00001

        if linear_model:
            X_lin = np.array([[rate, local_s1, local_guess]])
            predicted = float(linear_model.predict(X_lin)[0])
        else:
            input_features = create_features([rate, sales_first, guess])
            feature_cols = get_feature_columns()
            input_df = input_features[feature_cols]

            if scaler:
                input_data = scaler.transform(input_df)
            else:
                input_data = input_df.values

            predicted = float(model.predict(input_data)[0])
        
        # Dynamic Growth logic
        recent_avg = (sales_first + guess) / 2
        growth_floor = recent_avg * 1.10
        predicted = max(predicted, growth_floor, 0.1)
        diff = abs(predicted - target_sales)

        if diff < best_diff:
            best_diff = diff
            best_guess = guess
            best_predicted = predicted

    return {
        'target_sales': target_sales,
        'required_sales_second': round(best_guess, 2),
        'predicted_sales': round(best_predicted, 2),
        'achieved': best_diff < target_sales * 0.1,
        'message': f'To target Rs {target_sales:,.2f}, aim for sales_second near Rs {best_guess:,.2f}'
    }


# ============================================
# Festival and Season Detection
# ============================================

FESTIVE_MONTH_MAP = {
    1: ['New Year', 'Pongal'],
    3: ['Holi'],
    4: ['Easter'],
    8: ['Raksha Bandhan', 'Onam'],
    9: ['Ganesh Chaturthi', 'Navratri'],
    10: ['Durga Puja', 'Diwali'],
    11: ['Diwali', 'Post-Diwali Demand'],
    12: ['Christmas', 'Year-End Demand']
}


def _is_festive_month(month):
    return month in FESTIVE_MONTH_MAP


def analyze_festive_impact(sales_data, date_data=None, horizon=6):
    """
    Analyze festive impact and estimate future festive demand.
    """
    result = {
        'has_date_context': False,
        'festive_growth_pct': 0.0,
        'normal_avg_sales': 0.0,
        'festive_avg_sales': 0.0,
        'festive_uplift_factor': 1.0,
        'normal_vs_festive': {},
        'trend_analysis': {},
        'next_festive_forecast': [],
        'festival_month_breakdown': []
    }

    if sales_data is None or len(sales_data) < 3:
        return result

    sales_series = pd.to_numeric(pd.Series(sales_data), errors='coerce').dropna().reset_index(drop=True)
    if len(sales_series) < 3:
        return result

    if date_data is not None:
        date_series = pd.to_datetime(pd.Series(date_data), errors='coerce')
    else:
        # Fallback synthetic monthly dates if real dates are not provided.
        date_series = pd.date_range(end=pd.Timestamp.today(), periods=len(sales_series), freq='MS')
        date_series = pd.Series(date_series)

    df = pd.DataFrame({'date': date_series, 'sales': sales_series})
    df = df.dropna(subset=['date', 'sales']).sort_values('date').reset_index(drop=True)
    if len(df) < 3:
        return result

    result['has_date_context'] = date_data is not None
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['is_festive'] = df['month'].apply(_is_festive_month)

    festive = df[df['is_festive']]
    normal = df[~df['is_festive']]

    festive_avg = float(festive['sales'].mean()) if len(festive) > 0 else float(df['sales'].mean())
    normal_avg = float(normal['sales'].mean()) if len(normal) > 0 else float(df['sales'].mean())
    growth_pct = ((festive_avg - normal_avg) / normal_avg * 100.0) if normal_avg > 0 else 0.0
    uplift = (festive_avg / normal_avg) if normal_avg > 0 else 1.0

    result['festive_growth_pct'] = round(growth_pct, 2)
    result['normal_avg_sales'] = round(normal_avg, 2)
    result['festive_avg_sales'] = round(festive_avg, 2)
    result['festive_uplift_factor'] = round(max(0.8, min(2.5, uplift)), 4)
    result['normal_vs_festive'] = {
        'normal_avg_sales': round(normal_avg, 2),
        'festive_avg_sales': round(festive_avg, 2),
        'difference': round(festive_avg - normal_avg, 2),
        'growth_pct': round(growth_pct, 2)
    }

    if len(festive) >= 2:
        x = np.arange(len(festive))
        slope, _, r_value, _, _ = stats.linregress(x, festive['sales'].values)
        festive_direction = 'upward' if slope > 0 else 'downward' if slope < 0 else 'stable'
        result['trend_analysis'] = {
            'direction': festive_direction,
            'slope': round(float(slope), 2),
            'strength_r2': round(float(r_value ** 2), 4)
        }
    else:
        result['trend_analysis'] = {'direction': 'stable', 'slope': 0.0, 'strength_r2': 0.0}

    festive_month_stats = (
        df[df['is_festive']]
        .groupby('month', as_index=False)['sales']
        .mean()
        .sort_values('sales', ascending=False)
    )
    for _, row in festive_month_stats.iterrows():
        month = int(row['month'])
        result['festival_month_breakdown'].append({
            'month': month,
            'festivals': FESTIVE_MONTH_MAP.get(month, ['Festival Season']),
            'avg_sales': round(float(row['sales']), 2)
        })

    # Future festive demand forecast: estimate upcoming festive months only.
    last_month = df['date'].max().to_period('M').to_timestamp()
    monthly_mean = (
        df.groupby(df['date'].dt.to_period('M'))['sales']
        .sum()
        .sort_index()
    )
    baseline = monthly_mean.tail(min(6, len(monthly_mean))).mean() if len(monthly_mean) else df['sales'].mean()
    for step in range(1, horizon + 1):
        future_month = (last_month + pd.offsets.MonthBegin(step))
        m = int(future_month.month)
        if _is_festive_month(m):
            uplift_val = baseline * result['festive_uplift_factor']
            result['next_festive_forecast'].append({
                'month': future_month.strftime('%Y-%m'),
                'festivals': FESTIVE_MONTH_MAP.get(m, ['Festival Season']),
                'normal_baseline': round(float(baseline), 2),
                'festive_adjusted_demand': round(float(uplift_val), 2)
            })

    return result

def detect_festivals_and_seasons(sales_data):
    """
    Detect festivals and seasonal patterns that may affect sales.
    
    Args:
        sales_data: List of sales values in chronological order (should have date info)
    
    Returns:
        Dictionary with festival/season insights
    """
    # Analysis results
    result = {
        'festival_insights': [],
        'seasonal_impact': [],
        'peak_periods': [],
        'recommendations': [],
        'festive_analytics': {}
    }
    
    if len(sales_data) < 3:
        result['festival_insights'].append({
            'type': 'info',
            'message': 'Not enough data to detect festival/seasonal patterns. Upload more historical data.'
        })
        return result
    
    sales_array = np.array(sales_data)
    
    # Calculate statistics
    mean_sales = np.mean(sales_array)
    std_sales = np.std(sales_array)
    
    # Find high sales periods (above average + 1 std)
    high_threshold = mean_sales + (0.5 * std_sales)
    high_sales_indices = np.where(sales_array > high_threshold)[0]
    
    # Find low sales periods (below average - 0.5 std)
    low_threshold = mean_sales - (0.5 * std_sales)
    low_sales_indices = np.where(sales_array < low_threshold)[0]
    
    # Analyze if sales increased significantly
    if len(sales_data) >= 6:
        # Compare first half vs second half
        first_half = np.mean(sales_array[:len(sales_array)//2])
        second_half = np.mean(sales_array[len(sales_array)//2:])
        change_pct = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
        
        if change_pct > 20:
            result['festival_insights'].append({
                'type': 'success',
                'title': 'Significant Growth',
                'message': f'Sales have increased by {change_pct:.1f}% over the analyzed period. This could be due to seasonal demand, festival periods, or improved business performance.'
            })
        elif change_pct < -20:
            result['festival_insights'].append({
                'type': 'warning',
                'title': 'Declining Sales',
                'message': f'Sales have decreased by {abs(change_pct):.1f}% over the analyzed period. Review pricing strategy and market conditions.'
            })
    
    # Detect volatility
    cv = (std_sales / mean_sales) * 100 if mean_sales > 0 else 0
    
    if cv > 30:
        result['festival_insights'].append({
            'type': 'info',
            'title': 'High Volatility',
            'message': f'Sales show {cv:.1f}% coefficient of variation, indicating high fluctuation. This may be due to seasonal festivals or irregular demand patterns.'
        })
    else:
        result['festival_insights'].append({
            'type': 'success',
            'title': 'Stable Sales',
            'message': f'Sales show {cv:.1f}% coefficient of variation, indicating relatively stable performance.'
        })
    
    # Month-based analysis (if data has monthly patterns)
    if len(sales_data) >= 12:
        # Find best and worst months
        monthly_avg = []
        for i in range(min(12, len(sales_data))):
            month_sales = [sales_data[j] for j in range(i, len(sales_data), 12) if j < len(sales_data)]
            if month_sales:
                monthly_avg.append((i+1, np.mean(month_sales)))
        
        if monthly_avg:
            best_month = max(monthly_avg, key=lambda x: x[1])
            worst_month = min(monthly_avg, key=lambda x: x[1])
            
            # Map months to festivals
            month_festivals = {
                1: ['New Year', 'Pongal'],
                2: ['Valentine\'s Day'],
                3: ['Holi', 'International Women\'s Day'],
                4: ['Bihu', 'Easter'],
                5: ['Mother\'s Day', 'Summer Sales'],
                6: ['Father\'s Day'],
                7: ['Monsoon Sale', 'Raksha Bandhan'],
                8: ['Independence Day', 'Onam', 'Ganesh Chaturthi'],
                9: ['Ganesh Chaturthi', 'Navratri', 'Durga Puja'],
                10: ['Durga Puja', 'Diwali Prep'],
                11: ['Diwali', 'Grand Finale Sale'],
                12: ['Christmas', 'New Year Prep', 'Winter Sale']
            }
            
            result['seasonal_impact'].append({
                'type': 'positive',
                'title': f'Best Performing Month: {best_month[0]}',
                'message': f'Month {best_month[0]} shows highest average sales. Potential影响因素: {", ".join(month_festivals.get(best_month[0], ["Regular demand"]))}'
            })
            
            result['seasonal_impact'].append({
                'type': 'warning',
                'title': f'Lowest Performing Month: {worst_month[0]}',
                'message': f'Month {worst_month[0]} shows lowest average sales. Consider promotions or inventory adjustments.'
            })
    
    # Generate recommendations
    result['recommendations'].append({
        'type': 'info',
        'title': 'Festival Planning',
        'message': 'Plan inventory and marketing campaigns 2-3 months ahead of major festivals (Diwali, Holi, Ganesh Chaturthi) to maximize sales.'
    })
    
    if cv > 20:
        result['recommendations'].append({
            'type': 'info',
            'title': 'Demand Forecasting',
            'message': 'Use historical data to create seasonal demand forecasts. Consider offering off-season promotions to smooth sales volatility.'
        })
    
    result['festive_analytics'] = analyze_festive_impact(sales_data=sales_data, date_data=None)
    return result

