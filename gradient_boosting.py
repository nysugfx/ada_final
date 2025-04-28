# gradient_boosting.py

"""
NYC Central Park Temperature Prediction - Gradient Boosting Model
=================================================================

This script trains and evaluates a Gradient Boosting Regressor to predict
daily high temperatures in Central Park, NY.

It includes:
1. Feature engineering (lags, rolling stats, cyclical encodings, heat index).
2. Time-series train/test split.
3. Imputation, scaling, and one-hot encoding.
4. Model training with GradientBoostingRegressor.
5. Evaluation via RMSE, MAE, and R².
6. Model diagnostics:
   - Feature importances plot.
   - Actual vs. Predicted time-series plot.
   - Learning curves.
   - Partial dependence plots.
   - Residual analysis.
   - Hyperparameter tuning with GridSearchCV.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import shap

def compute_heat_index(T, RH):
    c1, c2, c3 = -42.379, 2.04901523, 10.14333127
    c4, c5, c6 = -0.22475541, -6.83783e-3, -5.481717e-2
    c7, c8, c9 = 1.22874e-3, 8.5282e-4, -1.99e-6
    return (c1 + c2*T + c3*RH + c4*T*RH + c5*T*T + c6*RH*RH
            + c7*T*T*RH + c8*T*RH*RH + c9*T*T*RH*RH)

def compute_dew_point(T, RH):
    Tc = (T - 32) * 5/9
    a, b = 17.27, 237.7
    alpha = (a * Tc)/(b + Tc) + np.log(RH/100)
    Td = (b * alpha)/(a - alpha)
    return Td * 9/5 + 32

def add_features(df):
    df = df.copy().sort_values('date').set_index('date')
    # lag features
    for lag in range(1, 8):
        df[f'tmax_lag_{lag}'] = df['tmax'].shift(lag)
    # rolling stats
    for w in (3, 7):
        df[f'tmax_roll_mean_{w}'] = df['tmax'].shift(1).rolling(window=w, min_periods=1).mean()
        df[f'tmax_roll_std_{w}']  = df['tmax'].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
    # cyclical
    df['dayofyear'] = df.index.dayofyear
    df['sin_doy']   = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['cos_doy']   = np.cos(2 * np.pi * df['dayofyear'] / 365)
    # optional meteorological
    if {'tavg','humidity'}.issubset(df.columns):
        df['heat_index'] = compute_heat_index(df['tavg'], df['humidity'])
        df['dew_point']  = compute_dew_point(df['tavg'], df['humidity'])
    # drop missing
    required = [f'tmax_lag_{i}' for i in range(1, 8)]
    required += [f'tmax_roll_mean_{w}' for w in (3, 7)]
    required += [f'tmax_roll_std_{w}'  for w in (3, 7)]
    return df.dropna(subset=required).reset_index()

def main():
    # output folder
    fig_dir = os.path.join('reports', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # load & engineer
    data_path = os.path.join('data', 'processed', 'preprocessed_dataset.csv')
    df_raw = pd.read_csv(data_path, parse_dates=['date'])
    df_fe  = add_features(df_raw)

    # prepare X,y
    df_fe.set_index('date', inplace=True)
    X = df_fe.drop(columns=['tmax'])
    y = df_fe['tmax']

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # numeric preprocessing
    num_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns
                if X_train[c].notna().any()]
    imp    = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train_num = pd.DataFrame(
        scaler.fit_transform(imp.fit_transform(X_train[num_cols])),
        columns=num_cols, index=X_train.index
    )
    X_test_num  = pd.DataFrame(
        scaler.transform(imp.transform(X_test[num_cols])),
        columns=num_cols, index=X_test.index
    )

    # categorical one-hot
    cat_cols = X_train.select_dtypes(include=['object','category']).columns.tolist()
    if cat_cols:
        X_train_cat = pd.get_dummies(X_train[cat_cols], dummy_na=True)
        X_test_cat  = pd.get_dummies(X_test[cat_cols], dummy_na=True)
        X_train_cat, X_test_cat = X_train_cat.align(X_test_cat, join='left', axis=1, fill_value=0)
    else:
        X_train_cat = pd.DataFrame(index=X_train.index)
        X_test_cat  = pd.DataFrame(index=X_test.index)

    # combine
    X_train_prepared = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test_prepared  = pd.concat([X_test_num,  X_test_cat ], axis=1)

    # train
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    )
    model.fit(X_train_prepared, y_train)
    y_pred = model.predict(X_test_prepared)

    # evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")

    # feature importances
    importances = model.feature_importances_
    feat_names  = X_train_prepared.columns
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,6))
    plt.barh(feat_names[idx][:15][::-1], importances[idx][:15][::-1])
    plt.xlabel('Importance'); plt.title('Top 15 Features')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_importances.png'), dpi=300)
    plt.close()

    # actual vs predicted
    plt.figure(figsize=(12,4))
    plt.plot(y_test.index, y_test,  label='Actual')
    plt.plot(y_test.index, y_pred,   label='Predicted')
    plt.xlabel('Date'); plt.ylabel('tmax'); plt.title('Actual vs Predicted')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'actual_vs_predicted.png'), dpi=300)
    plt.close()

    # learning curves
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_prepared, y_train, cv=5,
        scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1,1.0,5), n_jobs=-1
    )
    train_rmse = np.sqrt(-train_scores)
    val_rmse   = np.sqrt(-val_scores)
    plt.figure()
    plt.plot(train_sizes, train_rmse.mean(axis=1), 'o-', label='Train RMSE')
    plt.plot(train_sizes, val_rmse.mean(axis=1),   'o-', label='Validation RMSE')
    plt.xlabel('Training Size'); plt.ylabel('RMSE'); plt.legend()
    plt.title('Learning Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'learning_curve.png'), dpi=300)
    plt.close()

    # hyperparameter tuning
    param_grid = {
        'n_estimators': [100,200,300],
        'learning_rate': [0.01,0.05,0.1],
        'max_depth': [3,4,5]
    }
    gs = GridSearchCV(
        GradientBoostingRegressor(subsample=0.8, random_state=42),
        param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )
    gs.fit(X_train_prepared, y_train)
    print("Best params:", gs.best_params_)

    # partial dependence
    PartialDependenceDisplay.from_estimator(
        gs.best_estimator_, X_test_prepared,
        features=['tmax_lag_1','sin_doy'],
        kind='average', subsample=500, random_state=42
    )
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'partial_dependence.png'), dpi=300)
    plt.close()

    # residual analysis
    residuals = y_test - gs.best_estimator_.predict(X_test_prepared)
    plt.figure()
    plt.scatter(gs.best_estimator_.predict(X_test_prepared), residuals, alpha=0.3)
    plt.axhline(0, color='black', linewidth=1)
    plt.xlabel('Predicted'); plt.ylabel('Residual'); plt.title('Residuals vs Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'residuals_plot.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()

