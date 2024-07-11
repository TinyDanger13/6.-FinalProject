import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

df = pd.read_csv('cleaned_astra_and_competitors.csv')

# Numerical values only
numerical_features = df.select_dtypes(include=[np.number])
X = numerical_features.drop(columns=['Price'])
y = numerical_features['Price']

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# Create polynomial features and scale the data
# (used to capture non-linear relationships between features and the target variable)
poly = PolynomialFeatures(degree=2, include_bias=False)
scaler = StandardScaler()

# Models
models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'LightGBM': LGBMRegressor(),
    'CatBoost': CatBoostRegressor(verbose=0)}

# Creating a pipeline with polynomial features and scaling
pipeline = Pipeline([
    ('poly', poly),
    ('scaler', scaler),
    ('model', Ridge())]) # Placeholder, will be set for each model

# Hyperparameter tuning params
param_grids = {
    'Ridge': {'model__alpha': [0.1, 1.0, 10.0]},
    'Lasso': {'model__alpha': [0.1, 1.0, 10.0]},
    'Random Forest': {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20]},
    'Gradient Boosting': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1, 0.2]},
    'XGBoost': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1, 0.2]},
    'LightGBM': {'model__n_estimators': [100, 200], 'model__learning_rate': [0.01, 0.1, 0.2]},
    'CatBoost': {'model__iterations': [500, 1000], 'model__learning_rate': [0.01, 0.1, 0.2]}}

# Evaluating models using grid search
results = {}
best_estimators = {}

for name, model in models.items():
    pipeline.set_params(model=model)
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R-squared': r2}
    best_estimators[name] = best_model
    print(f"{name} - Best Params: {grid_search.best_params_} - MSE: {mse}, R-squared: {r2}")

# Printing the results
print("\nModel Comparison:")
for model_name, metrics in results.items():
    print(f"{model_name}: MSE = {metrics['MSE']}, R-squared = {metrics['R-squared']}")

# Selecting the best model based on MSE
best_model_name = min(results, key=lambda k: results[k]['MSE'])
best_model = best_estimators[best_model_name]

# Training the best model on the entire training set and evaluating
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"\nBest Model: {best_model_name}")
print(f"MSE: {mse_best}")
print(f"R-squared: {r2_best}")

# Visualizing the results
plt.figure(figsize=(14, 6))

# Scatter plot for actual vs predicted prices
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_best, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title(f'Actual vs. Predicted Prices ({best_model_name})')

# Residual plot
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_best
plt.scatter(y_pred_best, residuals, color='blue')
plt.hlines(0, min(y_pred_best), max(y_pred_best), colors='red', linestyles='dashed')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title(f'Residual Plot ({best_model_name})')

plt.tight_layout()
plt.show()

# Saving the best model to a file
joblib_file = f'best_model_{best_model_name.replace(" ", "_").lower()}_all_v4.pkl'
joblib.dump(best_model, joblib_file)
print(f"Best model saved to {joblib_file}")