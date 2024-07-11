import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv('opel_cleaned.csv')

print(df.head())

# Numerical values only
numerical_features = df.select_dtypes(include=[np.number])
X = numerical_features.drop(columns=['Price'])
y = numerical_features['Price']

print(X)

# Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Evaluating models w/ cross-validation
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R-squared': r2}
    print(f"{name} - MSE: {mse}, R-squared: {r2}")

# Printing results
print("\nModel Comparison:")
for model_name, metrics in results.items():
    print(f"{model_name}: MSE = {metrics['MSE']}, R-squared = {metrics['R-squared']}")

# Choosing the best model based on MSE and R-squared
best_model_name = min(results, key=lambda k: results[k]['MSE'])
best_model = models[best_model_name]

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

# Residual plot (to understand the error distribution)
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_best
plt.scatter(y_pred_best, residuals, color='blue')
plt.hlines(0, min(y_pred_best), max(y_pred_best), colors='red', linestyles='dashed')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals')
plt.title(f'Residual Plot ({best_model_name})')

plt.tight_layout()
plt.show()

# Saveing the best model to a file
joblib_file = f'best_model_{best_model_name.replace(" ", "_").lower()}_v3.pkl'
joblib.dump(best_model, joblib_file)
print(f"Best model saved to {joblib_file}")