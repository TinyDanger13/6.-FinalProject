import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('astra_and_competitors.csv')

# Numerical values only
numerical_features = df.select_dtypes(include=[np.number])
X = numerical_features.drop(columns=['Price'])
y = numerical_features['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# Ridge regression model with cross-validation and hyperparameter tuning
parameters = {'alpha': [0.1, 1.0, 10.0]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, parameters, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['alpha']
model = Ridge(alpha=best_alpha)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluating
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared: {r2}')

# Visualizing the predicted vs. actual prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs. Predicted Price')
plt.show()