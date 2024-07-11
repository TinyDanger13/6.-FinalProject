import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model
model_file = 'best_model_random_forest_all_v5.pkl'
best_model = joblib.load(model_file)

# Identifying numerical and categorical features
numerical_features = ['Year', 'Milage', 'Engine', 'Power']
categorical_features = ['Type', 'Engine type', 'Gearbox_type']

# Create a DataFrame for the trial data (similar to before)
trial_data = {
    'Title': ['Opel Astra', 'Opel Astra', 'Opel Astra'],
    'Year': [2001, 2014, 2020],
    'Type': ['Universalas', 'Universalas', 'Universalas'],
    'Engine type': ['Benzinas', 'Dyzelinas', 'Dyzelinas'],
    'Gearbox_type': ['Mechaninė', 'Mechaninė', 'Automatinė'],
    'Milage': [240000, 290000, 111600],
    'City': ['Marijampolė', 'Utena', 'Vilnius'],
    'Engine': [1600, 1600, 1500],
    'Power': [74, 100, 90]
}

df_trial = pd.DataFrame(trial_data)

# Predict car prices using the loaded model
y_pred = best_model.predict(df_trial)

# Add predictions to the trial data DataFrame
df_trial['Predicted_Price'] = y_pred

# Display the trial data with predicted prices
print("Trial Data with Predicted Prices:")
print(df_trial[['Title', 'Year', 'Predicted_Price']])
