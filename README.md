# Car price prediction model

## Purpose

When given the car parameters, the model predicts the car price based on AutoPlius site data for OpelAstra cars and it's direct competitors.

## Poccess

[step by step](https://docs.google.com/spreadsheets/d/1YO4mmn-raGmYBzZeasGXwMAk03m2vs_TX9yg0nIk85g/edit?usp=sharing)

## Outcome

### Final model
Model_v5: \
Best Model: Random Forest \
MSE: 1726833685311902.5 \
R-squared: 0.8451868631173116 \
best_model_random_forest_all_v5.pkl

Model takes the following car features: \
Title, Year, Type, Engine type, Gearbox_type, Mileage, City, Engine and Power.

Model output: Predicted price.

### Car 1

Price: 1250 \
PredictedPrice: 1117.95 \
Difference: -132.05 \
Difference, %: -11.81 

### Car 2

Price: 4750 \
PredictedPrice: 5625.98 \
Difference: 875.98 \
Difference, %: 15.57 

### Car 3

Price: 12000 \
PredictedPrice: 12452.41 \
Difference: 452.41 \
Difference, %: 3.63

### Model evaluation		     	    
Avg. model difference, %: 2.46
