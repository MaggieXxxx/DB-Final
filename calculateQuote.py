import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from datetime import datetime

# Load data from the Profile table
df_profile = pd.read_csv('profile.csv')

# Convert DOB to Age
df_profile['DOB'] = pd.to_datetime(df_profile['DOB'])
df_profile['Age'] = df_profile['DOB'].apply(lambda x: datetime.now().year - x.year)

# Selecting relevant features
features = ['Age', 'CustomerGender', 'Mileage', 'TrafficViolations', 'Accidents', 'DrivingExperience']
X = df_profile[features]
y = df_profile['MonthlyPremium']

# Preprocessing
numeric_features = ['Age', 'Mileage', 'TrafficViolations', 'Accidents', 'DrivingExperience']
categorical_features = ['CustomerGender']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])

# Define the model with XGBoost regressor
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter grid
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.1]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model using grid search
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Use the best model for predictions
best_model = grid_search.best_estimator_

# Predict and evaluate
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error with Best Model: {mse}")

# Scatter plot for actual vs predicted
#plt.scatter(y_test, y_pred)
#plt.xlabel("Actual Monthly Premium")
#plt.ylabel("Predicted Monthly Premium")
#plt.title("Actual vs Predicted Monthly Premium")
#plt.show()

# Print actual vs predicted for the first 10 instances
for actual, predicted in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {actual}, Predicted: {predicted}")

# Function to predict premium based on user input
def predict_premium(model):
    # User input for each feature
    age = int(input("Enter Age: "))
    customer_gender = input("Enter Gender (e.g., M, F): ")
    mileage = int(input("Enter Mileage: "))
    traffic_violations = int(input("Enter Number of Traffic Violations: "))
    accidents = int(input("Enter Number of Accidents: "))
    driving_experience = int(input("Enter Driving Experience in Years: "))

    # Create DataFrame from user input
    data = pd.DataFrame({
        'Age': [age],
        'CustomerGender': [customer_gender],
        'Mileage': [mileage],
        'TrafficViolations': [traffic_violations],
        'Accidents': [accidents],
        'DrivingExperience': [driving_experience]
    })

    # Make a prediction using the best model
    premium_pred = model.predict(data)
    print(f"Predicted Monthly Premium: ${premium_pred[0]:.2f}")

# After training your model, use this function to make predictions
predict_premium(best_model)
