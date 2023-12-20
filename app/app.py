from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import pypyodbc as odbc
from flask import request
from flask import redirect
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import datetime

connection_string = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:echang.database.windows.net,1433;Database=car_insurance_db;Uid=SANHO_LEE;Pwd=DMSFALL2023%-;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
conn = odbc.connect(connection_string)

app = Flask(__name__)
app.secret_key = 'secret_key'

@app.route('/')
def login():
    return render_template('login.html')

@app.post('/login') 
def validate():
    ssn = request.form['ssn']
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Customer WHERE CustomerSSN = ?', [ssn])  # Use parameterized query to prevent SQL injection
    
    customer = cursor.fetchone()
    if not customer:
        flash('Invalid SSN.', 'error')
        return redirect(url_for('login'))
    

    # customer basic info
    customer_info = {
        'SSN': customer[0],
        'FirstName': customer[1],
        'LastName': customer[2], 
        'DOB': customer[3],
        'Gender': customer[4],
        'Email': customer[5],
        'Phone': customer[6],
        'LiscenseNumber': customer[7],
    }
    print('customer info:', customer_info)
    session['customer_info'] = customer_info

    # customer driving profile, used in learning algorithm to predict premium
    cursor.execute('SELECT * FROM Driving_History WHERE CustomerSSN = ?', [ssn])
    driving_history = cursor.fetchone()
    cursor.execute('SELECT * FROM Vehicle WHERE CustomerSSN = ?', [ssn])
    vehicle = cursor.fetchone()

    print("driving history: ", driving_history)
    print("vehicle info: ", vehicle)


    # Calculate age based on Date of Birth
    dob = customer[3]
    current_date = datetime.now()
    driver_age = current_date.year - dob.year - ((current_date.month, current_date.day) < (dob.month, dob.day))
    driving_profile = {
        'age': driver_age,
        'gender': customer[4],
        'mileage': vehicle[5],
        'trafficviolations': driving_history[1],
        'accidents': driving_history[2],
        'drivingexperience': driving_history[3]
    }
    session['driving_profile'] = driving_profile
    print("driving profile:" , driving_profile)
    return redirect(url_for('home'))

@app.route('/home')
def home():
    customer_info = session.get('customer_info', None)
    print('Customer info:', customer_info)
    return render_template('home.html', user=customer_info)

@app.route('/profile')
def profile():
    customer_info = session.get('customer_info', None)
    return render_template('profile.html', user=customer_info)

@app.route('/view-policy')
def view_policy():
    return render_template('view_policy.html')

@app.route('/file-claim')
def file_claim():
    return render_template('file_claim.html')

@app.route('/generate-quote')
def generate_quote():
    driving_profile = session.get('driving_profile', None)
    premium_pred = predict_premium(quote_calculation_model, driving_profile)

    return render_template('generate_quote.html', quote=premium_pred)

'''
@app.post('/')
def get_customer_info():
    ssn = request.json.get('ssn')  # Get SSN from the JSON body of the request
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Customer WHERE CustomerSSN = ?', [ssn])  # Use parameterized query to prevent SQL injection
    
    customer = cursor.fetchone()
    print(customer)

    # SSN, lastname, firstname, dob, gender, email, phone, lisc#
    # get address from ADDRESS table, addressline1, zip, custssn, agentssn, addressline2, state, city
    if customer:
        customer_info = {
            'SSN': customer[0],
            'FirstName': customer[1],
            'LastName': customer[2], 
            'DOB': customer[3],
            'Gender': customer[4],
            'Email': customer[5],
            'Phone': customer[6],
            'LiscenseNumber': customer[7],
        }
        redirect('/home')
        return jsonify(customer_info)
    else:
        return jsonify({'error': 'Customer not found'}), 404  # Return error message with HTTP status code 404
'''  
# machine learning algorithm to predict premium
def learning_model():
    sql_query = "SELECT * FROM Profile"
    # Close the connection
    df_profile = pd.read_sql_query(sql_query, conn)
    print(df_profile)
    # Convert DOB to Age
    df_profile['dob'] = pd.to_datetime(df_profile['dob'])
    df_profile['age'] = df_profile['dob'].apply(lambda x: datetime.now().year - x.year)

    # Selecting relevant features
    features = ['age', 'customergender', 'mileage', 'trafficviolations', 'accidents', 'drivingexperience']
    X = df_profile[features]
    y = df_profile['monthlypremium']

    # Preprocessing
    numeric_features = ['age', 'mileage', 'trafficviolations', 'accidents', 'drivingexperience']
    categorical_features = ['customergender']

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

    # Print actual vs predicted for the first 10 instances
    for actual, predicted in zip(y_test[:10], y_pred[:10]):
        print(f"Actual: {actual}, Predicted: {predicted}")

    # After training your model, use this function to make predictions
    return best_model

# Function to predict premium based on the best model chosen by machine learning algorithm
def predict_premium(model, driving_profile):

    # Create DataFrame from user input
    data = pd.DataFrame({
        'age': [driving_profile['age']],
        'customergender': [driving_profile['gender']],
        'mileage': [driving_profile['mileage']],
        'trafficviolations': [driving_profile['trafficviolations']],
        'accidents': [driving_profile['accidents']],
        'drivingexperience': [driving_profile['drivingexperience']]
    })

    # Make a prediction using the best model
    premium_pred = model.predict(data)
    return premium_pred
    # print(f"Predicted Monthly Premium for age {driving_profile['age']}: ${premium_pred[0]:.2f}")


if __name__ == '__main__':
    quote_calculation_model = learning_model()
    app.run(debug=True, port=3000)
    
    