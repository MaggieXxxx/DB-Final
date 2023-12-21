# DBMS CSCI-GA.2433 Fall 2023 - Final Project

Team Members: Eugene Chang (N17404284), Sanho Lee (N15250101)

## Demo of Auto Insurance Facilitator Web Application
![demo gif](documentation/demo.gif)
## How to run the application

### 1. Install Required Packages
##### This application uses the `Flask` framework to serve static files.

First, you need to install the required Python packages.  Open your terminal or command prompt and install the following packages using `pip`:

```bash
pip install Flask pypyodbc pandas xgboost scikit-learn
```

### 2. Navigate to the Application Directory
Change into the 'app' directory where the application is stored:
```bash
cd path/to/app
```
### 3. Run the Application
Now, you can run the application using Python. In the terminal, execute the following command:
```bash
python3 app.py
```
* The application should now be running and accessible through a web browser at: `http://127.0.0.1:3000/`
* The application only allows sign-ins using SSN with customers stored in the `CUSTOMER` Table. For testing purposes, please use the `SSN 107-10-9797` 

### Additional Information
* Ensure you have Python 3 installed as python3 app.py assumes `Python 3`.
* If you encounter any issues with the database connection, it may be because of the database being run from a new IP address. We have made sure to allow all IP addresses, but if this is still the case, please try contacting us `via email`. The server may also be offline due to inactivity and may require a bit of time to become online again. 
* The application uses various libraries for data processing and machine learning. If you encounter any errors related to missing packages, ensure all the listed packages are installed correctly.

## Use Cases

### Viewing insurance profile
* Customers who have registered their information to the database through their auto insurnace providers are able to enter their SSN (not advisable for real use case to use SSN) to view insurance profile details: account holder, vehicle information, current contract information, driving history.

### Generating a Quote
* Customers are able to get a real-time quote for estimated monthly auto insurnace premium based on their information
* Quotes are generated in real-time based on the analytics conducted by a supervised machine learning model that is trained on a dataset of insurance profiles (age of driver, age of car, years of driving experience, gender, etc.)
#### [Link to Machine Learning model]() 

### Filing a claim
* Customers can skip the paperwork and hassle that comes with filing an auto insurance claim by going online and filling out a claim form. A representative agent will respond within 3-5 business days. 
* For the purposes of this project, the steps required to submit a claim are simplified. Of course, in real life, it is not as easy as this.

## Using pyodbc
For this project, we used `pyodbc`, a Python ODBC, in order to connect to the Azure SQL server housing our Auto Insurance database tables. Below are examples of using pyodbc:
### Using SELECT to get CUSTOMER info
* Using a `SELECT` query with pyodbc, the Flask app is able to retrieve the Customer information (along with other information from the DB) and work with the data

##### Example code from `app.py`
```Python
cursor = conn.cursor()
cursor.execute('SELECT * FROM Customer WHERE CustomerSSN = ?', [ssn])  # Use parameterized query to prevent SQL injection
customer = cursor.fetchone()
```
### Using INSERT form submit a new CLAIM
* Whenever a user submits a new claim, the client asynchronously sends a `POST` request to the API endpoint `/file-claim`
* The server retrieves this data in a JSON format, and using an `INSERT` query with pyodbc, it adds a new `CLAIM` tuple into the Azure SQL Database.
##### Example code from `app.py`
```python
cursor.execute('INSERT INTO Claim (ClaimID, AccidentDate, AccidentDesc, ClaimAmount, Status, CustomerSSN, ContractID) VALUES (?, ?, ?, ?, ?, ?, ?)', new_claim)
conn.commit()
```