# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess the dataset by importing the required libraries, loading the California Housing dataset, selecting the input features and target variables, splitting the data into training and testing sets, and standardizing the features and target values.

2.Train the regression models by creating an SGD Regressor wrapped with MultiOutputRegressor for multi-output prediction and fitting it on the scaled training data. Also train a Linear Regression model and a single-output SGD Regressor for comparison.

3.Make predictions and evaluate the models by predicting the outputs for the test data, converting the scaled predictions back to their original values, and calculating the Mean Squared Error (MSE) for overall performance and for each output.

4.Compare the results by displaying the actual and predicted values for sample test instances and comparing the MSE of the Linear Regression and SGD Regressor models to analyse their prediction accuracy.


## Program:
```python
/*
Program to implement Multi-Output Regression using SGDRegressor
and compare Linear Regression with SGDRegressor.
Developed by: K SriSaran Karthik
*/

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROGRAM 1: MULTI-OUTPUT REGRESSION USING SGDREGRESSOR
# ============================================================

# Load dataset
data = fetch_california_housing()

# Select the first three features
X = data.data[:, :3]

# Create multi-output target
Y = np.column_stack((data.target, data.data[:, 6]))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example X (first row):", X[0])
print("Example Y (first row):", Y[0])

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("\nTraining shapes:", X_train.shape, Y_train.shape)
print("Testing shapes:", X_test.shape, Y_test.shape)

# Scale features and targets
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print("\nScaled X_train mean:", X_train_scaled.mean(axis=0))
print("Scaled Y_train mean:", Y_train_scaled.mean(axis=0))

# Initialize and train the multi-output SGD model
sgd_multi = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    random_state=42
)

multi_output_sgd = MultiOutputRegressor(sgd_multi)
multi_output_sgd.fit(X_train_scaled, Y_train_scaled)

# Make predictions
Y_pred_scaled = multi_output_sgd.predict(X_test_scaled)

# Convert predictions to original units
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)

print("\nFirst 5 predictions (original units):")
print(Y_pred[:5])

# Calculate overall Mean Squared Error
mse_multi = mean_squared_error(Y_test_orig, Y_pred)
print("\nMean Squared Error (multi-output):", mse_multi)

# Calculate MSE for each output
mse_per_output = np.mean((Y_test_orig - Y_pred) ** 2, axis=0)
print("MSE per output:", mse_per_output)

# Display sample predictions
for i in range(5):
    print(f"\nExample {i + 1}")
    print("Inputs (raw):", X_test[i])
    print("True outputs:", Y_test_orig[i])
    print("Predicted outputs:", Y_pred[i])
    print("-" * 40)

print("\nDeveloped by: K SriSaran Karthik")


# ============================================================
# PROGRAM 2: LINEAR REGRESSION VS SGDREGRESSOR
# ============================================================

# Load dataset
data = fetch_california_housing()

X = data.data[:, :3]
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

# Train SGD Regressor model
sgd = SGDRegressor(
    max_iter=1000,
    tol=1e-3,
    eta0=0.01,
    learning_rate="constant",
    random_state=42
)

sgd.fit(X_train_scaled, y_train)

sgd_pred = sgd.predict(X_test_scaled)

# Compare model performance
lr_mse = mean_squared_error(y_test, lr_pred)
sgd_mse = mean_squared_error(y_test, sgd_pred)

print("\n" + "=" * 60)
print("LINEAR REGRESSION VS SGDREGRESSOR")
print("=" * 60)

print("Linear Regression MSE:", lr_mse)
print("SGDRegressor MSE:", sgd_mse)

if lr_mse < sgd_mse:
    print("Linear Regression performed better on the test set.")
elif sgd_mse < lr_mse:
    print("SGDRegressor performed better on the test set.")
else:
    print("Both models produced the same MSE.")

print("\nDeveloped by: K SriSaran Karthik")
```

## Output:
<img width="747" height="411" alt="Screenshot 2026-10-29 144125" src="https://github.com/user-attachments/assets/ee9baff6-d7f1-4345-94a3-f2d3ab685526" />
<img width="728" height="687" alt="Screenshot 2026-10-29 144132" src="https://github.com/user-attachments/assets/0b25edf1-4ad9-4a94-a20a-583ef99a4558" />
<img width="456" height="86" alt="Screenshot 2026-10-29 144140" src="https://github.com/user-attachments/assets/0b7a39b2-3fe2-4e4e-841d-122cecea873a" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
