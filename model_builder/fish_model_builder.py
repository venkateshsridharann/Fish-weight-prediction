import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import joblib
import pickle

# Load the dataset
df = pd.read_csv("Fish.csv")

# Encode the categorical 'Species' column
df['Species'] = df['Species'].astype('category').cat.codes

# Define the feature columns and target column
X = df.drop('Weight', axis=1)
y = df['Weight']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred_rf = rf_model.predict(X_test)

# Calculate evaluation metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Mean Squared Error: {mse_rf}')
print(f'Random Forest R-squared: {r2_rf}')

# Save the trained model to a file
with open('fish_weight_predictor.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

print("Model saved to 'fish_weight_predictor.pkl'")
