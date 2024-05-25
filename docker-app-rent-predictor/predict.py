import joblib
import pandas as pd
import numpy as np
import re
import requests
import warnings
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom transformer for KMeans clustering
class KMeansTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
    
    def fit(self, X, y=None):
        self.kmeans.fit(X[['Latitude', 'Longitude']])
        return self
    
    def transform(self, X):
        zones = self.kmeans.predict(X[['Latitude', 'Longitude']])
        X = X.copy()
        X['Zone'] = zones
        return X

# Custom transformer for Label Encoding 'Street'
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.classes_ = None
    
    def fit(self, X, y=None):
        self.label_encoder.fit(np.append(X, 'unknown'))
        self.classes_ = self.label_encoder.classes_
        return self
    
    def transform(self, X):
        X_encoded = X.apply(lambda x: x if x in self.classes_ else 'unknown')
        return self.label_encoder.transform(X_encoded)
    
    def inverse_transform(self, X):
        return self.label_encoder.inverse_transform(X)

# Function to geocode an address using the Swisstopo API
def geocode_address(street, zip_code, city):
    address = f"{street}, {zip_code}, {city}"
    params = {
        'searchText': address,
        'type': 'locations',
        'sr': '2056',
        'limit': 1
    }
    response = requests.get("https://api3.geo.admin.ch/rest/services/api/SearchServer", params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            location = data['results'][0]['attrs']
            return location['lat'], location['lon']
    return None, None

# Function to get user input
def get_user_input():
    try:
        rooms = float(input("Enter number of rooms: "))
        area = float(input("Enter area in square meters: "))
        street = input("Enter street name: ")
        region = input("Enter region: ")
        city = input("Enter city: ")
        zip_code = input("Enter zip code: ")

        latitude, longitude = geocode_address(street, zip_code, city)
        if latitude is None or longitude is None:
            print("Failed to geocode address.")
            return None

        return {
            'Rooms': rooms,
            'Area': area,
            'Street': street,
            'Region': region,
            'City': city,
            'Zip': zip_code,
            'Latitude': latitude,
            'Longitude': longitude
        }
    except ValueError as e:
        print(f"Invalid input: {e}")
        return None

# Function to make predictions with new data
def predict_new_data(pipeline, label_encoder, new_data):
    new_data = new_data.copy()
    new_data['Street'] = label_encoder.transform(new_data['Street'])
    return pipeline.predict(new_data)

# Function to train and save the model with progress bar
def train_and_save_model():
    # Load dataset
    data = pd.read_csv('data_immo_geocoded.csv')

    # Remove the ID column if present
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    # Remove house numbers from the 'Street' column
    data['Street'] = data['Street'].str.replace(r'\d+', '', regex=True).str.strip()

    # Remove outliers in the 'Price' column using IQR
    Q1 = data['Price'].quantile(0.25)
    Q3 = data['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data['Price'] >= lower_bound) & (data['Price'] <= upper_bound)]

    # Remove all missing values
    filtered_data = filtered_data.dropna()

    # Encode 'Street' using custom label encoding
    label_encoder = CustomLabelEncoder()
    filtered_data['Street'] = label_encoder.fit_transform(filtered_data['Street'])

    # Initialize encoders
    onehot_encoder = OneHotEncoder(drop='first', handle_unknown='ignore')

    # Define the Preprocessing Pipeline
    def get_preprocessor():
        numerical_features = ['Rooms', 'Area']
        categorical_features = ['Street', 'Region', 'City', 'Zone', 'Zip']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), numerical_features),
                ('cat', onehot_encoder, categorical_features)
            ],
            remainder='passthrough'
        )
        return preprocessor

    # Define the model
    model = RandomForestRegressor(n_jobs=-1)

    # Cross-Validation
    X = filtered_data.drop(columns=['Price'])
    y = filtered_data['Price']

    # Create the pipeline with KMeansTransformer
    pipeline = Pipeline(steps=[
        ('kmeans', KMeansTransformer()),
        ('preprocessor', get_preprocessor()),
        ('model', model)
    ])

    # Define Scoring Metrics
    scoring = {
        'MAE': make_scorer(mean_absolute_error),
        'RMSE': make_scorer(mean_squared_error, greater_is_better=False),
        'R2': make_scorer(r2_score)
    }

    # Perform Cross-Validation with progress bar
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    print("Performing cross-validation...")
    mae_scores, rmse_scores, r2_scores = [], [], []

    for train_index, test_index in tqdm(cv.split(X), total=cv.get_n_splits(), desc="Cross-Validation Progress"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))

    # Calculate adjusted R2 score
    n = len(X)
    p = X.shape[1]
    adjusted_r2_scores = [1 - (1 - r2) * (n - 1) / (n - p - 1) for r2 in r2_scores]

    # Display Results
    print(f"MAE: {np.mean(mae_scores):.2f}")
    print(f"RMSE: {np.mean(rmse_scores):.2f}")
    print(f"R2: {np.mean(r2_scores):.2f}")
    print(f"Adjusted R2: {np.mean(adjusted_r2_scores):.2f}")

    # Training the pipeline with the full data
    print("Training the model with full data...")
    pipeline.fit(X, y)

    # Save the trained pipeline
    joblib.dump(pipeline, 'real_estate_pipeline.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("Model retrained and saved successfully.")

# Main function
def main():
    while True:
        choice = input("Choose an option:\n1. Make a prediction\n2. Retrain the model\nEnter 1 or 2: ")
        if choice == '1':
            try:
                pipeline = joblib.load('real_estate_pipeline.pkl')
                label_encoder = joblib.load('label_encoder.pkl')
            except FileNotFoundError:
                print("Model files not found. Please retrain the model first.")
                continue
            
            user_input = get_user_input()
            if user_input is None:
                continue

            # Convert user input to DataFrame
            input_df = pd.DataFrame([user_input])

            # Make prediction
            prediction = predict_new_data(pipeline, label_encoder, input_df)
            print(f"Predicted price: {prediction[0]:.2f}")
        
        elif choice == '2':
            confirm = input("This will overwrite the existing model files. Are you sure? (yes/no): ")
            if confirm.lower() == 'yes':
                train_and_save_model()
            else:
                print("Model retraining cancelled.")
        
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
