import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
print('Loading data...')
df = pd.read_csv('house_prices.csv')

# Extract BHK
df['BHK'] = df['Title'].str.extract(r'(\d+)\s*BHK')
df['BHK'] = pd.to_numeric(df['BHK'], errors='coerce')

# Convert Amount to Lakhs
def convert_amount(val):
    try:
        val = str(val).strip()
        if 'Cr' in val:
            return float(val.replace('Cr', '').strip()) * 100
        elif 'Lac' in val:
            return float(val.replace('Lac', '').strip())
        else:
            return np.nan
    except:
        return np.nan

df['Price_in_Lakhs'] = df['Amount(in rupees)'].apply(convert_amount)

# Carpet area
df['Carpet_Area_sqft'] = df['Carpet Area'].str.replace('sqft', '', regex=False).str.strip()
df['Carpet_Area_sqft'] = pd.to_numeric(df['Carpet_Area_sqft'], errors='coerce')

# Bathroom
df['Bathroom'] = df['Bathroom'].replace('> 10', '10')
df['Bathroom'] = pd.to_numeric(df['Bathroom'], errors='coerce')

# Keep useful columns and drop others
columns_to_drop = ['Index', 'Title', 'Description', 'Amount(in rupees)', 
                   'Price (in rupees)', 'Carpet Area', 'Floor', 'facing', 
                   'overlooking', 'Society', 'Balcony', 'Car Parking', 
                   'Super Area', 'Dimensions', 'Plot Area', 'Ownership']

for c in columns_to_drop:
    if c in df.columns:
        df = df.drop(columns=[c])

# Drop rows missing target or key features
print('Dropping rows with missing target/features...')
df = df.dropna(subset=['Price_in_Lakhs'])
df = df.dropna(subset=['BHK'])
df = df.dropna(subset=['Carpet_Area_sqft'])

# Fill some remaining missings
if 'Bathroom' in df.columns:
    df['Bathroom'] = df['Bathroom'].fillna(df['Bathroom'].median())
if 'Transaction' in df.columns:
    df['Transaction'] = df['Transaction'].fillna(df['Transaction'].mode()[0])
if 'Furnishing' in df.columns:
    df['Furnishing'] = df['Furnishing'].fillna('Unfurnished')
if 'Status' in df.columns:
    df['Status'] = df['Status'].fillna('Ready to Move')

# Remove outliers
df = df[(df['Price_in_Lakhs'] > 1) & (df['Price_in_Lakhs'] < 5000)]
df = df[(df['Carpet_Area_sqft'] > 100) & (df['Carpet_Area_sqft'] < 10000)]
df = df[df['BHK'] <= 5]

# Encode categoricals
le_location = LabelEncoder()
le_transaction = LabelEncoder()
le_furnishing = LabelEncoder()
le_status = LabelEncoder()

print('Fitting label encoders...')
df['location_encoded'] = le_location.fit_transform(df['location'])
df['Transaction_encoded'] = le_transaction.fit_transform(df['Transaction'])
df['Furnishing_encoded'] = le_furnishing.fit_transform(df['Furnishing'])
df['Status_encoded'] = le_status.fit_transform(df['Status'])

# Features and target
features = ['BHK', 'Carpet_Area_sqft', 'Bathroom', 'location_encoded',
            'Transaction_encoded', 'Furnishing_encoded', 'Status_encoded']
X = df[features]
y = df['Price_in_Lakhs']

# Train Random Forest
print('Training Random Forest...')
model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X, y)

# Save model and encoders
print('Saving model and encoders...')
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

encoders = {
    'le_location': le_location,
    'le_transaction': le_transaction,
    'le_furnishing': le_furnishing,
    'le_status': le_status,
    'features': features
}
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print('Done. Files created: rf_model.pkl, encoders.pkl')
