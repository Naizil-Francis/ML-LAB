import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer

# Step 1: Data Cleaning
data = pd.read_csv('housing.csv')
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
imputer = SimpleImputer(strategy='mean')
data['longitude'] = imputer.fit_transform(data[['longitude']])

# Step 2: Data Integration (if necessary)
df_housing = pd.read_csv('housing.csv')
df_additional = pd.read_csv('new.csv')
merged_df = pd.merge(df_housing, df_additional, on='longitude', how='inner')
print('Merged datasets:\n',merged_df)

# Step 3: Data Transformation
categorical_cols = ['ocean_proximity']
encoder = OneHotEncoder(drop='first', sparse=False)
encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]), columns=encoder.get_feature_names_out(categorical_cols))
numerical_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
scaler = StandardScaler()
scaled_cols = pd.DataFrame(scaler.fit_transform(data[numerical_cols]), columns=numerical_cols)
data = pd.concat([encoded_cols, scaled_cols], axis=1)

# Save the processed data to a new CSV file
data.to_csv('processed_data.csv', index=False)
print(data)