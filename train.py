import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# ================== 1. LOAD DATA ======================
df = pd.read_csv(r"C:\My_project\PriceMyRide\data\cardekho.csv")

# ================== 2. DROP USELESS COLUMN ============
df = df.drop('name', axis=1)

# ================== 3. FIX BLANK VALUES ===============
df = df.replace(" ", np.nan)

numeric_cols = [
    'km_driven',
    'mileage(km/ltr/kg)',
    'engine',
    'max_power',
    'seats'
]

# Convert numeric columns to float
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values with median
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ================== 4. FEATURE ENGINEERING ============
df['car_age'] = 2025 - df['year']
df = df.drop('year', axis=1)

# ================== 5. SPLIT FEATURES/TARGET ==========
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# ================== 6. CATEGORICAL COLUMNS ============
categorical_columns = [
    'fuel',
    'seller_type',
    'transmission',
    'owner'
]

# ================== 7. PREPROCESSOR ===================
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

# ================== 8. MODEL ==========================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', model)
])

# ================== 9. TRAIN/TEST SPLIT ===============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================== 10. TRAIN MODEL ====================
pipeline.fit(X_train, y_train)

# ================== 11. EVALUATE =======================
y_pred = pipeline.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# ================== 12. SAVE MODEL =====================
joblib.dump(pipeline, "model.pkl")

print("Model saved successfully as model.pkl!")
