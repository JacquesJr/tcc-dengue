"""Random forest model."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

def train_random_forest(df, target_col='DENG_CASES_COUNT'):
    """Random forest model."""
    x = df.drop(columns=[target_col, 'ID_MUNICIP'])
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    importancias = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
    return model, mse, importancias

def prever_casos(df):
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Ensure target variable exists
    if 'DENG_CASES' not in df.columns:
        raise ValueError("Target variable 'DENG_CASES' not found in dataframe")
    
    # Columns we always want to drop (non-feature columns)
    always_drop = ['DENG_CASES']  # We'll extract this as our target variable
    
    # Explicitly identify date columns that need to be dropped
    date_columns = []
    if 'DT_NOTIFIC' in df.columns:
        date_columns.append('DT_NOTIFIC')
    if 'Data Medicao' in df.columns:
        date_columns.append('Data Medicao')
    
    # Find any other non-numeric columns (except lag features)
    non_numeric_cols = []
    for col in df.columns:
        # Skip lag features which are numeric and important
        if col.startswith('lag_'):
            continue
        # Skip columns we're already planning to drop
        if col in always_drop or col in date_columns:
            continue
        # Check if column is non-numeric
        if df[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df[col]):
            non_numeric_cols.append(col)
    
    # Combine all columns to drop
    cols_to_drop = always_drop + date_columns + non_numeric_cols
    
    # Only include columns to drop that actually exist in the dataframe
    cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    # Extract features and target
    y = df['DENG_CASES']
    
    # Remove target and non-numeric columns from features
    X = df.drop(columns=cols_to_drop)
    
    # Verify we have numeric data only
    numeric_cols = X.select_dtypes(include=['number']).columns
    X = X[numeric_cols]
    
    # Split into train and test
    X_train = X[:-12]
    y_train = y[:-12]
    X_test = X[-12:]
    y_test = y[-12:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    return model, y_test, y_pred, mae
