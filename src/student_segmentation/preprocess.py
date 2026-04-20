import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df.columns = df.columns.str.lower().str.strip()

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['numberoffriends'] = pd.to_numeric(df['numberoffriends'], errors='coerce')

    df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
    df['age'] = df['age'].fillna(df['age'].median())

    return df

def scale_data(df):
    numeric_cols = df.select_dtypes(include='number').columns

    pt = PowerTransformer()
    df[numeric_cols] = pt.fit_transform(df[numeric_cols])

    scaler = StandardScaler()
    X = scaler.fit_transform(df[numeric_cols])

    return X, df