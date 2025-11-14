import argparse
import json
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def detect_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    feature_df = df.drop(columns=[target])
    categorical_cols = [c for c in feature_df.columns if feature_df[c].dtype == 'object']
    numeric_cols = [c for c in feature_df.columns if np.issubdtype(feature_df[c].dtype, np.number)]
    return numeric_cols, categorical_cols


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols),
        ],
        remainder='drop'
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    return Pipeline(steps=[('preprocess', preprocessor), ('model', model)])


def main():
    parser = argparse.ArgumentParser(description='Train car price model and save pipeline + metadata.')
    parser.add_argument('--csv', required=True, help='Path to CSV file with car data')
    parser.add_argument('--target', default='price', help='Target column (default: price)')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test size ratio (default: 0.3)')
    parser.add_argument('--currency-rate', type=float, default=1.0, help='Multiplier to convert prices to MAD (default: 1.0)')
    parser.add_argument('--out-model', default='models/rf_model.joblib', help='Output path for saved pipeline')
    parser.add_argument('--out-meta', default='models/metadata.json', help='Output path for metadata JSON')

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)

    if args.target not in df.columns:
        raise ValueError(f'Target column "{args.target}" not in CSV columns: {list(df.columns)}')

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=[args.target])

    # Currency conversion for target
    df[args.target] = df[args.target].astype(float) * float(args.currency_rate)

    numeric_cols, categorical_cols = detect_feature_types(df, args.target)

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR2: {r2:.4f}')

    # Save model and metadata
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_model)

    metadata = {
        'target': args.target,
        'numeric_features': numeric_cols,
        'categorical_features': categorical_cols,
        'training_rows': int(X_train.shape[0]),
        'test_rows': int(X_test.shape[0]),
        'metrics': {'rmse': rmse, 'mae': mae, 'r2': r2},
        'currency_rate': float(args.currency_rate),
    }

    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    with out_meta.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f'Model saved to: {out_model}')
    print(f'Metadata saved to: {out_meta}')


if __name__ == '__main__':
    main()


