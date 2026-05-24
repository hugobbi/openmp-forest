import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    os.makedirs("processed", exist_ok=True)

    print("Loading raw data...")

    df_data = pd.read_csv("raw/data.csv", index_col=0).rename_axis("sample").reset_index()
    # feature_map = {feature: idx for idx, feature in enumerate(df_data.columns) if feature != "sample"}
    # df_data = df_data.rename(columns=feature_map)
    
    df_labels = pd.read_csv("raw/labels.csv", index_col=0).rename_axis("sample").reset_index()
    # use column at index 1 (second column) as the label/class column
    label_col = df_labels.columns[1]
    label_map = {label: idx for idx, label in enumerate(df_labels[label_col].unique())}
    print("Label mapping:", label_map)
    df_labels["label"] = df_labels[label_col].map(label_map)
    if label_col != "label":
        df_labels = df_labels.drop(columns=[label_col])

    df = pd.merge(df_data, df_labels, on="sample").drop(columns=["sample"])

    print("Merged DataFrame:\n", df.head())

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print("Train set:\n", df_train.head(), "\n\nTest set:\n", df_test.head())

    df_train.to_csv("processed/train.csv", index=False)
    df_test.to_csv("processed/test.csv", index=False)

    print("Data parsing completed. Train and test sets saved to 'processed/' directory.")