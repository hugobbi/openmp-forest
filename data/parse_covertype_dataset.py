import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("Loading data...")
    df_data = pd.read_csv("covertype/covtype.csv")
    print("Data loaded. Sample:\n", df_data.head())
    
    df_train, df_test = train_test_split(df_data, test_size=0.2, random_state=42)
    print("Train set:\n", df_train.head(), "\n\nTest set:\n", df_test.head())
    print("Train label distribution:\n", df_train["Cover_Type"].value_counts())
    print("Test label distribution:\n", df_test["Cover_Type"].value_counts())

    os.makedirs("processed", exist_ok=True)
    df_train.to_csv("processed/ctype_train.csv", index=False)
    df_test.to_csv("processed/ctype_test.csv", index=False)

if __name__ == "__main__":
    main()