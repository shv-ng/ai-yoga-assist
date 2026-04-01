import pandas as pd
import os


def merge_datasets(
    existing_csv="data/poses.csv",
    kaggle_csv="data/kaggle_landmarks.csv",
    output_csv="data/poses_full.csv",
):
    if not os.path.exists(existing_csv):
        print(f"Error: Existing data file {existing_csv} not found.")
        return

    if not os.path.exists(kaggle_csv):
        print(
            f"Error: Kaggle landmarks file {kaggle_csv} not found. Run src/kaggle_to_landmarks.py first."
        )
        return

    print(f"Loading {existing_csv}...")
    df_existing = pd.read_csv(existing_csv)

    print(f"Loading {kaggle_csv}...")
    df_kaggle = pd.read_csv(kaggle_csv)

    print("Concatenating datasets...")
    df_full = pd.concat([df_existing, df_kaggle], ignore_index=True)

    print("\nFull Class Distribution:")
    print(df_full["label"].value_counts())

    df_full.to_csv(output_csv, index=False)
    print(f"\nSaved merged dataset to {output_csv}")
    print(f"Total rows: {len(df_full)}")


if __name__ == "__main__":
    merge_datasets()
