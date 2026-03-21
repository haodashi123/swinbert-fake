import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Configuration
INPUT_CSV = "data/fakeddit_with_captions.csv"
IMAGE_DIR = "data/images"
OUTPUT_TRAIN = "data/dataset_train.csv"
OUTPUT_VAL = "data/dataset_val.csv"
OUTPUT_TEST = "data/dataset_test.csv"
RANDOM_SEED = 42

def main():
    print(f"Reading data from {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"Total rows in CSV: {len(df)}")

    # 1. Image Existence Check
    print("Checking image existence...")
    
    def check_image_exists(row):
        # Assuming image filename corresponds to 'id' column
        # Adjust column name if necessary (e.g., 'image_id' or 'id')
        img_id = str(row['id'])
        img_path = os.path.join(IMAGE_DIR, f"{img_id}.jpg")
        return os.path.exists(img_path)

    # Ensure 'id' column exists
    if 'id' not in df.columns:
        print("Error: 'id' column not found in CSV. Available columns:", df.columns)
        return

    # Filter
    df['image_exists'] = df.apply(check_image_exists, axis=1)
    df_clean = df[df['image_exists']].drop(columns=['image_exists'])
    
    print(f"Valid samples (image exists): {len(df_clean)} (Dropped {len(df) - len(df_clean)})")

    if len(df_clean) == 0:
        print("Error: No valid samples found.")
        return

    # 2. Stratified Split
    # Target label: '2_way_label'
    if '2_way_label' not in df_clean.columns:
        print("Error: '2_way_label' column not found.")
        return

    print("Splitting dataset...")
    
    # First, split out Test set (10%)
    # Stratify based on 2_way_label
    df_temp, df_test = train_test_split(
        df_clean, 
        test_size=0.10, 
        random_state=RANDOM_SEED, 
        stratify=df_clean['2_way_label']
    )

    # Then, split remaining 90% into Train (80% total) and Val (10% total)
    # Val needs to be 1/9 of the remaining 90% to be 10% of total
    val_size_relative = 1/9 
    
    df_train, df_val = train_test_split(
        df_temp, 
        test_size=val_size_relative, 
        random_state=RANDOM_SEED, 
        stratify=df_temp['2_way_label']
    )

    # 3. Save
    print("Saving splits...")
    df_train.to_csv(OUTPUT_TRAIN, index=False)
    df_val.to_csv(OUTPUT_VAL, index=False)
    df_test.to_csv(OUTPUT_TEST, index=False)

    # 4. Statistics
    def print_stats(name, d):
        print(f"\n[{name}]")
        print(f"Count: {len(d)}")
        print("Label Distribution:")
        print(d['2_way_label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
        print(d['2_way_label'].value_counts())

    print_stats("Train Set", df_train)
    print_stats("Validation Set", df_val)
    print_stats("Test Set", df_test)

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
