import pandas as pd
import requests
import os
import concurrent.futures
from tqdm import tqdm
import argparse
import random

def download_image(row, save_dir):
    """
    Downloads a single image.
    Returns the row if successful, None otherwise.
    """
    image_id = row['id']
    image_url = row['image_url']
    
    # Simple validation of URL
    if not isinstance(image_url, str) or not image_url.startswith('http'):
        return None

    save_path = os.path.join(save_dir, f"{image_id}.jpg")
    
    # Skip if already exists
    if os.path.exists(save_path):
        return row

    try:
        # User-agent to avoid some 403s
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(image_url, stream=True, timeout=10, headers=headers)
        
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return row
        else:
            return None
    except Exception as e:
        return None

def main(limit=5000):
    tsv_path = "fakeddit_multimodal_only_samples/multimodal_train.tsv"
    save_dir = "data/images"
    output_csv = "data/fakeddit_downloaded.csv"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Loading {tsv_path}...")
    try:
        df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: {tsv_path} not found.")
        return

    # Filter valid rows
    df = df[df['2_way_label'].notna()]
    df = df[df['image_url'].notna()]
    
    print(f"Total potential rows: {len(df)}")
    
    # Stratified Sampling
    df_real = df[df['2_way_label'] == 0]
    df_fake = df[df['2_way_label'] == 1]
    
    target_per_class = limit // 2
    
    print(f"Sampling target: {target_per_class} per class (Total limit: {limit})...")
    
    # Sample Real
    if len(df_real) > target_per_class:
        sampled_real = df_real.sample(n=target_per_class, random_state=42)
    else:
        print(f"Warning: Not enough Real samples ({len(df_real)} < {target_per_class}). Taking all.")
        sampled_real = df_real
        
    # Sample Fake
    if len(df_fake) > target_per_class:
        sampled_fake = df_fake.sample(n=target_per_class, random_state=42)
    else:
        print(f"Warning: Not enough Fake samples ({len(df_fake)} < {target_per_class}). Taking all.")
        sampled_fake = df_fake
        
    # Combine and Shuffle
    sampled_df = pd.concat([sampled_real, sampled_fake])
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final sampled dataset size: {len(sampled_df)}")
    print(f"Class distribution: {sampled_df['2_way_label'].value_counts().to_dict()}")
    
    # Download
    records = sampled_df.to_dict('records')
    successful_rows = []
    
    print(f"Starting download...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_row = {executor.submit(download_image, row, save_dir): row for row in records}
        
        with tqdm(total=len(records), unit="img") as pbar:
            for future in concurrent.futures.as_completed(future_to_row):
                result = future.result()
                if result is not None:
                    successful_rows.append(result)
                pbar.update(1)
    
    print(f"\nDownloaded {len(successful_rows)} images.")
    
    # Save successful metadata
    if successful_rows:
        result_df = pd.DataFrame(successful_rows)
        result_df.to_csv(output_csv, index=False)
        print(f"Saved metadata to {output_csv}")
        print(f"Final saved class distribution: {result_df['2_way_label'].value_counts().to_dict()}")
    else:
        print("No images downloaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000, help="Total samples to attempt (split evenly)")
    args = parser.parse_args()
    
    main(limit=args.limit)
