import pandas as pd
import requests
import os
import concurrent.futures
from tqdm import tqdm
import argparse

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
    
    # Skip if already exists (optional optimization, remove if re-download is needed)
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
            # print(f"Failed {image_id}: Status {response.status_code}")
            return None
    except Exception as e:
        # print(f"Error {image_id}: {e}")
        return None

def main(limit=5000):
    tsv_path = "fakeddit_multimodal_only_samples/multimodal_train.tsv"
    save_dir = "data/images"
    output_csv = "data/fakeddit_downloaded.csv"
    
    print(f"Loading {tsv_path}...")
    # on_bad_lines='skip' to handle potential parsing issues
    df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    
    # Filter: 2_way_label exists (not null)
    # Also ensuring image_url is valid string
    df = df[df['2_way_label'].notna()]
    df = df[df['image_url'].notna()]
    
    print(f"Total potential rows: {len(df)}")
    
    successful_rows = []
    
    # ThreadPoolExecutor
    # We submit tasks in batches or all at once. Since we have a limit,
    # let's try to be smart. If the file is huge, submitting all might use too much memory.
    # But Fakeddit samples might be reasonable (e.g. < 100k).
    # Let's assume we can convert to list of dicts and iterate.
    
    records = df.to_dict('records')
    
    print(f"Starting download (Target: {limit} images)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks? Or just enough?
        # Let's submit 2 * limit initially to buffer for failures.
        # If not enough, we can submit more, but simpler to just submit a large chunk.
        # Given user requirement "limit=5000", we stop SAVING after 5000.
        
        # Actually, let's submit all valid URL rows. 
        # But to avoid memory issues if df is huge, we can do it.
        # For this specific task, let's just map all.
        
        future_to_row = {executor.submit(download_image, row, save_dir): row for row in records}
        
        with tqdm(total=limit, unit="img") as pbar:
            for future in concurrent.futures.as_completed(future_to_row):
                result = future.result()
                if result is not None:
                    successful_rows.append(result)
                    pbar.update(1)
                    
                    if len(successful_rows) >= limit:
                        print("\nReached limit. Cancelling remaining tasks (if possible)...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
    
    print(f"\nDownloaded {len(successful_rows)} images.")
    
    # Save successful metadata
    if successful_rows:
        result_df = pd.DataFrame(successful_rows)
        result_df.to_csv(output_csv, index=False)
        print(f"Saved metadata to {output_csv}")
    else:
        print("No images downloaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5000, help="Max images to download")
    args = parser.parse_args()
    
    if not os.path.exists("data/images"):
        os.makedirs("data/images")
        
    main(limit=args.limit)
