import os
import pandas as pd
from google_play_scraper import reviews, Sort
from src.config import RAW_REVIEWS_PATH, DATA_DIR

def scrape_reviews(app_id='com.grabtaxi.passenger', count=600):
    print(f"Scraping {count} reviews for app: {app_id}...")
    result, _ = reviews(
        app_id,
        lang='id', 
        country='id', 
        sort=Sort.NEWEST, 
        count=count 
    )
    df_raw = pd.DataFrame(result)
    df_raw = df_raw[['content', 'score']]
    df_raw.rename(columns={'content': 'text'}, inplace=True)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df_raw.to_csv(RAW_REVIEWS_PATH, index=False)
    print(f"Scraping selesai. Data mentah disimpan di: {RAW_REVIEWS_PATH}")
    return df_raw

if __name__ == "__main__":
    scrape_reviews(count=5000)
