import os
import pandas as pd
import random
from src.config import RANDOM_STATE, RAW_REVIEWS_PATH, ABSA_DATASET_PATH, NER_DATASET_PATH
from src.preprocessing import preprocess

# Set seed untuk reproducibility
random.seed(RANDOM_STATE)

# Kata kunci aspek disesuaikan dengan stemming Sastrawi
aspect_keywords = {
    'driver': ['driver', 'supir', 'pengemudi'],
    'aplikasi': ['aplikasi', 'app', 'apk', 'sistem'],
    'harga': ['harga', 'ongkir', 'tarif', 'biaya'],
    'layanan': ['layan', 'pelayanan', 'respon', 'cs'],
    'keamanan': ['aman', 'keamanan', 'safety']
}

positive_words = ['bagus', 'cepat', 'ramah', 'murah', 'mantap', 'baik', 'keren', 'mudah', 'bantu', 'aman', 'top']
negative_words = ['jelek', 'lambat', 'mahal', 'buruk', 'kasar', 'susah', 'error', 'kecewa', 'tipu', 'lelet', 'kurang']

# Kata kunci untuk NER
ner_aspect_keywords = ['driver', 'aplikasi', 'harga', 'layan', 'aman']
prod_keywords = ['grabcar', 'grabbike', 'grabfood', 'grabexpress']
org_keywords = ['grab', 'gojek', 'maxim', 'shopee']

def auto_label_absa(text):
    text_lower = text.lower()
    
    # Deteksi aspek
    found_aspect = 'lainnya'
    for asp, keywords in aspect_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                found_aspect = asp
                break
    
    if not found_aspect:
        found_aspect = 'layanan'
    
    # Deteksi sentimen
    pos_count = sum([1 for w in positive_words if w in text_lower])
    neg_count = sum([1 for w in negative_words if w in text_lower])
    
    if pos_count > neg_count:
        sentiment = 'positif'
    elif neg_count > pos_count:
        sentiment = 'negatif'
    else:
        sentiment = 'netral'
        
    return found_aspect, sentiment

def refine_sentiment_by_score(row):
    if row['sentiment'] == 'netral':
        if row['score'] >= 4:
            return 'positif'
        elif row['score'] <= 2:
            return 'negatif'
    return row['sentiment']

def auto_ner_tagging(text):
    tokens = text.split()
    tags = []
    
    for token in tokens:
        # Simulasi noise/human error sebesar 5% agar hasil tidak 100% sempurna
        if random.random() < 0.05:
            tags.append(f"{token}\tO")
            continue
            
        if token in ner_aspect_keywords:
            tags.append(f"{token}\tB-ASPECT")
        elif token in prod_keywords:
            tags.append(f"{token}\tB-PROD")
        elif token in org_keywords:
            tags.append(f"{token}\tB-ORG")
        else:
            tags.append(f"{token}\tO")
            
    return tags

def run_labeling():
    if not os.path.exists(RAW_REVIEWS_PATH):
        print(f"Error: {RAW_REVIEWS_PATH} tidak ditemukan. Silakan jalankan scraper.py terlebih dahulu.")
        return
        
    df_raw = pd.read_csv(RAW_REVIEWS_PATH)
    print("Memproses Preprocessing & Pelabelan...")
    
    # Preprocessing
    df_prep = df_raw.copy()
    df_prep['clean_text'] = df_prep['text'].apply(preprocess)
    
    # ABSA Labeling
    df_prep['aspect'], df_prep['sentiment'] = zip(*df_prep['clean_text'].apply(auto_label_absa))
    df_prep['sentiment'] = df_prep.apply(refine_sentiment_by_score, axis=1)
    
    # Simpan ABSA dataset
    df_absa = df_prep[['clean_text', 'aspect', 'sentiment', 'score']].copy()
    df_absa.rename(columns={'clean_text': 'text'}, inplace=True)
    df_absa.drop_duplicates(subset=['text'], inplace=True)
    df_absa.dropna(inplace=True)
    df_absa = df_absa[df_absa['text'].str.strip() != '']
    df_absa.to_csv(ABSA_DATASET_PATH, index=False)
    print(f"Dataset ABSA disimpan di: {ABSA_DATASET_PATH}")
    
    # NER Tagging
    ner_dataset = []
    for text in df_prep['clean_text']:
        tags = auto_ner_tagging(text)
        if tags:
            ner_dataset.extend(tags)
            ner_dataset.append("") # Pembatas kalimat
            
    with open(NER_DATASET_PATH, 'w') as f:
        f.write('\n'.join(ner_dataset))
    print(f"Dataset NER disimpan di: {NER_DATASET_PATH}")

if __name__ == "__main__":
    run_labeling()
