import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from src.config import (
    INDOBERT_ABSA_MODEL_DIR, INDOBERT_ABSA_TOKENIZER_DIR,
    INDOBERT_NER_MODEL_DIR, INDOBERT_NER_TOKENIZER_DIR
)
from src.preprocessing import preprocess
from src.labeling import auto_label_absa

# Global model pointers
tokenizer_absa = None
model_absa = None
tokenizer_ner = None
model_ner = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    global tokenizer_absa, model_absa, tokenizer_ner, model_ner
    if model_absa is None:
        print("Memuat model IndoBERT ABSA...")
        tokenizer_absa = AutoTokenizer.from_pretrained(INDOBERT_ABSA_TOKENIZER_DIR)
        model_absa = AutoModelForSequenceClassification.from_pretrained(INDOBERT_ABSA_MODEL_DIR).to(device)
        model_absa.eval()
        
    if model_ner is None:
        print("Memuat model IndoBERT NER...")
        tokenizer_ner = AutoTokenizer.from_pretrained(INDOBERT_NER_TOKENIZER_DIR, use_fast=True)
        model_ner = AutoModelForTokenClassification.from_pretrained(INDOBERT_NER_MODEL_DIR).to(device)
        model_ner.eval()

def predict_absa(text):
    load_models()
    clean = preprocess(text)
    # Deteksi aspek statis
    aspect, _ = auto_label_absa(clean)
    
    # Prediksi sentimen dengan IndoBERT pasangan kalimat (Teks + Aspek)
    inputs = tokenizer_absa(clean, aspect, return_tensors='pt', truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    inv_label_map = {0: 'negatif', 1: 'netral', 2: 'positif'}
    with torch.no_grad():
        outputs = model_absa(**inputs)
    pred_idx = torch.argmax(outputs.logits, dim=1).item()
    sentiment = inv_label_map[pred_idx]
    
    return aspect, sentiment

def predict_ner(text):
    load_models()
    clean = preprocess(text)
    tokens = clean.split()
    if not tokens:
        return ""
        
    inputs = tokenizer_ner(tokens, is_split_into_words=True, return_tensors='pt', truncation=True, padding=True, max_length=128)
    word_ids = inputs.word_ids()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model_ner(**inputs)
        
    preds = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    if not isinstance(preds, list):
        preds = [preds]
        
    result = []
    previous_word_idx = None
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            label_name = model_ner.config.id2label[preds[i]]
            result.append(f"{tokens[word_idx]} [{label_name}]")
        previous_word_idx = word_idx
        
    return " ".join(result)

if __name__ == "__main__":
    test_sentences = [
        "Saya sangat kecewa dengan harga grabcar sekarang sangat mahal.",
        "Aplikasi grab sangat mudah digunakan, drivernya ramah dan top.",
        "Gojek dan maxim lebih murah layanannya dibanding ini."
    ]
    print("\n=== UJI COBA PREDIKSI DENGAN INDOBERT ===")
    for text in test_sentences:
        asp, sent = predict_absa(text)
        ner = predict_ner(text)
        print(f"Teks: {text}")
        print(f"ABSA -> Aspek: {asp} | Sentimen: {sent}")
        print(f"NER  -> {ner}")
        print("-" * 50)
