import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, Trainer, TrainingArguments
from src.config import (
    INDOBERT_ABSA_MODEL_DIR, INDOBERT_ABSA_TOKENIZER_DIR,
    INDOBERT_NER_MODEL_DIR, INDOBERT_NER_TOKENIZER_DIR
)
from src.dataset import get_absa_data_splits, get_ner_data_splits, ABSADataset, NERDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_all():
    print("=== EVALUASI MODEL ABSA INDOBERT ===")
    # Load splits
    _, _, _, X_test_text, X_test_aspect, y_test = get_absa_data_splits()
    
    tokenizer_absa = AutoTokenizer.from_pretrained(INDOBERT_ABSA_TOKENIZER_DIR)
    model_absa = AutoModelForSequenceClassification.from_pretrained(INDOBERT_ABSA_MODEL_DIR).to(device)
    
    test_encodings = tokenizer_absa(X_test_text, X_test_aspect, truncation=True, padding=True, max_length=128)
    test_dataset = ABSADataset(test_encodings, y_test)
    
    training_args = TrainingArguments(output_dir='./temp_eval_absa', report_to='none')
    trainer_absa = Trainer(model=model_absa, args=training_args)
    
    predictions = trainer_absa.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    print("\nClassification Report (ABSA):")
    print(classification_report(y_test, y_pred, target_names=['negatif', 'netral', 'positif'], zero_division=0))
    
    # Error Analysis ABSA
    error_cases = []
    inv_label_map = {0: 'negatif', 1: 'netral', 2: 'positif'}
    for text, true, pred in zip(X_test_text, y_test, y_pred):
        if true != pred:
            error_cases.append({
                'Text': text,
                'True': inv_label_map[true],
                'Pred': inv_label_map[pred]
            })
    df_errors = pd.DataFrame(error_cases)
    print(f"\nTotal Error ABSA: {len(df_errors)}")
    if len(df_errors) > 0:
        print(df_errors.head(10))
        
    print("\n" + "="*60 + "\n")
    print("=== EVALUASI MODEL NER INDOBERT ===")
    _, _, test_sentences, test_tags, label2id, id2label = get_ner_data_splits()
    
    tokenizer_ner = AutoTokenizer.from_pretrained(INDOBERT_NER_TOKENIZER_DIR, use_fast=True)
    model_ner = AutoModelForTokenClassification.from_pretrained(INDOBERT_NER_MODEL_DIR).to(device)
    
    def align_labels_with_tokens(sentences_words, tags_list):
        encodings = tokenizer_ner(sentences_words, is_split_into_words=True, truncation=True, padding=True, max_length=128)
        labels = []
        for i, label in enumerate(tags_list):
            word_ids = encodings.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        return encodings, labels

    test_encodings_ner, test_labels_ner = align_labels_with_tokens(test_sentences, test_tags)
    test_dataset_ner = NERDataset(test_encodings_ner, test_labels_ner)
    
    trainer_args_ner = TrainingArguments(output_dir='./temp_eval_ner', report_to='none')
    trainer_ner = Trainer(model=model_ner, args=trainer_args_ner)
    
    predictions_ner = trainer_ner.predict(test_dataset_ner)
    preds_flat = np.argmax(predictions_ner.predictions, axis=2)
    
    y_true_n = []
    y_pred_n = []
    for i, labels in enumerate(test_labels_ner):
        for j, val in enumerate(labels):
            if val != -100:
                y_true_n.append(val)
                y_pred_n.append(preds_flat[i][j])
                
    y_true_str = [id2label[val] for val in y_true_n]
    y_pred_str = [id2label[val] for val in y_pred_n]
    
    print("\nClassification Report (NER Token-Level):")
    print(classification_report(y_true_str, y_pred_str, zero_division=0))

if __name__ == "__main__":
    evaluate_all()
