import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from src.config import RANDOM_STATE, INDOBERT_ABSA_MODEL_DIR, INDOBERT_ABSA_TOKENIZER_DIR
from src.dataset import get_absa_data_splits, ABSADataset

def train_absa():
    print("Mempersiapkan data splits ABSA...")
    X_train_text, X_train_aspect, y_train, X_test_text, X_test_aspect, y_test = get_absa_data_splits()
    
    # Load tokenizer IndoBERT
    model_name = 'indobenchmark/indobert-base-p1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Tokenisasi data pasangan kalimat (Teks + Aspek) -> True ABSA...")
    train_encodings = tokenizer(X_train_text, X_train_aspect, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(X_test_text, X_test_aspect, truncation=True, padding=True, max_length=128)
    
    train_dataset = ABSADataset(train_encodings, y_train)
    test_dataset = ABSADataset(test_encodings, y_test)
    
    print("Memuat pre-trained model IndoBERT Sequence Classification...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        return {'accuracy': acc, 'f1': f1}
        
    training_args = TrainingArguments(
        output_dir='./results_absa',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs_absa',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        seed=RANDOM_STATE,
        report_to='none'
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Memulai training IndoBERT ABSA...")
    trainer.train()
    
    print(f"Menyimpan model ke: {INDOBERT_ABSA_MODEL_DIR}...")
    model.save_pretrained(INDOBERT_ABSA_MODEL_DIR)
    tokenizer.save_pretrained(INDOBERT_ABSA_TOKENIZER_DIR)
    print("Training ABSA selesai dan berhasil disimpan!")

if __name__ == "__main__":
    train_absa()
