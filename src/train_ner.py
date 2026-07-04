import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from src.config import RANDOM_STATE, INDOBERT_NER_MODEL_DIR, INDOBERT_NER_TOKENIZER_DIR
from src.dataset import get_ner_data_splits, NERDataset

def train_ner():
    print("Mempersiapkan data splits NER...")
    train_sentences, train_tags, test_sentences, test_tags, label2id, id2label = get_ner_data_splits()
    
    # Load tokenizer
    model_name = 'indobenchmark/indobert-base-p1'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    def align_labels_with_tokens(sentences_words, tags_list):
        encodings = tokenizer(sentences_words, is_split_into_words=True, truncation=True, padding=True, max_length=128)
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

    print("Menyelaraskan label token dengan subwords BERT...")
    train_encodings_ner, train_labels_ner = align_labels_with_tokens(train_sentences, train_tags)
    test_encodings_ner, test_labels_ner = align_labels_with_tokens(test_sentences, test_tags)
    
    train_dataset = NERDataset(train_encodings_ner, train_labels_ner)
    test_dataset = NERDataset(test_encodings_ner, test_labels_ner)
    
    print("Memuat pre-trained model IndoBERT Token Classification...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=4, id2label=id2label, label2id=label2id
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        y_true_flat = []
        y_pred_flat = []
        
        for i, label_seq in enumerate(labels):
            for j, val in enumerate(label_seq):
                if val != -100:
                    y_true_flat.append(id2label[val])
                    y_pred_flat.append(id2label[predictions[i][j]])
                    
        acc = accuracy_score(y_true_flat, y_pred_flat)
        f1 = f1_score(y_true_flat, y_pred_flat, average='weighted', zero_division=0)
        return {'accuracy': acc, 'f1': f1}
        
    training_args = TrainingArguments(
        output_dir='./results_ner',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs_ner',
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
    
    print("Memulai training IndoBERT NER...")
    trainer.train()
    
    print(f"Menyimpan model ke: {INDOBERT_NER_MODEL_DIR}...")
    model.save_pretrained(INDOBERT_NER_MODEL_DIR)
    tokenizer.save_pretrained(INDOBERT_NER_TOKENIZER_DIR)
    print("Training NER selesai dan berhasil disimpan!")

if __name__ == "__main__":
    train_ner()
