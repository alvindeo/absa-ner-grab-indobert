import os

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

RAW_REVIEWS_PATH = os.path.join(DATA_DIR, 'raw_reviews.csv')
ABSA_DATASET_PATH = os.path.join(DATA_DIR, 'absa_dataset.csv')
NER_DATASET_PATH = os.path.join(DATA_DIR, 'ner_bio_dataset.tsv')

INDOBERT_ABSA_MODEL_DIR = os.path.join(MODELS_DIR, 'indobert_absa_model')
INDOBERT_ABSA_TOKENIZER_DIR = os.path.join(MODELS_DIR, 'indobert_absa_tokenizer')

INDOBERT_NER_MODEL_DIR = os.path.join(MODELS_DIR, 'indobert_ner_model')
INDOBERT_NER_TOKENIZER_DIR = os.path.join(MODELS_DIR, 'indobert_ner_tokenizer')

# Global configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
