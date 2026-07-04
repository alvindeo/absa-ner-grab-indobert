import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi Sastrawi
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
default_stopwords = set(factory_stopword.get_stop_words())
# Kata negasi/penting yang dikecualikan agar tidak mengubah polaritas sentimen
ignored_stopwords = {'tidak', 'jangan', 'kurang', 'bukan', 'belum', 'tidaklah'}
custom_stopwords = default_stopwords - ignored_stopwords

# Kamus slang/typo
slang_dict = {
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'bs': 'bisa', 'gk': 'tidak',
    'ga': 'tidak', 'gak': 'tidak', 'tdk': 'tidak', 'bgt': 'banget', 'bgs': 'bagus',
    'klo': 'kalau', 'kalo': 'kalau', 'sm': 'sama', 'dr': 'dari', 'jd': 'jadi',
    'jgn': 'jangan', 'nya': '', 'udh': 'sudah', 'sdh': 'sudah', 'tp': 'tapi',
    'drivernya': 'driver', 'aplikasinya': 'aplikasi', 'sy': 'saya', 'jg': 'juga',
    'kpd': 'kepada', 'krn': 'karena', 'bikin': 'buat'
}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 1. Case folding
    text = text.lower()
    # 2. Cleaning URL, mention, simbol, angka
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_slang(text):
    words = text.split()
    normalized = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized)

def preprocess(text):
    text = clean_text(text)
    text = normalize_slang(text)
    # Stopword removal dengan custom stopword list
    words = text.split()
    filtered = [w for w in words if w not in custom_stopwords]
    text = ' '.join(filtered)
    # Stemming Sastrawi
    text = stemmer.stem(text)
    return text

def tokenize(text):
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return nltk.word_tokenize(text)
