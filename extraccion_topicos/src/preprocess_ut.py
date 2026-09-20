from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

tokenizer = RegexpTokenizer(r'\w+')
STOPWORDS_ES = set(stopwords.words('spanish'))
stemmer = SnowballStemmer("spanish")

def text_preprocess(text):
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS_ES]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def apply_text_preprocess(x):
    return x.apply(text_preprocess)