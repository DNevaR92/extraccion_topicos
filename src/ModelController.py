import joblib
import os
import pandas as pd
from preprocess_ut import apply_text_preprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "resources", "models")

pipeline = joblib.load(os.path.join(MODELS_DIR, "tfidf_pipeline.joblib"))
tsvd = joblib.load(os.path.join(MODELS_DIR, "svd.joblib"))
modelo_svm = joblib.load(os.path.join(MODELS_DIR, "model.joblib"))

def predict(texto):
    texto_series = pd.Series([texto])          # <-- cambio clave
    tfidf_vec = pipeline.transform(texto_series)
    lsa_vec = tsvd.transform(tfidf_vec)
    ods_predicho = modelo_svm.predict(lsa_vec)
    return ods_predicho[0]