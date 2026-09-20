import streamlit as st
import pandas as pd
import sys
import os

# Para que Python encuentre el módulo dentro de src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from ModelController import predict

st.set_page_config(page_title="Clasificador de ODS", page_icon="🌍")

st.title("Clasificador de textos por ODS")
st.write("Ingresa un texto y el modelo predice a qué Objetivo de Desarrollo Sostenible (ODS) está relacionado.")

modo = st.radio("¿Cómo quieres ingresar el texto?", ["Escribir texto", "Subir archivo (Excel/CSV)"])

if modo == "Escribir texto":
    texto = st.text_area("Texto a clasificar", height=150)
    if st.button("Predecir ODS"):
        if texto.strip() == "":
            st.warning("Escribe un texto antes de predecir.")
        else:
            ods = predict(texto)
            st.success(f"ODS predicho: **{ods}**")

else:
    archivo = st.file_uploader("Sube un archivo con una columna 'textos'", type=["csv", "xlsx"])
    if archivo is not None:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)

        if "textos" not in df.columns:
            st.error("El archivo debe tener una columna llamada 'textos'.")
        else:
            df["ODS_predicho"] = df["textos"].apply(predict)
            st.dataframe(df)
            st.download_button(
                "Descargar resultados en CSV",
                df.to_csv(index=False).encode("utf-8"),
                "resultados_ods.csv",
                "text/csv"
            )