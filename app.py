import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib  # Pour charger ton modèle
# import numpy as np
# Message de bienvenu
st.title("KALDANA Application Streamlit 🚀")
st.write("Bienvenue sur mon application déployée en ligne !")

# Charger le modèle
model = joblib.load("modele.pkl")  # Assure-toi d’avoir bien ajouté ton fichier modèle

# Charger les données
@st.cache_data
def load_data():
    return pd.read_csv("MathE_dataset.csv")

df = load_data()

# 📌 Interface améliorée
st.set_page_config(page_title="MathE Prediction App", layout="wide")

st.title("📊 MathE Prediction App 🚀")

# Ajouter des onglets pour la navigation
tab1, tab2, tab3 = st.tabs(["📂 Aperçu des Données", "📈 Visualisation", "🤖 Prédiction"])

# 📌 1. Onglet "Aperçu des Données"
with tab1:
    st.subheader("Aperçu du Jeu de Données")
    st.write(df.head())  # Afficher les premières lignes
    st.write("**Nombre total de lignes :**", df.shape[0])
    st.write("**Nombre total de colonnes :**", df.shape[1])
    st.write("**Valeurs manquantes :**", df.isnull().sum().sum())

# 📌 2. Onglet "Visualisation"
with tab2:
    st.subheader("Analyse et Visualisation des Données")

    # Sélectionner une colonne à afficher
    column = st.selectbox("Sélectionne une colonne :", df.columns)

    # Afficher un histogramme
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, bins=20, ax=ax)
    st.pyplot(fig)

# 📌 3. Onglet "Prédiction"
with tab3:
    st.subheader("Faites une Prédiction")
    
    # Entrée utilisateur (Exemple : valeurs de features)
    student_id = st.number_input("Student ID", min_value=0, step=1)
    question_id = st.number_input("Question ID", min_value=0, step=1)
    
    # Bouton de prédiction
    if st.button("Prédire"):
        features = np.array([[student_id, question_id]])  # Adapter selon ton modèle
        prediction = model.predict(features)
        st.success(f"Prédiction du modèle : {prediction[0]}")
