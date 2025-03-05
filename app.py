import streamlit as st
import pandas as pd
# Message de bienvenu
st.title("KALDANA Application MathE-Prediction 🚀")
st.write("Bienvenue sur mon application déployée en ligne !")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Pour charger ton modèle
import numpy as np

# Charger le modèle
model = joblib.load("modele.pkl")  # Assure-toi d’avoir bien ajouté ton fichier modèle

# Charger les données
@st.cache_data
def load_data():
    return pd.read_csv("MathE_dataset.csv")

df = load_data()

