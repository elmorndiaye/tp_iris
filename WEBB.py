import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Charger le modèle sauvegardé
with open("model_NB.pkl", "rb") as f:
    model_NB = pickle.load(f)

st.title("🌸 Prédiction d'espèce de fleur ")

# --- Présentation avant la prédiction ---
st.header("Présentation des espèces d'Iris")
st.write(
    """
    Le modèle est entraîné sur le célèbre **jeu de données Iris** de Fisher.
    Il distingue trois espèces de fleurs :
    - **Iris setosa** 🌼  
    - **Iris versicolor** 🌺  
    - **Iris virginica** 🌷  
    
    Voici leurs différences morphologiques principales :
    """
)

# Afficher une seule image avec les 3 types de fleurs
image = Image.open("image.png")  # ← mets ton image dans un dossier 'images'
st.image(image, caption="Les trois espèces d'Iris : Setosa, Versicolor et Virginica", use_container_width=True)

st.divider()  # ligne de séparation

# --- Formulaire de saisie ---
st.header("Tester le modèle")

sepal_length = st.number_input("Longueur du sépale (cm)")
sepal_width = st.number_input("Largeur du sépale (cm)")
petal_length = st.number_input("Longueur du pétale (cm)")
petal_width = st.number_input("Largeur du pétale (cm)")

if st.button("🔍 Prédire l'espèce"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model_NB.predict(input_data)
    st.success(f"🌿 Espèce prédite : **{prediction[0]}**")


