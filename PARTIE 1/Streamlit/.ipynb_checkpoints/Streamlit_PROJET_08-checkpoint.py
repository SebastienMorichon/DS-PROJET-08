# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import pickle
import shap
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Charger le modèle, l'imputer et le scaler
with open("../Model/best_lightgbm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("../Model/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("../Model/imputer.pkl", "rb") as imputer_file:
    imputer = pickle.load(imputer_file)

# Charger les colonnes nécessaires
data = pd.read_csv("../Bases de données/Base De Donnée Prétraitée.csv")
required_columns = list(data.drop(columns=['TARGET']).columns)  # Colonnes nécessaires pour la prédiction

# Configuration de l'application Streamlit
st.title("Prédiction d'Acceptation de Prêt")
st.write("Analyse des données clients pour évaluer leurs probabilités d'acceptation de prêt.")

# Chargement d'un fichier Excel par l'utilisateur
uploaded_file = st.file_uploader("Téléversez un fichier Excel (.xlsx)", type=["xlsx"])

# Si un fichier est téléversé
if uploaded_file:
    try:
        # Lecture des données
        client_data = pd.read_excel(uploaded_file)
        st.write("Aperçu des données chargées :")
        st.dataframe(client_data.head())

        # Vérifier les colonnes manquantes
        missing_columns = [col for col in required_columns if col not in client_data.columns]
        if missing_columns:
            st.warning(f"Colonnes manquantes : {', '.join(missing_columns)}")
            for col in missing_columns:
                client_data[col] = 0  # Ajouter les colonnes manquantes avec des valeurs par défaut
            st.info("Les colonnes manquantes ont été ajoutées avec des valeurs par défaut (0).")

        # Conversion des colonnes pour éviter les erreurs de type
        client_data = client_data.convert_dtypes()  # Convertir automatiquement les types problématiques
        for col in required_columns:
            if col in client_data.columns:
                client_data[col] = pd.to_numeric(client_data[col], errors='coerce')  # Convertir en numérique

        client_data.fillna(0, inplace=True)  # Remplir les valeurs manquantes

        # Prétraitement des données
        st.info("Prétraitement des données...")
        data_to_predict = client_data[required_columns]
        data_imputed = imputer.transform(data_to_predict)
        normalized_data = scaler.transform(data_imputed)

        # Prédictions
        st.info("Calcul des prédictions...")
        predictions_proba = model.predict_proba(normalized_data)[:, 1]
        predictions = ["Accordé" if proba < 0.5 else "Refusé" for proba in predictions_proba]

        client_data["Probabilité de défaut"] = predictions_proba  # Ajouter la probabilité
        client_data["Décision"] = predictions  # Ajouter la décision

        # SHAP : importance globale des caractéristiques
        st.write("**Importance globale des caractéristiques :**")
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(normalized_data)
        shap_values_to_use = shap_values[1] if isinstance(shap_values, list) else shap_values
        mean_abs_shap_values = abs(shap_values_to_use).mean(axis=0)
        top_indices = mean_abs_shap_values.argsort()[-5:][::-1]  # Indices des 5 variables les plus importantes
        top_features = [required_columns[i] for i in top_indices]

        st.write(f"Les 5 principales caractéristiques sont : {', '.join(top_features)}")

        # Modifier les informations pour un client en conservant les autres variables
        st.write("**Modifier les informations pour voir l'impact :**")
        client_index = st.selectbox(
            "Sélectionnez un client :", 
            range(len(client_data)), 
            format_func=lambda x: f"Client {client_data['SK_ID_CURR'].iloc[x]}"
        )
        
        # Récupérer toutes les données du client sélectionné
        selected_client_data = client_data.iloc[client_index].copy()
        
        # Modifier uniquement les 5 principales caractéristiques
        st.write(f"**Client sélectionné : {client_data['SK_ID_CURR'].iloc[client_index]}**")
        for feature in top_features:
            new_value = st.number_input(
                f"Modifier {feature} :", 
                value=float(selected_client_data[feature]), 
                key=f"modify_{feature}"
            )
            selected_client_data[feature] = new_value  # Mettre à jour la valeur modifiée
        
        # Recalculer les prédictions avec les modifications
        if st.button("Recalculer avec les modifications"):
            # Convertir les données du client en DataFrame avec toutes les colonnes
            modified_data_df = pd.DataFrame([selected_client_data])
            
            # Prétraiter les données complètes du client
            modified_imputed = imputer.transform(modified_data_df[required_columns])  # Imputation des données
            modified_normalized = scaler.transform(modified_imputed)  # Normalisation des données
        
            # Prédire avec les données modifiées
            new_predictions_proba = model.predict_proba(modified_normalized)[:, 1]
            new_decision = "Accordé" if new_predictions_proba[0] < 0.5 else "Refusé"
        
            # Afficher les résultats recalculés
            st.write(f"**Résultats après modification :**")
            st.write(f"Probabilité de défaut : {new_predictions_proba[0]:.2f}")
            st.write(f"Nouvelle décision : {new_decision}")


      # Ajouter un nouveau client avec médiane pour les autres variables
        st.write("**Ajouter un nouveau client :**")
        
        # Initialiser les données pour un nouveau client avec la médiane des variables
        median_values = client_data[required_columns].median()  # Calculer la médiane pour chaque colonne
        new_client_data = median_values.copy()  # Initialiser les données du nouveau client avec la médiane
        
        # Permettre à l'utilisateur de modifier les 5 principales variables
        st.write("**Modifier les 5 principales variables :**")
        for feature in top_features:
            new_value = st.number_input(
                f"Valeur pour {feature} :", 
                value=float(median_values[feature]),  # Valeur initiale = médiane
                key=f"new_{feature}"
            )
            new_client_data[feature] = new_value  # Mettre à jour les valeurs modifiées
        
        # Recalculer les prédictions pour le nouveau client
        if st.button("Calculer le score pour ce nouveau client"):
            # Convertir les données du nouveau client en DataFrame
            new_client_df = pd.DataFrame([new_client_data])
        
            # Prétraiter les données complètes du nouveau client
            new_client_imputed = imputer.transform(new_client_df[required_columns])  # Imputation des données
            new_client_normalized = scaler.transform(new_client_imputed)  # Normalisation des données
        
            # Prédire avec les données du nouveau client
            new_client_proba = model.predict_proba(new_client_normalized)[:, 1]
            new_client_decision = "Accordé" if new_client_proba[0] < 0.5 else "Refusé"
        
            # Afficher les résultats pour le nouveau client
            st.write(f"**Résultats pour le nouveau client :**")
            st.write(f"Probabilité de défaut : {new_client_proba[0]:.2f}")
            st.write(f"Décision : {new_client_decision}")


    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement du fichier : {e}")
