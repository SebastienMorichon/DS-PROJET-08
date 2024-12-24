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
required_columns = list(data.drop(columns=['TARGET']).columns)

# Configuration de l'application Streamlit
st.title("Prédiction d'Acceptation de Prêt")
st.write("Analyse des données clients pour évaluer leurs probabilités d'acceptation de prêt.")

# Téléversement du fichier
uploaded_file = st.file_uploader("Téléversez un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        # Charger le fichier Excel
        client_data = pd.read_excel(uploaded_file)
        st.write("Aperçu des données chargées :")
        st.dataframe(client_data.head())

        # Vérifier les colonnes manquantes
        missing_columns = [col for col in required_columns if col not in client_data.columns]
        if missing_columns:
            st.warning(f"Les colonnes suivantes sont manquantes dans le fichier Excel : {', '.join(missing_columns)}")
            for col in missing_columns:
                client_data[col] = 0  # Ajouter les colonnes manquantes avec des valeurs par défaut
            st.info("Les colonnes manquantes ont été ajoutées avec des valeurs par défaut (0).")

        # Prétraitement des données
        st.info("Prétraitement des données...")
        data_to_predict = client_data[required_columns]
        data_imputed = imputer.transform(data_to_predict)
        normalized_data = scaler.transform(data_imputed)

        # Vérification des dimensions des données normalisées
        if normalized_data.shape[0] == 0:
            st.error("Le fichier téléversé ne contient aucune ligne valide après prétraitement. Veuillez vérifier le fichier.")
            st.stop()

        # Prédictions
        st.info("Calcul des prédictions...")
        predictions_proba = model.predict_proba(normalized_data)[:, 1]
        predictions = ["Accordé" if proba < 0.5 else "Refusé" for proba in predictions_proba]

        # Ajouter les résultats au DataFrame
        client_data["Probabilité de défaut"] = predictions_proba
        client_data["Décision"] = predictions

        # Affichage des résultats
        st.success("Résultats des prédictions :")
        st.dataframe(client_data[["SK_ID_CURR", "Probabilité de défaut", "Décision"]])

        # Explications globales avec SHAP
        st.write("**Importance globale des caractéristiques :**")
        explainer = shap.Explainer(model, feature_perturbation="tree_path_dependent")
        shap_values = explainer.shap_values(normalized_data)

        if isinstance(shap_values, list):
            shap_values_to_use = shap_values[1]
        else:
            shap_values_to_use = shap_values

        mean_abs_shap_values = abs(shap_values_to_use).mean(axis=0)
        top_indices = mean_abs_shap_values.argsort()[-10:][::-1]
        top_features = [required_columns[i] for i in top_indices]

        shap_values_top = shap_values_to_use[:, top_indices]
        normalized_data_top = normalized_data[:, top_indices]

        fig_summary = plt.figure()
        shap.summary_plot(shap_values_top, normalized_data_top, feature_names=top_features, show=False)
        st.pyplot(fig_summary)

        # Sélection du client
        st.write("**Sélectionnez un client pour voir les explications locales :**")
        client_index = st.selectbox(
            "Choisissez l'index du client",
            options=range(len(client_data)),
            format_func=lambda x: f"Client {client_data['SK_ID_CURR'].iloc[x]}"
        )

        st.write(f"**Décision pour le client {client_data['SK_ID_CURR'].iloc[client_index]} :**")
        st.write(f"Décision : {client_data['Décision'].iloc[client_index]}")
        st.write(f"Probabilité de défaut : {client_data['Probabilité de défaut'].iloc[client_index]:.2f}")

        # Distance au seuil
        threshold = 0.5
        distance_to_threshold = abs(predictions_proba[client_index] - threshold)
        st.write(f"Distance au seuil de décision : {distance_to_threshold:.2f}")

        # Informations descriptives du client
        st.write("**Informations descriptives du client sélectionné :**")
        st.dataframe(client_data.iloc[client_index][top_features])

        # Comparaison avec des clients similaires
        st.write("**Comparer avec des clients similaires :**")
        selected_feature = st.selectbox("Sélectionnez une variable pour filtrer :", top_features)
        feature_value = client_data[selected_feature].iloc[client_index]
        similar_clients = client_data[
            (client_data[selected_feature] > feature_value * 0.8) &
            (client_data[selected_feature] < feature_value * 1.2)
        ]
        st.write(f"**Clients similaires basés sur {selected_feature} :**")
        st.dataframe(similar_clients[["SK_ID_CURR", selected_feature]])

        # Modifier les informations pour voir l'impact
        st.write("**Modifier les informations pour voir l'impact :**")
        modified_data = client_data.iloc[client_index][top_features].copy()

        for col in top_features:
            new_value = st.number_input(f"Modifier {col} :", value=modified_data[col], key=f"modify_{col}")
            modified_data[col] = new_value

        if st.button("Recalculer avec les modifications"):
            data_to_predict_modified = pd.DataFrame([modified_data], columns=top_features)
            data_imputed_modified = imputer.transform(data_to_predict_modified)
            normalized_data_modified = scaler.transform(data_imputed_modified)

            predictions_proba_modified = model.predict_proba(normalized_data_modified)[:, 1]
            prediction_modified = "Accordé" if predictions_proba_modified[0] < 0.5 else "Refusé"

            st.write(f"**Résultats après modification :**")
            st.write(f"Probabilité de défaut : {predictions_proba_modified[0]:.2f}")
            st.write(f"Décision : {prediction_modified}")

        # Ajouter un nouveau client
        st.write("**Ajoutez un nouveau client :**")
        new_client_data = {}
        for col in top_features:
            new_client_data[col] = st.number_input(f"Valeur pour {col} :", value=0.0, key=f"new_{col}")

        if st.button("Calculer le score pour ce nouveau client"):
            try:
                new_client_df = pd.DataFrame([new_client_data])
                data_imputed_new = imputer.transform(new_client_df)
                normalized_data_new = scaler.transform(data_imputed_new)

                predictions_proba_new = model.predict_proba(normalized_data_new)[:, 1]
                prediction_new = "Accordé" if predictions_proba_new[0] < 0.5 else "Refusé"

                st.write(f"**Résultats pour le nouveau client :**")
                st.write(f"Probabilité de défaut : {predictions_proba_new[0]:.2f}")
                st.write(f"Décision : {prediction_new}")
            except Exception as e:
                st.error(f"Une erreur s'est produite : {e}")

    except Exception as e:
        st.error(f"Une erreur s'est produite lors du traitement du fichier : {e}")
