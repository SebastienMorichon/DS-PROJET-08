# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import pickle
import shap
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Chargement des modèles et des outils nécessaires
with open("Model/best_lightgbm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("Model/scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("Model/imputer.pkl", "rb") as imputer_file:
    imputer = pickle.load(imputer_file)

# Chargement des colonnes nécessaires
data = pd.read_csv("Bases de données/Base De Donnée Prétraitée.csv")
required_columns = list(data.drop(columns=['TARGET']).columns)

# Configuration du menu latéral
menu = st.sidebar.selectbox(
    "Menu", 
    ["Analyse du fichier", "Modification des données du client", "Ajout d'un nouveau client", "Graphique", "Positionnement du client"]
)

# Téléversement du fichier client
uploaded_file = st.sidebar.file_uploader("Téléversez un fichier Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    # Lecture des données
    client_data = pd.read_excel(uploaded_file)

    # Ajout des colonnes manquantes
    missing_columns = [col for col in required_columns if col not in client_data.columns]
    for col in missing_columns:
        client_data[col] = 0

    # Nettoyage et remplissage des données
    client_data = client_data.convert_dtypes()
    for col in required_columns:
        if col in client_data.columns:
            client_data[col] = pd.to_numeric(client_data[col], errors='coerce')
    client_data.fillna(0, inplace=True)

    # Calcul des prédictions
    data_imputed = imputer.transform(client_data[required_columns])
    normalized_data = scaler.transform(data_imputed)
    predictions_proba = model.predict_proba(normalized_data)[:, 1]
    predictions = ["Accordé" if proba < 0.5 else "Refusé" for proba in predictions_proba]

    # Ajout des colonnes de résultats
    client_data["Probabilité de défaut"] = predictions_proba
    client_data["Décision"] = predictions

    # Extraction des caractéristiques principales
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(normalized_data)
    shap_values_to_use = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs_shap_values = abs(shap_values_to_use).mean(axis=0)
    top_indices = mean_abs_shap_values.argsort()[-10:][::-1]
    top_features = [required_columns[i] for i in top_indices]

    # Gestion des onglets
    if menu == "Analyse du fichier":
        st.title("Analyse du fichier")

        # Sélection d'un client pour afficher des informations spécifiques
        client_index = st.selectbox(
            "Sélectionnez un client pour afficher ses informations :", 
            range(len(client_data)), 
            format_func=lambda x: f"Client {client_data['SK_ID_CURR'].iloc[x]}"
        )

        # Extraire les informations du client sélectionné
        selected_client = client_data.iloc[client_index]

        # Afficher des informations spécifiques au client
        st.write(f"### Informations spécifiques pour le client {selected_client['SK_ID_CURR']}")
        st.write(f"- **Probabilité de défaut :** {selected_client['Probabilité de défaut']:.2f}")
        st.write(f"- **Décision :** {'✅ Accordé' if selected_client['Décision'] == 'Accordé' else '❌ Refusé'}")

        # Afficher les 5 principales caractéristiques du client
        st.write("#### 5 principales caractéristiques :")
        for feature in top_features[:5]:
            st.write(f"- **{feature} :** {selected_client[feature]}")

        # Style des décisions (Accordé/Refusé)
        def style_decision(value):
            if value == "Accordé":
                return "color: green; font-weight: bold;"
            elif value == "Refusé":
                return "color: red; font-weight: bold;"
            return ""

        # Affichage des résultats avec style
        st.write("**Résultats des prédictions :**")
        styled_table = (
            client_data[["SK_ID_CURR", "Probabilité de défaut", "Décision"]]
            .style.applymap(style_decision, subset=["Décision"])
        )
        st.write(styled_table.to_html(escape=False), unsafe_allow_html=True)

        # Affichage de l'importance globale des caractéristiques
        st.write("**Importance globale des caractéristiques :**")
        shap_values_top = shap_values_to_use[:, top_indices]
        normalized_data_top = normalized_data[:, top_indices]
        fig_summary = plt.figure()
        shap.summary_plot(shap_values_top, normalized_data_top, feature_names=top_features, show=False)
        st.pyplot(fig_summary)

        # Explications locales avec SHAP pour un client spécifique
        st.write("**Importance locale pour un client spécifique :**")
        shap_values_client = shap_values_to_use[client_index]
        data_client = normalized_data[client_index]
        force_plot_html = shap.force_plot(
            explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
            shap_values_client,
            data_client,
            feature_names=required_columns,
            matplotlib=False
        )
        components.html(f"<head>{shap.getjs()}</head><body>{force_plot_html.html()}</body>", height=500)

    elif menu == "Positionnement du client":
        st.title("Positionnement du client")

        # Sélection d'une variable
        variable = st.selectbox("Sélectionnez une variable :", required_columns)

        # Sélection d'un client
        client_index = st.selectbox(
            "Sélectionnez un client :", 
            range(len(client_data)), 
            format_func=lambda x: f"Client {client_data['SK_ID_CURR'].iloc[x]}"
        )

        # Extraire la valeur de la variable pour le client
        client_value = client_data.loc[client_index, variable]

        # Création du graphique de distribution
        fig, ax = plt.subplots()
        ax.hist(client_data[variable], bins=30, alpha=0.7, label="Tous les clients")
        ax.axvline(client_value, color='red', linestyle='dashed', linewidth=2, label=f"Client {client_data['SK_ID_CURR'].iloc[client_index]}")
        ax.set_title(f"Positionnement du client pour la variable {variable}")
        ax.set_xlabel(variable)
        ax.set_ylabel("Fréquence")
        ax.legend()

        # Affichage du graphique
        st.pyplot(fig)

    elif menu == "Modification des données du client":
        # (Code existant pour la modification des données du client)

        st.title("Modification des données du client")
        client_index = st.selectbox(
            "Sélectionnez un client :", 
            range(len(client_data)), 
            format_func=lambda x: f"Client {client_data['SK_ID_CURR'].iloc[x]}"
        )
        selected_client_data = client_data.iloc[client_index].copy()
        st.write(f"**Client sélectionné : {client_data['SK_ID_CURR'].iloc[client_index]}**")

        # Modification des 5 principales caractéristiques
        for feature in top_features[:5]:
            new_value = st.number_input(f"Modifier {feature} :", value=float(selected_client_data[feature]))
            selected_client_data[feature] = new_value

        if st.button("Recalculer avec les modifications"):
            modified_data_df = pd.DataFrame([selected_client_data])
            modified_imputed = imputer.transform(modified_data_df[required_columns])
            modified_normalized = scaler.transform(modified_imputed)
            new_predictions_proba = model.predict_proba(modified_normalized)[:, 1]
            new_decision = "Accordé" if new_predictions_proba[0] < 0.5 else "Refusé"
            st.write(f"**Résultats après modification :**")
            st.write(f"Probabilité de défaut : {new_predictions_proba[0]:.2f}")
            st.write(f"Nouvelle décision : {new_decision}")

    elif menu == "Ajout d'un nouveau client":
        # (Code existant pour l'ajout d'un nouveau client)

        st.title("Ajout d'un nouveau client")
        median_values = client_data[required_columns].median()
        new_client_data = median_values.copy()

        for feature in top_features[:5]:
            new_value = st.number_input(f"Valeur pour {feature} :", value=float(median_values[feature]))
            new_client_data[feature] = new_value

        if st.button("Calculer le score pour ce nouveau client"):
            new_client_df = pd.DataFrame([new_client_data])
            new_client_imputed = imputer.transform(new_client_df[required_columns])
            new_client_normalized = scaler.transform(new_client_imputed)
            new_client_proba = model.predict_proba(new_client_normalized)[:, 1]
            new_client_decision = "Accordé" if new_client_proba[0] < 0.5 else "Refusé"
            st.write(f"**Résultats pour le nouveau client :**")
            st.write(f"Probabilité de défaut : {new_client_proba[0]:.2f}")
            st.write(f"Décision : {new_client_decision}")

    elif menu == "Graphique":
        # (Code existant pour l'onglet Graphique)

        st.title("Graphique Bi-varié")
        x_axis = st.selectbox("Sélectionnez l'axe X :", required_columns, index=0)
        y_axis = st.selectbox("Sélectionnez l'axe Y :", required_columns, index=1)
        client_index = st.selectbox(
            "Sélectionnez un client :", 
            range(len(client_data)), 
            format_func=lambda x: f"Client {client_data['SK_ID_CURR'].iloc[x]}"
        )
        client_x = client_data.loc[client_index, x_axis]
        client_y = client_data.loc[client_index, y_axis]

        fig, ax = plt.subplots()
        ax.scatter(client_data[x_axis], client_data[y_axis], alpha=0.6, label="Clients")
        ax.scatter(client_x, client_y, color="red", s=100, label=f"Client {client_data['SK_ID_CURR'].iloc[client_index]}")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title("Graphique Bi-varié")
        ax.legend()
        st.pyplot(fig)

else:
    st.warning("Veuillez téléverser un fichier Excel pour commencer.")
