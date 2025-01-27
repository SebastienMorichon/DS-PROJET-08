# Utilisation d'une image Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier de dépendances
COPY requirements.txt requirements.txt

# Installer les outils nécessaires et mettre à jour pip
RUN apt-get update && apt-get install -y \
    g++ \
    build-essential \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers dans le conteneur
COPY . .

# Exposer le port 8501
EXPOSE 8081

# Utiliser la variable PORT dans la commande CMD
CMD ["sh", "-c", "streamlit run Streamlit_PROJET_08.py --server.port=${PORT:-8081} --server.address=0.0.0.0"]




