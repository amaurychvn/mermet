import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# 1. CHARGEMENT ET FILTRAGE DES DONNÉES
# ==========================================
url_github = "https://raw.githubusercontent.com/amaurychvn/petrole-bot/refs/heads/main/prix_carburants_quotidien.csv?token=GHSAT0AAAAAADVEPTSYR4QFCPKENZTLLC4M2M546NA"

# Lecture du fichier
df_carburants = pd.read_csv(url_github)

# On filtre pour ne garder que les lignes concernant les camions (Gazole)
df_gazole = df_carburants[df_carburants['Carburant'] == 'Gazole'].copy()

# On convertit la colonne 'Date' au bon format et on la met en index
df_gazole['Date'] = pd.to_datetime(df_gazole['Date'])
df_gazole.set_index('Date', inplace=True)

# On ne garde plus que la colonne 'Prix' qui nous intéresse
df_gazole = df_gazole[['Prix']]

# ==========================================
# 2. RÉCUPÉRATION DU BRENT ET EUR/USD
# ==========================================
tickers = ["BZ=F", "EURUSD=X"]

# On cale les dates sur celles de votre fichier (ex: 2024-01-01 à aujourd'hui)
start_date = df_gazole.index.min().strftime('%Y-%m-%d')
end_date = df_gazole.index.max().strftime('%Y-%m-%d')

print(f"Téléchargement des données financières du {start_date} au {end_date}...")
# On télécharge les prix de clôture ('Close')
donnees_finance = yf.download(tickers, start=start_date, end=end_date)['Close']
donnees_finance.rename(columns={'BZ=F': 'Brent', 'EURUSD=X': 'EUR_USD'}, inplace=True)

# ==========================================
# 3. FUSION DES DONNÉES
# ==========================================
# On fusionne le prix du Gazole avec le Brent et l'EUR/USD selon la date
df_merged = df_gazole.join(donnees_finance, how='inner').dropna()

print(f"Nombre de jours exploitables après fusion : {len(df_merged)}")

# ==========================================
# 4. MODÈLE DE PRÉDICTION (Machine Learning)
# ==========================================
# Les variables qui expliquent le prix (Features)
X = df_merged[['Brent', 'EUR_USD']]
# Ce que l'on cherche à prédire (Target)
y = df_merged['Prix']

# Séparation : 80% pour entraîner la machine, 20% pour vérifier si elle a compris
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement
modele = LinearRegression()
modele.fit(X_train, y_train)

# ==========================================
# 5. ÉVALUATION ET VISUALISATION
# ==========================================
predictions = modele.predict(X_test)

print(f"Erreur quadratique moyenne (MSE) : {mean_squared_error(y_test, predictions):.4f}")
print(f"Score R
