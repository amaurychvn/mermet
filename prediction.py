import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ==========================================
# 1. CHARGEMENT ET FILTRAGE DES DONNÉES
# ==========================================

# L'URL "Raw" pure de votre fichier (SANS le ?token=... à la fin)
url_github = "https://raw.githubusercontent.com/amaurychvn/petrole-bot/refs/heads/main/prix_carburants_quotidien.csv"

# Collez votre vrai token généré à l'étape 6 ici
mon_token = "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"

# Autorisation et lecture
headers = {'Authorization': f'token {mon_token}'}
reponse = requests.get(url_github, headers=headers)

# Création du tableau de données complet
df_carburants = pd.read_csv(StringIO(reponse.text))

# Filtrage pour ne garder que le Gazole
df_gazole = df_carburants[df_carburants['Carburant'] == 'Gazole'].copy()
df_gazole['Date'] = pd.to_datetime(df_gazole['Date'])
df_gazole.set_index('Date', inplace=True)
df_gazole = df_gazole[['Prix']]

print("Fichier chargé et filtré avec succès !")

# ==========================================
# 2. RÉCUPÉRATION DU BRENT ET EUR/USD
# ==========================================
tickers = ["BZ=F", "EURUSD=X"]
start_date = df_gazole.index.min().strftime('%Y-%m-%d')
end_date = df_gazole.index.max().strftime('%Y-%m-%d')

donnees_finance = yf.download(tickers, start=start_date, end=end_date)['Close']
donnees_finance.rename(columns={'BZ=F': 'Brent', 'EURUSD=X': 'EUR_USD'}, inplace=True)

# GESTION DES WEEK-ENDS : on ré-indexe pour avoir tous les jours de l'année
# et on remplit les jours sans bourse (week-ends) avec le prix du vendredi (ffill)
donnees_finance = donnees_finance.asfreq('D').ffill()

# Calculer le prix du baril en Euros (car nous payons en Euros à la pompe)
donnees_finance['Brent_Euros'] = donnees_finance['Brent'] / donnees_finance['EUR_USD']

# AJOUT DES DÉCALAGES TEMPELS (Lags) 
# Le prix à la pompe dépend du prix du brut d'il y a 1, 2 ou 3 semaines.
donnees_finance['Brent_Eur_J_moins_7'] = donnees_finance['Brent_Euros'].shift(7)
donnees_finance['Brent_Eur_J_moins_14'] = donnees_finance['Brent_Euros'].shift(14)
donnees_finance['Brent_Eur_J_moins_21'] = donnees_finance['Brent_Euros'].shift(21)

# ==========================================
# 3. FUSION DES DONNÉES
# ==========================================
# On fusionne. Les premières lignes auront des NaN à cause du "shift", on les supprime.
df_merged = df_gazole.join(donnees_finance, how='inner').dropna()

# ==========================================
# 4. MODÈLE DE PRÉDICTION
# ==========================================
# On utilise maintenant les prix passés pour prédire le prix actuel
X = df_merged[['Brent_Euros', 'Brent_Eur_J_moins_7', 'Brent_Eur_J_moins_14', 'Brent_Eur_J_moins_21']]
y = df_merged['Prix']

# SÉPARATION CHRONOLOGIQUE (Important : shuffle=False)
# On s'entraîne sur le passé, on teste sur l'avenir (les 20 derniers % des dates)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

modele = LinearRegression()
modele.fit(X_train, y_train)

# ==========================================
# 5. ÉVALUATION
# ==========================================
predictions = modele.predict(X_test)

print(f"Erreur quadratique moyenne (MSE) : {mean_squared_error(y_test, predictions):.4f}")
print(f"Score R2 : {r2_score(y_test, predictions):.4f}")

# Petit bonus : regarder le poids des variables pour voir ce qui influence le plus
for feature, coef in zip(X.columns, modele.coef_):
    print(f"Impact de {feature} : {coef:.5f}")
