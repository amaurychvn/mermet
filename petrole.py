import requests
import zipfile
import io
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime
import os

# Configuration
URL_INSTANTANE = "https://donnees.roulez-eco.fr/opendata/instantane"
# Pour les archives, l'URL change chaque année, on peut les lister
URLS_ARCHIVES = [
    "https://donnees.roulez-eco.fr/opendata/annee/2024",
    "https://donnees.roulez-eco.fr/opendata/annee/2025",
    # Ajouter 2025 quand disponible en archive, sinon il faut utiliser l'instantané quotidiennement
]

# Configuration
# On récupère le dossier où se trouve le script actuel
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "prix_carburants_quotidien.csv")

def parse_xml_data(xml_content, is_instantane=False):
    """
    Lit le contenu XML et extrait : Date, Type de carburant, Prix.
    Version corrigée pour le format de date avec 'T' (ex: 2024-01-02T00:37:00).
    """
    root = ET.fromstring(xml_content)
    data = []

    for pdv in root.findall('pdv'):
        for prix in pdv.findall('prix'):
            valeur = prix.get('valeur')
            nom = prix.get('nom')
            maj = prix.get('maj')

            if valeur and nom and maj:
                try:
                    val_float = float(valeur)

                    # CORRECTION ICI : Ajout du 'T' dans le format
                    # On gère le cas avec 'T' (XML récent) et le cas avec espace (vieux XML potentiels)
                    if 'T' in maj:
                        date_fmt = "%Y-%m-%dT%H:%M:%S"
                    else:
                        date_fmt = "%Y-%m-%d %H:%M:%S"

                    date_obj = datetime.strptime(maj, date_fmt).date()

                    if is_instantane:
                        date_ref = datetime.now().date()
                    else:
                        date_ref = date_obj

                    data.append({
                        'Date': date_ref,
                        'Carburant': nom,
                        'Prix': val_float
                    })
                except ValueError:
                    continue
    return data


def get_data_from_url(url, is_instantane=False):
    """Télécharge, dézippe et parse les données avec protection anti-bot."""
    print(f"Téléchargement de {url}...")

    # On se fait passer pour un navigateur pour ne pas être bloqué
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        r = requests.get(url, headers=headers, timeout=30)

        # Vérification 1 : Code HTTP
        if r.status_code != 200:
            print(f" -> Erreur HTTP : {r.status_code}")
            return []

        # Vérification 2 : Est-ce bien un ZIP ?
        content_type = r.headers.get('Content-Type', '')
        if 'zip' not in content_type and 'application/octet-stream' not in content_type:
            print(f" -> Attention : Le fichier reçu n'est pas un ZIP (Type: {content_type}).")
            # Souvent, si on reçoit du 'text/html', c'est une page d'erreur du site
            return []

        # Traitement du ZIP
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            filename = z.namelist()[0]
            print(f" -> Fichier trouvé dans l'archive : {filename}")
            with z.open(filename) as f:
                xml_content = f.read()
                data = parse_xml_data(xml_content, is_instantane)
                print(f" -> {len(data)} lignes extraites.")
                return data

    except Exception as e:
        print(f" -> Exception critique : {e}")
        return []


def aggregate_daily(df):
    """Groupe par Date et Carburant et calcule la moyenne."""
    # On ne garde que les colonnes utiles
    df = df[['Date', 'Carburant', 'Prix']]

    # Calcul de la moyenne
    df_agg = df.groupby(['Date', 'Carburant'], as_index=False)['Prix'].mean()

    # Arrondi pour faire propre (3 décimales, standard essence)
    df_agg['Prix'] = df_agg['Prix'].round(3)
    return df_agg


def initialisation():
    print("--- DÉBUT INITIALISATION ---")
    all_data = []

    # 1. Récupération des archives
    for url in URLS_ARCHIVES:
        data = get_data_from_url(url, is_instantane=False)
        all_data.extend(data)

    # Création DataFrame
    df = pd.DataFrame(all_data)

    if not df.empty:
        # Agrégation
        df_final = aggregate_daily(df)

        # Sauvegarde
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"Succès ! Fichier '{OUTPUT_FILE}' créé avec {len(df_final)} lignes.")
    else:
        print("Aucune donnée récupérée.")


def mise_a_jour():
    print("--- DÉBUT MISE À JOUR QUOTIDIENNE ---")
    # 1. Lire le fichier existant
    if not os.path.exists(OUTPUT_FILE):
        print("Fichier historique introuvable. Lancement de l'initialisation...")
        initialisation()
        return

    df_hist = pd.read_csv(OUTPUT_FILE)

    # 2. Récupérer le flux instantané (les prix d'aujourd'hui)
    new_data = get_data_from_url(URL_INSTANTANE, is_instantane=True)
    df_new = pd.DataFrame(new_data)

    if not df_new.empty:
        # Agrégation pour avoir la moyenne d'aujourd'hui
        # Note : is_instantane=True a déjà forcé la date à "Aujourd'hui"
        df_new_agg = aggregate_daily(df_new)

        # 3. Fusionner (Concaténer)
        # On supprime d'abord si la date d'aujourd'hui existe déjà (pour éviter les doublons si tu lances le script 2 fois)
        today_str = str(datetime.now().date())
        df_hist = df_hist[df_hist['Date'] != today_str]

        df_final = pd.concat([df_hist, df_new_agg], ignore_index=True)

        # Sauvegarde
        df_final.to_csv(OUTPUT_FILE, index=False)
        print(f"Mise à jour terminée. Données du {today_str} ajoutées.")
    else:
        print("Impossible de récupérer le flux instantané.")


# --- Point d'entrée du script ---
if __name__ == "__main__":
    # Décommente la ligne dont tu as besoin :

    # Etape 1 : À faire tourner une seule fois au début
    # initialisation()

    # Etape 2 : À programmer tous les jours
    mise_a_jour()
