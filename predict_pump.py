# -*- coding: utf-8 -*-
"""
predict_pump.py
================
Système de prédiction du prix moyen du Gazole en France à 14 jours.

Hébergement GitHub Pages :
  Settings → Pages → Source : « Deploy from a branch »
  Branch : main   |   Folder : / (root)
  URL résultante : https://amaurychvn.github.io/mermet/

Exécution locale :
  $ pip install -r requirements_predict.txt
  $ python predict_pump.py

Le script lit les CSVs depuis le cwd (cohérent avec actions/checkout).
Chaque appel est idempotent : mêmes entrées => mêmes sorties.
"""

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
np.random.seed(42)

# =============================================================================
#  CONSTANTES
# =============================================================================
RANDOM_STATE = 42
HORIZON = 14                     # nb de jours à prédire
BACKTEST_SELECTION_DAYS = 30     # backtest pour sélectionner le modèle
BACKTEST_METRICS_DAYS = 14       # backtest pour calculer les métriques

# Fenêtres de lags (cf. spec)
LAGS_GAZOLE = list(range(1, 8))     # J-1 à J-7
LAGS_BRENT = list(range(7, 22))     # J-7 à J-21
LAGS_EURUSD = list(range(7, 15))    # J-7 à J-14
MAX_LAG = max(max(LAGS_GAZOLE), max(LAGS_BRENT), max(LAGS_EURUSD))

# Fichiers
PATH_PRICES = "prix_carburants_quotidien.csv"
PATH_BRENT = "brent_usd.csv"
PATH_EURUSD = "eurusd.csv"
PATH_HTML = "index.html"
PATH_PREDICTIONS_CSV = "predictions.csv"

# URLs externes
REPO_URL = "https://github.com/amaurychvn/mermet"
WORKFLOW_URL = "https://github.com/amaurychvn/mermet/actions/workflows/predict.yml"

TZ_PARIS = ZoneInfo("Europe/Paris")

# =============================================================================
#  LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict_pump")


# =============================================================================
#  CHARGEMENT & PRÉPARATION
# =============================================================================
def load_data():
    """Lecture robuste des trois CSVs depuis le cwd."""
    log.info("Chargement des données…")
    try:
        df_prices = pd.read_csv(PATH_PRICES, parse_dates=["Date"])
        df_brent = pd.read_csv(PATH_BRENT, parse_dates=["Date"])
        df_eurusd = pd.read_csv(PATH_EURUSD, parse_dates=["Date"])
    except Exception as e:
        log.error(f"Échec lecture des CSVs : {e}")
        sys.exit(1)
    log.info(f"  prix_carburants : {len(df_prices)} lignes")
    log.info(f"  brent_usd       : {len(df_brent)} lignes")
    log.info(f"  eurusd          : {len(df_eurusd)} lignes")
    return df_prices, df_brent, df_eurusd


def prepare_data(df_prices, df_brent, df_eurusd):
    """
    Aligne tout sur le calendrier complet du Gazole avec forward-fill
    pour les exogènes (week-ends, fériés). Calcule Brent en EUR puis
    log-prix et log-returns du Gazole.
    """
    log.info("Préparation et alignement des séries…")

    # Filtrer Gazole uniquement (la typo GLPc/GPLc n'affecte pas le Gazole).
    df_g = (
        df_prices.loc[df_prices["Carburant"] == "Gazole", ["Date", "Prix"]]
        .rename(columns={"Prix": "Gazole"})
        .sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
    )

    full_idx = df_g.index  # calendrier complet du Gazole (quotidien sans trous)

    df_b = (
        df_brent.set_index("Date").sort_index()["Prix_USD"]
        .reindex(full_idx, method="ffill")
    )
    df_e = (
        df_eurusd.set_index("Date").sort_index()["Taux"]
        .reindex(full_idx, method="ffill")
    )

    df = df_g.copy()
    df["Brent_USD"] = df_b
    df["EUR_USD"] = df_e
    df["Brent_EUR"] = df["Brent_USD"] / df["EUR_USD"]

    # Suppression d'éventuelles lignes initiales sans exogène (Brent commence
    # généralement le 02/01/2024, donc le 01/01/2024 peut être incomplet).
    n_before = len(df)
    df = df.dropna()
    if (n_before - len(df)) > 0:
        log.info(f"  {n_before - len(df)} ligne(s) retirée(s) (exogènes manquants en début)")

    df["LogPrice"] = np.log(df["Gazole"])
    df["LogReturn"] = df["LogPrice"].diff()

    log.info(
        f"  série finale : {len(df)} jours, "
        f"du {df.index[0].date()} au {df.index[-1].date()}"
    )
    return df


# =============================================================================
#  PRÉVISION DES EXOGÈNES (ARIMA univarié)
# =============================================================================
def forecast_exog_arima(series, n_steps):
    """
    Prévision ponctuelle d'une série univariée (Brent_EUR ou EUR/USD).
    On utilise un ARIMA(1,1,1) simple. C'est volontairement minimal —
    les prévisions exogènes constituent une source d'incertitude
    additionnelle non propagée formellement dans les IC du modèle Gazole
    (limite documentée du système).
    """
    try:
        result = ARIMA(series.values, order=(1, 1, 1)).fit()
        return np.asarray(result.get_forecast(steps=n_steps).predicted_mean)
    except Exception as e:
        log.warning(f"  ARIMA exogène a échoué ({e}) — fallback : valeur constante.")
        return np.repeat(series.values[-1], n_steps)


# =============================================================================
#  MODÈLE 1 : SARIMAX
#   - ordres (1,1,1) sur le log-prix (la différenciation interne donne
#     un modèle équivalent à une ARMA sur les log-returns)
#   - exogènes : Brent_EUR(t-1) et EUR/USD(t-1)
#   - IC 95 % natifs et naturellement croissants avec l'horizon (variance
#     d'un cumul d'innovations).
# =============================================================================
def fit_sarimax(df):
    """SARIMAX(1,1,1) sur le log-prix, exog = (Brent_EUR_{t-1}, EUR/USD_{t-1})."""
    y = df["LogPrice"].iloc[1:].values
    exog = np.column_stack(
        [df["Brent_EUR"].iloc[:-1].values, df["EUR_USD"].iloc[:-1].values]
    )
    model = SARIMAX(
        y,
        exog=exog,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def predict_sarimax(result, last_brent_eur, last_eurusd,
                    future_brent_eur, future_eurusd, n_steps=HORIZON):
    """
    Prévision multi-step SARIMAX avec IC 95 %.

    Stratégie multi-step : DIRECTE — get_forecast(steps=n) traite tout le
    chemin en une seule passe via la représentation espace-état, ce qui
    propage correctement la variance des innovations à chaque pas. Les IC
    s'élargissent donc naturellement.

    Pour l'exog à T+k, on a besoin de (Brent_EUR_{T+k-1}, EUR/USD_{T+k-1}).
    Pour T+1 c'est la dernière valeur observée ; pour T+2..T+14 on utilise
    les prévisions ARIMA des exogènes.
    """
    future_exog_brent = np.concatenate([[last_brent_eur], future_brent_eur[: n_steps - 1]])
    future_exog_eurusd = np.concatenate([[last_eurusd], future_eurusd[: n_steps - 1]])
    future_exog = np.column_stack([future_exog_brent, future_exog_eurusd])

    fc = result.get_forecast(steps=n_steps, exog=future_exog)
    mean_log = np.asarray(fc.predicted_mean)
    ci = fc.conf_int(alpha=0.05)
    ci = ci.values if hasattr(ci, "values") else ci
    return mean_log, ci[:, 0], ci[:, 1]


def predict_sarimax_full(df, future_brent_eur, future_eurusd, n_steps=HORIZON):
    """Wrapper : ajuste SARIMAX puis renvoie un DataFrame indexé par date."""
    result = fit_sarimax(df)
    last_b = df["Brent_EUR"].iloc[-1]
    last_e = df["EUR_USD"].iloc[-1]
    mean_log, low_log, up_log = predict_sarimax(
        result, last_b, last_e, future_brent_eur, future_eurusd, n_steps
    )
    dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=n_steps, freq="D")
    return pd.DataFrame(
        {
            "Prix_predit": np.exp(mean_log),
            "IC_bas": np.exp(low_log),
            "IC_haut": np.exp(up_log),
        },
        index=pd.Index(dates, name="Date"),
    )


# =============================================================================
#  MODÈLE 2 : XGBoost quantile
#   - Trois boosters : q=0.025, q=0.5, q=0.975 (objectif reg:quantileerror)
#   - Cible : log-return 1-step
#   - IC k-step : on suppose les erreurs i.i.d. dans le temps et on cumule
#     la variance pas à pas (équivalent à élargir en √k pour un spread
#     constant). Stratégie multi-step : RÉCURSIVE.
# =============================================================================
def build_features_at_pos(df, pos):
    """Features pour prédire le log-return à la position 'pos'."""
    feats = {}
    for lag in LAGS_GAZOLE:
        feats[f"gazole_lag_{lag}"] = df["Gazole"].iloc[pos - lag] if pos - lag >= 0 else np.nan
    for lag in LAGS_BRENT:
        feats[f"brent_eur_lag_{lag}"] = df["Brent_EUR"].iloc[pos - lag] if pos - lag >= 0 else np.nan
    for lag in LAGS_EURUSD:
        feats[f"eurusd_lag_{lag}"] = df["EUR_USD"].iloc[pos - lag] if pos - lag >= 0 else np.nan
    target_date = df.index[pos]
    feats["dow"] = int(target_date.dayofweek)
    feats["month"] = int(target_date.month)
    return feats


def make_training_set(df, end_pos):
    """Construit X / y depuis la position MAX_LAG jusqu'à end_pos (exclu)."""
    rows = []
    for pos in range(MAX_LAG, end_pos):
        feats = build_features_at_pos(df, pos)
        target = df["LogReturn"].iloc[pos]
        if pd.notna(target) and not any(pd.isna(v) for v in feats.values()):
            feats["_target"] = target
            rows.append(feats)
    if not rows:
        return None, None
    df_t = pd.DataFrame(rows)
    return df_t.drop(columns=["_target"]), df_t["_target"]


def fit_xgb_quantile(X, y):
    """Trois boosters quantiles (0.025, 0.5, 0.975)."""
    models = {}
    for label, alpha in [("q025", 0.025), ("q50", 0.5), ("q975", 0.975)]:
        m = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=alpha,
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0,
        )
        m.fit(X, y)
        models[label] = m
    return models


def predict_xgb_recursive(df, models, feature_cols, future_brent_eur,
                          future_eurusd, n_steps=HORIZON):
    """
    Prédiction récursive XGBoost sur n_steps.

    À chaque pas k :
      - les lags de Gazole qui pointent vers le futur sont remplacés par
        les prix médians prédits aux pas antérieurs ;
      - les lags d'exogènes futurs sont fournis par les prévisions ARIMA.
    Les IC s'accumulent : à l'horizon k, la demi-largeur ≈ √(somme des
    spreads quantiles au carré sur les k pas), ce qui produit un
    élargissement monotone (équivalent √k pour un spread constant).
    """
    last_pos = len(df) - 1
    last_log_price = df["LogPrice"].iloc[-1]

    cum_median = 0.0
    cum_var_low = 0.0
    cum_var_high = 0.0

    pred_prices_path = []  # prix médians prédits aux pas précédents
    out = []

    for k in range(1, n_steps + 1):
        future_date = df.index[-1] + pd.Timedelta(days=k)
        feats = {}

        for lag in LAGS_GAZOLE:
            target_pos = last_pos + k - lag
            if target_pos <= last_pos:
                feats[f"gazole_lag_{lag}"] = df["Gazole"].iloc[target_pos]
            else:
                pred_step = target_pos - last_pos  # 1..k-1
                feats[f"gazole_lag_{lag}"] = pred_prices_path[pred_step - 1]

        for lag in LAGS_BRENT:
            target_pos = last_pos + k - lag
            if target_pos <= last_pos:
                feats[f"brent_eur_lag_{lag}"] = df["Brent_EUR"].iloc[target_pos]
            else:
                pred_step = target_pos - last_pos
                feats[f"brent_eur_lag_{lag}"] = float(future_brent_eur[pred_step - 1])

        for lag in LAGS_EURUSD:
            target_pos = last_pos + k - lag
            if target_pos <= last_pos:
                feats[f"eurusd_lag_{lag}"] = df["EUR_USD"].iloc[target_pos]
            else:
                pred_step = target_pos - last_pos
                feats[f"eurusd_lag_{lag}"] = float(future_eurusd[pred_step - 1])

        feats["dow"] = int(future_date.dayofweek)
        feats["month"] = int(future_date.month)

        X_pred = pd.DataFrame([feats])[feature_cols]
        med = float(models["q50"].predict(X_pred)[0])
        lo = float(models["q025"].predict(X_pred)[0])
        hi = float(models["q975"].predict(X_pred)[0])

        # On force des spreads positifs (les modèles quantiles peuvent
        # exceptionnellement croiser sur de faibles écarts).
        sigma_low_step = max(med - lo, 1e-6)
        sigma_high_step = max(hi - med, 1e-6)

        cum_median += med
        cum_var_low += sigma_low_step ** 2
        cum_var_high += sigma_high_step ** 2

        log_med = last_log_price + cum_median
        log_low = log_med - np.sqrt(cum_var_low)
        log_high = log_med + np.sqrt(cum_var_high)

        price_med = float(np.exp(log_med))
        pred_prices_path.append(price_med)

        out.append({
            "Date": future_date,
            "Prix_predit": price_med,
            "IC_bas": float(np.exp(log_low)),
            "IC_haut": float(np.exp(log_high)),
        })

    return pd.DataFrame(out).set_index("Date")


# =============================================================================
#  BACKTESTS
# =============================================================================
def backtest_one_step(df, model_name, n_days):
    """
    Backtest 1-step glissant. Pour chacun des n_days derniers jours,
    on ré-entraîne sur df[:t] et on prédit le jour t à l'horizon J+1
    avec son IC 95 %.

    Retourne (preds, actuals, lowers, uppers, dates) — listes alignées
    et filtrées des éventuelles itérations en erreur.
    """
    preds, lowers, uppers, actuals, dates = [], [], [], [], []
    for i in range(n_days, 0, -1):
        cut_pos = len(df) - i  # df[:cut_pos] = entraînement ; target à cut_pos
        df_train = df.iloc[:cut_pos]
        if len(df_train) < MAX_LAG + 30:
            continue
        try:
            if model_name == "SARIMAX":
                res = fit_sarimax(df_train)
                last_b = df_train["Brent_EUR"].iloc[-1]
                last_e = df_train["EUR_USD"].iloc[-1]
                fc = res.get_forecast(
                    steps=1, exog=np.array([[last_b, last_e]])
                )
                mean_log = float(np.asarray(fc.predicted_mean)[0])
                ci = fc.conf_int(alpha=0.05)
                ci = ci.values if hasattr(ci, "values") else ci
                p = float(np.exp(mean_log))
                pl = float(np.exp(ci[0, 0]))
                ph = float(np.exp(ci[0, 1]))
            else:  # XGBoost
                X, y = make_training_set(df_train, len(df_train))
                if X is None or len(X) < 30:
                    continue
                feat_cols = X.columns.tolist()
                models = fit_xgb_quantile(X, y)
                feats = build_features_at_pos(df, cut_pos)
                if any(pd.isna(v) for v in feats.values()):
                    continue
                X_pred = pd.DataFrame([feats])[feat_cols]
                med = float(models["q50"].predict(X_pred)[0])
                lo = float(models["q025"].predict(X_pred)[0])
                hi = float(models["q975"].predict(X_pred)[0])
                last_log = df["LogPrice"].iloc[cut_pos - 1]
                p = float(np.exp(last_log + med))
                pl = float(np.exp(last_log + lo))
                ph = float(np.exp(last_log + hi))

            preds.append(p)
            lowers.append(pl)
            uppers.append(ph)
            actuals.append(float(df["Gazole"].iloc[cut_pos]))
            dates.append(df.index[cut_pos])
        except Exception as e:
            log.warning(f"  Backtest {model_name} (j -{i}) : {e}")
            continue
    return (np.array(preds), np.array(actuals),
            np.array(lowers), np.array(uppers), dates)


def select_model(df):
    """Sélectionne le modèle qui minimise la MAPE J+1 sur 30 jours glissants."""
    log.info(f"Sélection de modèle — backtest sur {BACKTEST_SELECTION_DAYS} jours…")
    results = {}
    for name in ["SARIMAX", "XGBoost"]:
        log.info(f"  -> {name}")
        p, a, _, _, _ = backtest_one_step(df, name, BACKTEST_SELECTION_DAYS)
        if len(p) == 0:
            mape = np.inf
        else:
            mape = float(np.mean(np.abs((a - p) / a)) * 100)
        results[name] = mape
        log.info(f"     MAPE J+1 = {mape:.3f} %")
    best = min(results, key=results.get)
    log.info(f"Modèle sélectionné : {best}")
    return best, results


def compute_metrics(df, model_name):
    """
    Backtest 14j et calcul des trois métriques :
      - MAPE
      - Couverture IC 95 %
      - Précision directionnelle (J+1 vs prix observé J-1)
    """
    log.info(f"Métriques — backtest sur {BACKTEST_METRICS_DAYS} jours ({model_name})…")
    p, a, lo, hi, dates_test = backtest_one_step(df, model_name, BACKTEST_METRICS_DAYS)
    if len(p) == 0:
        return {
            "mape": float("nan"), "coverage": float("nan"),
            "directional": float("nan"),
            "errors_pct": [], "errors_dates": [],
        }
    mape = float(np.mean(np.abs((a - p) / a)) * 100)
    coverage = float(np.mean((a >= lo) & (a <= hi)) * 100)
    last_obs = np.array([df["Gazole"].iloc[df.index.get_loc(d) - 1] for d in dates_test])
    pred_dir = np.sign(p - last_obs)
    real_dir = np.sign(a - last_obs)
    directional = float(np.mean(pred_dir == real_dir) * 100)
    errors_pct = ((a - p) / a * 100).tolist()
    log.info(
        f"  MAPE = {mape:.3f} % | Couverture IC95 = {coverage:.1f} % | "
        f"Direction = {directional:.1f} %"
    )
    return {
        "mape": mape, "coverage": coverage, "directional": directional,
        "errors_pct": errors_pct, "errors_dates": dates_test,
    }


# =============================================================================
#  CONTEXTE MARCHÉS
# =============================================================================
def market_context(df):
    """Variations à 7j et 30j de Brent USD, Brent EUR, EUR/USD."""
    def var_pct(series, n):
        if len(series) <= n:
            return float("nan")
        return float((series.iloc[-1] / series.iloc[-1 - n] - 1) * 100)

    return {
        "brent_usd": {
            "current": float(df["Brent_USD"].iloc[-1]),
            "var_7d": var_pct(df["Brent_USD"], 7),
            "var_30d": var_pct(df["Brent_USD"], 30),
        },
        "brent_eur": {
            "current": float(df["Brent_EUR"].iloc[-1]),
            "var_7d": var_pct(df["Brent_EUR"], 7),
            "var_30d": var_pct(df["Brent_EUR"], 30),
        },
        "eur_usd": {
            "current": float(df["EUR_USD"].iloc[-1]),
            "var_7d": var_pct(df["EUR_USD"], 7),
            "var_30d": var_pct(df["EUR_USD"], 30),
        },
    }


# =============================================================================
#  MISE EN FORME FRANÇAISE
# =============================================================================
def fr_num(value, decimals=3):
    """Format français : virgule décimale, espace fine pour milliers."""
    if value is None:
        return "—"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "—"
    s = f"{value:,.{decimals}f}"
    return s.replace(",", "\u00A0").replace(".", ",")


def fr_pct(value, decimals=2, signed=True):
    if value is None:
        return "—"
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "—"
    fmt = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return fmt.format(value).replace(".", ",") + "\u202F%"


def fr_date(d):
    return pd.Timestamp(d).strftime("%d/%m/%Y")


# =============================================================================
#  GRAPHIQUES PLOTLY
# =============================================================================
def build_main_chart(df, forecast_df):
    last_30 = df.iloc[-30:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_30.index, y=last_30["Gazole"],
        mode="lines", name="Réel",
        line=dict(color="#1f3b6f", width=2.5),
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.3f} €/L<extra></extra>",
    ))
    # Bande IC : on trace haut puis bas avec fill='tonexty'
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["IC_haut"],
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["IC_bas"],
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor="rgba(230,134,50,0.18)",
        name="IC 95 %",
        hovertemplate="%{x|%d/%m/%Y}<br>IC bas %{y:.3f} €/L<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["Prix_predit"],
        mode="lines", name="Prédit",
        line=dict(color="#e68632", width=2.5, dash="dot"),
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.3f} €/L<extra></extra>",
    ))
    # Trait vertical : aujourd'hui = dernière date observée
    fig.add_vline(
        x=df.index[-1], line_dash="dash",
        line_color="#888", line_width=1,
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=40),
        height=380,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="", gridcolor="#eee"),
        yaxis=dict(title="€/L", gridcolor="#eee"),
        legend=dict(orientation="h", y=-0.15, x=0),
        hovermode="x unified",
    )
    return fig


def build_errors_chart(metrics):
    if not metrics["errors_dates"]:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics["errors_dates"], y=metrics["errors_pct"],
        mode="lines+markers",
        line=dict(color="#1f3b6f", width=1.5),
        marker=dict(size=5, color="#1f3b6f"),
        hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f} %<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#aaa", line_width=1)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=20),
        height=200,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#eee"),
        yaxis=dict(title="% (réel - prédit)", gridcolor="#eee"),
        showlegend=False,
        hovermode="x unified",
    )
    return fig


# =============================================================================
#  GÉNÉRATION HTML
# =============================================================================
def justification_text(model_name, mape_results):
    others = {k: v for k, v in mape_results.items() if k != model_name}
    other_name = list(others.keys())[0]
    sel = mape_results[model_name]
    oth = others[other_name]
    if not np.isfinite(oth) or not np.isfinite(sel):
        return f"{model_name} retenu (l'autre modèle n'a pu être évalué)."
    return (
        f"MAPE J+1 sur 30 j : {fr_num(sel, 3)}\u202F% pour {model_name} "
        f"vs {fr_num(oth, 3)}\u202F% pour {other_name}."
    )


def generate_html(df, forecast_df, metrics, market, model_name, mape_results):
    log.info("Génération du HTML…")
    last_price = float(df["Gazole"].iloc[-1])
    last_date = df.index[-1]
    pred_14 = float(forecast_df["Prix_predit"].iloc[-1])
    pred_14_lo = float(forecast_df["IC_bas"].iloc[-1])
    pred_14_hi = float(forecast_df["IC_haut"].iloc[-1])
    var_cumul = (pred_14 / last_price - 1) * 100
    # Pour un transporteur, hausse = mauvaise nouvelle => rouge
    var_color = "#c0392b" if var_cumul > 0 else "#27ae60"

    main_fig = build_main_chart(df, forecast_df)
    errors_fig = build_errors_chart(metrics)
    main_json = pio.to_json(main_fig)
    errors_json = pio.to_json(errors_fig) if errors_fig is not None else "null"

    # Tableau prédictions
    table_rows = []
    for d, row in forecast_df.iterrows():
        var_vs_last = (row["Prix_predit"] / last_price - 1) * 100
        color_v = "#c0392b" if var_vs_last > 0 else "#27ae60"
        table_rows.append(
            "<tr>"
            f"<td>{fr_date(d)}</td>"
            f"<td class='num'>{fr_num(row['Prix_predit'], 3)} €/L</td>"
            f"<td class='num' style='color:{color_v};'>{fr_pct(var_vs_last, 2)}</td>"
            f"<td class='num'>{fr_num(row['IC_bas'], 3)}</td>"
            f"<td class='num'>{fr_num(row['IC_haut'], 3)}</td>"
            "</tr>"
        )
    table_html = "\n          ".join(table_rows)

    # Cartes marchés
    def market_card(title, current, var7, var30, unit):
        c7 = "#c0392b" if (var7 or 0) > 0 else "#27ae60"
        c30 = "#c0392b" if (var30 or 0) > 0 else "#27ae60"
        unit_html = f" {unit}" if unit else ""
        return (
            "<div class='market-card'>"
            f"<div class='market-title'>{title}</div>"
            f"<div class='market-value'>{fr_num(current, 2)}{unit_html}</div>"
            "<div class='market-deltas'>"
            f"<span style='color:{c7};'>7\u202Fj : {fr_pct(var7, 2)}</span>"
            f"<span style='color:{c30};'>30\u202Fj : {fr_pct(var30, 2)}</span>"
            "</div></div>"
        )

    market_html = (
        market_card("Brent", market["brent_usd"]["current"],
                    market["brent_usd"]["var_7d"], market["brent_usd"]["var_30d"], "$/baril")
        + market_card("Brent (€)", market["brent_eur"]["current"],
                      market["brent_eur"]["var_7d"], market["brent_eur"]["var_30d"], "€/baril")
        + market_card("EUR/USD", market["eur_usd"]["current"],
                      market["eur_usd"]["var_7d"], market["eur_usd"]["var_30d"], "")
    )

    now_paris = datetime.now(TZ_PARIS).strftime("%d/%m/%Y à %H:%M (heure de Paris)")

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Prédiction Prix Gazole – 14 jours</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #f7f8fa;
    --card: #ffffff;
    --border: #e3e6eb;
    --text: #1a1a1a;
    --muted: #666;
    --primary: #1f3b6f;
    --accent: #e68632;
    --green: #27ae60;
    --red: #c0392b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: var(--text);
    background: var(--bg);
    line-height: 1.45;
    -webkit-font-smoothing: antialiased;
  }}
  header {{
    padding: 24px 20px 16px;
    background: white;
    border-bottom: 1px solid var(--border);
  }}
  header h1 {{
    margin: 0 0 6px;
    font-size: 1.4rem;
    color: var(--primary);
    font-weight: 600;
  }}
  header .sub {{ color: var(--muted); font-size: 0.85rem; }}
  header .repo-link {{
    margin-left: 8px; color: var(--muted);
    text-decoration: none; font-size: 0.8rem;
  }}
  header .repo-link:hover {{ color: var(--primary); }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 16px; }}
  section {{ margin-bottom: 22px; }}
  h2 {{
    font-size: 1.05rem; color: var(--primary);
    margin: 0 0 10px; font-weight: 600;
  }}
  .cards {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  .metrics-cards {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }}
  .market-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }}
  @media (max-width: 700px) {{
    .cards, .market-grid {{ grid-template-columns: 1fr; }}
    .metrics-cards {{ grid-template-columns: 1fr 1fr; }}
  }}
  .card, .market-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }}
  .card .label, .market-title {{
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }}
  .card .value {{
    font-size: 1.5rem; font-weight: 600; margin-top: 6px;
  }}
  .card .sub {{
    font-size: 0.82rem; color: var(--muted); margin-top: 4px;
  }}
  .metrics-cards .card .value {{ color: var(--primary); }}
  .market-value {{ font-size: 1.3rem; font-weight: 600; margin: 6px 0; }}
  .market-deltas {{ display: flex; gap: 14px; font-size: 0.85rem; flex-wrap: wrap; }}
  .chart {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px;
  }}
  .table-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: auto;
  }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; }}
  th, td {{
    padding: 9px 12px; text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: #f0f2f6; color: var(--primary);
    font-weight: 600; font-size: 0.82rem;
    text-transform: uppercase; letter-spacing: 0.03em;
  }}
  tbody tr:nth-child(even) {{ background: #fafbfc; }}
  td.num, th.num {{
    font-variant-numeric: tabular-nums; text-align: right;
  }}
  .recompute {{
    display: inline-block;
    background: var(--primary); color: white;
    text-decoration: none;
    padding: 11px 18px; border-radius: 8px;
    font-weight: 500; font-size: 0.92rem;
  }}
  .recompute:hover {{ background: #14284f; }}
  .recompute .hint {{
    display: block; font-size: 0.75rem; font-weight: 400;
    opacity: 0.85; margin-top: 2px;
  }}
  footer {{
    text-align: center; color: var(--muted);
    font-size: 0.78rem; padding: 24px 16px 32px;
  }}
</style>
</head>
<body>
<header>
  <h1>Prédiction Prix Gazole – 14 jours</h1>
  <div class="sub">
    Mise à jour : {now_paris}
    <a href="{REPO_URL}" class="repo-link" target="_blank" rel="noopener">repo GitHub</a>
  </div>
</header>
<main>

  <section>
    <div class="cards">
      <div class="card">
        <div class="label">Dernier prix connu</div>
        <div class="value">{fr_num(last_price, 3)} €/L</div>
        <div class="sub">{fr_date(last_date)}</div>
      </div>
      <div class="card">
        <div class="label">Prix prédit J+14</div>
        <div class="value">{fr_num(pred_14, 3)} €/L</div>
        <div class="sub">IC 95 % : {fr_num(pred_14_lo, 3)} – {fr_num(pred_14_hi, 3)} €/L</div>
      </div>
      <div class="card">
        <div class="label">Variation cumulée prédite</div>
        <div class="value" style="color:{var_color};">{fr_pct(var_cumul, 2)}</div>
        <div class="sub">sur 14 jours</div>
      </div>
    </div>
  </section>

  <section>
    <h2>Historique récent et prévision 14 jours</h2>
    <div class="chart"><div id="main-chart"></div></div>
  </section>

  <section>
    <h2>Détail des prédictions J+1 à J+14</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th class="num">Prix prédit</th>
            <th class="num">Var. vs dernier connu</th>
            <th class="num">IC bas (€/L)</th>
            <th class="num">IC haut (€/L)</th>
          </tr>
        </thead>
        <tbody>
          {table_html}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Performance du modèle</h2>
    <div class="metrics-cards">
      <div class="card">
        <div class="label">MAPE 14 j</div>
        <div class="value">{fr_num(metrics['mape'], 2)}\u202F%</div>
        <div class="sub">Erreur absolue moyenne</div>
      </div>
      <div class="card">
        <div class="label">Couverture IC 95 %</div>
        <div class="value">{fr_num(metrics['coverage'], 1)}\u202F%</div>
        <div class="sub">Cible : 95 %</div>
      </div>
      <div class="card">
        <div class="label">Précision directionnelle</div>
        <div class="value">{fr_num(metrics['directional'], 1)}\u202F%</div>
        <div class="sub">Sens de variation J+1</div>
      </div>
      <div class="card">
        <div class="label">Modèle sélectionné</div>
        <div class="value" style="font-size:1.15rem;">{model_name}</div>
        <div class="sub">{justification_text(model_name, mape_results)}</div>
      </div>
    </div>
  </section>

  <section>
    <h2>Erreurs des 14 dernières prédictions J+1</h2>
    <div class="chart"><div id="errors-chart"></div></div>
  </section>

  <section>
    <h2>Contexte marchés</h2>
    <div class="market-grid">
      {market_html}
    </div>
  </section>

  <section style="text-align:center; padding-top:8px;">
    <a class="recompute" href="{WORKFLOW_URL}" target="_blank" rel="noopener"
       title="Cliquer pour ouvrir GitHub Actions, puis 'Run workflow'.">
      Recalculer maintenant
      <span class="hint">Ouvre GitHub Actions, puis « Run workflow »</span>
    </a>
  </section>
</main>
<footer>
  Prédictions automatiques quotidiennes – modèles SARIMAX et XGBoost comparés en backtest 30 jours.
</footer>
<script>
  var mainFig = {main_json};
  Plotly.newPlot('main-chart', mainFig.data, mainFig.layout,
                 {{responsive: true, displayModeBar: false}});
  var errorsFig = {errors_json};
  if (errorsFig) {{
    Plotly.newPlot('errors-chart', errorsFig.data, errorsFig.layout,
                   {{responsive: true, displayModeBar: false}});
  }}
</script>
</body>
</html>
"""
    Path(PATH_HTML).write_text(html, encoding="utf-8")
    size_kb = len(html.encode("utf-8")) // 1024
    log.info(f"  -> {PATH_HTML} ({size_kb} Ko)")


def save_predictions_csv(forecast_df):
    """
    Écrit deux fichiers :
      - predictions.csv : les 14 prévisions du jour (écrasé à chaque run)
      - predictions/predictions_AAAA-MM-JJ.csv : archive horodatée
    """
    out = forecast_df.reset_index()[["Date", "Prix_predit", "IC_bas", "IC_haut"]].copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    for c in ("Prix_predit", "IC_bas", "IC_haut"):
        out[c] = out[c].round(4)

    # 1. Le fichier "courant" (toujours à la racine, écrasé)
    out.to_csv(PATH_PREDICTIONS_CSV, index=False)
    log.info(f"  -> {PATH_PREDICTIONS_CSV}")

    # 2. L'archive horodatée (jamais écrasée)
    archive_dir = Path("predictions")
    archive_dir.mkdir(exist_ok=True)
    today = datetime.now(TZ_PARIS).strftime("%Y-%m-%d")
    archive_path = archive_dir / f"predictions_{today}.csv"
    out.to_csv(archive_path, index=False)
    log.info(f"  -> {archive_path}")


# =============================================================================
#  ORCHESTRATION
# =============================================================================
def main():
    log.info("=== Démarrage prédiction Gazole ===")
    df_p, df_b, df_e = load_data()
    df = prepare_data(df_p, df_b, df_e)

    if len(df) < MAX_LAG + BACKTEST_SELECTION_DAYS + 30:
        log.error("Pas assez de données pour faire le backtest. Abandon.")
        sys.exit(2)

    # 1. Sélection automatique du modèle (MAPE J+1 sur 30j)
    best_model, mape_results = select_model(df)

    # 2. Prévisions exogènes sur 14 jours
    log.info("Prévision des exogènes (ARIMA 1,1,1)…")
    fut_brent = forecast_exog_arima(df["Brent_EUR"], HORIZON)
    fut_eur = forecast_exog_arima(df["EUR_USD"], HORIZON)

    # 3. Prédiction J+1 à J+14 avec le modèle gagnant
    log.info(f"Prédiction 14 jours avec {best_model}…")
    if best_model == "SARIMAX":
        forecast_df = predict_sarimax_full(df, fut_brent, fut_eur, HORIZON)
    else:
        X, y = make_training_set(df, len(df))
        feat_cols = X.columns.tolist()
        models = fit_xgb_quantile(X, y)
        forecast_df = predict_xgb_recursive(df, models, feat_cols, fut_brent, fut_eur, HORIZON)

    # 4. Métriques (backtest 14j avec le modèle gagnant)
    metrics = compute_metrics(df, best_model)

    # 5. Contexte marchés
    market = market_context(df)

    # 6. Sorties
    save_predictions_csv(forecast_df)
    generate_html(df, forecast_df, metrics, market, best_model, mape_results)

    log.info("=== Terminé ===")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        log.exception(f"Erreur fatale : {e}")
        sys.exit(1)
