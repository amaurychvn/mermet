import yfinance as yf
import pandas as pd
import os

CSV_PATH = "eurusd.csv"
TICKER = "EURUSD=X"  # 1 EUR = X USD

def main():
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH, parse_dates=["Date"])
        start = (existing["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        existing = pd.DataFrame(columns=["Date", "Taux"])
        start = "2024-01-01"

    new = yf.download(TICKER, start=start, progress=False, auto_adjust=False)
    if new.empty:
        print("Aucune nouvelle donnée EUR/USD.")
        return

    new = new[["Close"]].reset_index()
    new.columns = ["Date", "Taux"]
    new["Date"] = pd.to_datetime(new["Date"]).dt.tz_localize(None).dt.normalize()
    new["Taux"] = new["Taux"].round(5)

    out = pd.concat([existing, new]).drop_duplicates(subset=["Date"]).sort_values("Date")
    out.to_csv(CSV_PATH, index=False)
    print(f"EUR/USD : {len(new)} nouvelle(s) ligne(s). Total : {len(out)}.")

if __name__ == "__main__":
    main()
