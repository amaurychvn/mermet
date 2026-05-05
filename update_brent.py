import yfinance as yf
import pandas as pd
import os

CSV_PATH = "brent_usd.csv"
TICKER = "BZ=F"  # Brent Crude Oil Futures (ICE)

def main():
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH, parse_dates=["Date"])
        start = (existing["Date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        existing = pd.DataFrame(columns=["Date", "Prix_USD"])
        start = "2024-01-01"

    new = yf.download(TICKER, start=start, progress=False, auto_adjust=False)
    if new.empty:
        print("Aucune nouvelle donnée Brent.")
        return

    new = new[["Close"]].reset_index()
    new.columns = ["Date", "Prix_USD"]
    new["Date"] = pd.to_datetime(new["Date"]).dt.tz_localize(None).dt.normalize()
    new["Prix_USD"] = new["Prix_USD"].round(3)

    out = pd.concat([existing, new]).drop_duplicates(subset=["Date"]).sort_values("Date")
    out.to_csv(CSV_PATH, index=False)
    print(f"Brent : {len(new)} nouvelle(s) ligne(s). Total : {len(out)}.")

if __name__ == "__main__":
    main()
