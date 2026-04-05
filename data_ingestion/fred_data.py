import pandas as pd
from fredapi import Fred

FRED_SERIES = {
    "GDP": "GDP",
    "UNRATE": "UNRATE",
    "CPI": "CPIAUCSL",
    "INTEREST_RATE": "FEDFUNDS",
    "OIL_PRICE": "DCOILWTICO"
}

def fetch_fred_data(api_key, start_date, end_date):
    print("Fetching FRED macroeconomic data...")

    fred = Fred(api_key=api_key)

    all_data = []

    for name, series_id in FRED_SERIES.items():
        print(f"Fetching {name} ({series_id})...")

        data = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date
        )

        df = pd.DataFrame(data, columns=["value"])
        df.reset_index(inplace=True)
        df.columns = ["date", "value"]

        df["indicator"] = name

        all_data.append(df)

    macro_df = pd.concat(all_data, ignore_index=True)

    print("FRED data fetched successfully.")
    return macro_df