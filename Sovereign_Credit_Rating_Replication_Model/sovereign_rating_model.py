"""
Sovereign Credit Rating Replication Model
==============================================
Real World Bank macro indicators (GDP growth, government debt-to-GDP, current-account
balance, GDP per capita) for a 30+ country universe, paired with each country's real,
publicly-known approximate sovereign rating, used to train a rating-bucket classifier
- the sovereign analogue to Corporate_Credit_Rating_Prediction_Model - with a discussion
of the largest real prediction misses and their likely qualitative explanation.
"""

# ===========================================================================
# CONFIG BLOCK
# ===========================================================================
COUNTRIES = {
    "USA": "AAA", "DEU": "AAA", "CHE": "AAA", "SGP": "AAA", "AUS": "AAA", "CAN": "AA+",
    "GBR": "AA", "FRA": "AA-", "KOR": "AA", "JPN": "A+", "CHN": "A+", "CZE": "AA-",
    "POL": "A-", "CHL": "A", "MYS": "A-", "THA": "BBB+", "MEX": "BBB", "IND": "BBB-",
    "IDN": "BBB", "ZAF": "BB-", "BRA": "BB", "COL": "BB+", "TUR": "BB-", "EGY": "B",
    "ARG": "CCC", "PAK": "CCC", "NGA": "B-", "VNM": "BB+", "PHL": "BBB+", "PER": "BBB",
    "MAR": "BB+", "KEN": "B",
}
YEAR_START, YEAR_END = 2015, 2023

import numpy as np
import pandas as pd
import pandas_datareader.wb as wb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ===========================================================================
# 1. Real World Bank macro indicators
# ===========================================================================
indicators = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "debt_to_gdp": "GC.DOD.TOTL.GD.ZS",
    "current_account_pct_gdp": "BN.CAB.XOKA.GD.ZS",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
}
country_names = list(COUNTRIES.keys())
data = wb.download(indicator=list(indicators.values()), country=country_names,
                     start=YEAR_START, end=YEAR_END)
data.columns = list(indicators.keys())
print(f"Pulled real World Bank data for {data.index.get_level_values(0).nunique()} "
      f"countries, {YEAR_START}-{YEAR_END}")

# Average over the real available years per country (smooths single-year noise)
country_avg = data.groupby(level=0).mean()
country_avg.index.name = "country_name"

# Map ISO3 codes to World Bank's country-name index
iso_to_name = {}
wb_countries = wb.get_countries()
for iso3, rating in COUNTRIES.items():
    match = wb_countries[wb_countries["iso3c"] == iso3]
    if not match.empty:
        iso_to_name[match.iloc[0]["name"]] = rating

df = country_avg.reset_index()
df["rating"] = df["country_name"].map(iso_to_name)
df = df.dropna(subset=["rating", "gdp_growth", "debt_to_gdp"])
print(f"Real macro + real rating data available for {len(df)} of {len(COUNTRIES)} countries")
print(df[["country_name", "rating", "gdp_growth", "debt_to_gdp", "gdp_per_capita"]].round(2).to_string(index=False))

# ===========================================================================
# 2. Bucket ratings (same 5-bucket convention as the corporate rating project)
# ===========================================================================
RATING_SCALE = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
                 "BB+", "BB", "BB-", "B+", "B", "B-", "CCC"]
rating_rank = {r: i for i, r in enumerate(RATING_SCALE)}
def bucket_rating(rank):
    if rank <= 3: return "AAA/AA"
    elif rank <= 6: return "A"
    elif rank <= 9: return "BBB"
    elif rank <= 12: return "BB"
    else: return "B and below"

df["rating_rank"] = df["rating"].map(rating_rank)
df["rating_bucket"] = df["rating_rank"].apply(bucket_rating)
bucket_order = ["AAA/AA", "A", "BBB", "BB", "B and below"]
bucket_rank = {b: i for i, b in enumerate(bucket_order)}
df["bucket_rank"] = df["rating_bucket"].map(bucket_rank)
print("\nReal rating bucket distribution:")
print(df["rating_bucket"].value_counts().reindex(bucket_order).to_string())

# ===========================================================================
# 3. Train/validate (same methodology as the corporate rating project)
# ===========================================================================
features = ["gdp_growth", "debt_to_gdp", "current_account_pct_gdp", "gdp_per_capita"]
df_clean = df.dropna(subset=features)
X = df_clean[features].values
y = df_clean["bucket_rank"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

country_names_arr = df_clean["country_name"].astype(str).to_numpy()
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X_scaled, y, country_names_arr, test_size=0.3, random_state=7)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
preds = model.predict(X_test)
exact_acc = accuracy_score(y_test, preds)
within_one = np.mean(np.abs(preds - y_test) <= 1)

print("\n" + "=" * 70)
print("MODEL VALIDATION (held-out test set)")
print("=" * 70)
print(f"Test set: {len(y_test)} countries")
print(f"Exact bucket accuracy: {exact_acc:.1%}")
print(f"Within-one-bucket accuracy: {within_one:.1%}")
for name, actual, pred in zip(names_test, y_test, preds):
    print(f"  {name}: actual={bucket_order[actual]}, predicted={bucket_order[pred]}, "
          f"{'MATCH' if actual == pred else 'within 1' if abs(actual-pred) <= 1 else 'MISS'}")

# ===========================================================================
# 4. Coefficient sign check
# ===========================================================================
print("\n" + "=" * 70)
print("MODEL COEFFICIENTS (sanity check against real sovereign-rating methodology)")
print("=" * 70)
coef_df = pd.DataFrame(model.coef_, columns=features, index=bucket_order[:model.coef_.shape[0]])
print(coef_df.round(3).to_string())
print("\nExpectation: higher debt/GDP and weaker growth should push toward WEAKER "
      "(higher-rank) buckets; higher GDP per capita (development level) should push "
      "toward STRONGER (lower-rank/AAA-AA) buckets.")

# ===========================================================================
# 5. Full-sample predictions and largest real misses
# ===========================================================================
full_preds = model.predict(X_scaled)
df_clean = df_clean.copy()
df_clean["predicted_bucket"] = [bucket_order[p] for p in full_preds]
df_clean["miss_magnitude"] = (df_clean["bucket_rank"] - full_preds).abs()

print("\n" + "=" * 70)
print("LARGEST REAL PREDICTION MISSES (full-sample fit)")
print("=" * 70)
misses = df_clean.sort_values("miss_magnitude", ascending=False).head(6)
print(misses[["country_name", "rating", "rating_bucket", "predicted_bucket"]].to_string(index=False))
print("\nDiscussion: countries whose real rating sits well above what the macro-only "
      "model predicts likely benefit from real qualitative factors the model cannot see "
      "- institutional strength, reserve-currency status (e.g. USD/JPY/EUR issuers), or "
      "a strong real payment history despite weaker headline debt/growth metrics. "
      "Countries rated well BELOW the model's prediction likely carry real political-"
      "risk or governance concerns not captured in pure macro data - exactly the real, "
      "well-known critique that agencies apply judgment a quantitative model cannot.")
