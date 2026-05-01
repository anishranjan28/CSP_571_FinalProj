"""
Online Retail (UCI id=352) — data ingest, cleaning, and customer-level
feature engineering.

Reads paths and parameters from the project-level `config.yml` so paths
are not hardcoded twice.

Outputs (in directories defined by config.yml):
    online_retail_clean.csv      positive transactions only
    online_retail_cancelled.csv  cancellation invoices kept SEPARATELY
    customer_features.csv        customer-level features for PCA/clustering
    data_quality_summary.txt     headline counts and sanity checks

Run from repo root:
    python src/python/data_prep.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from ucimlrepo import fetch_ucirepo


# ---------------------------------------------------------------------------
# 0. Locate repo root + load config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yml"

if not CONFIG_PATH.exists():
    print(f"ERROR: config.yml not found at {CONFIG_PATH}", file=sys.stderr)
    sys.exit(1)

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DATA = CONFIG["data"]
WINSOR_Q = CONFIG["features"]["winsor_quantile"]


def _abspath(rel: str) -> Path:
    return REPO_ROOT / rel


# ---------------------------------------------------------------------------
# 1. Fetch
# ---------------------------------------------------------------------------
print("Fetching Online Retail dataset from UCI ...")
online_retail = fetch_ucirepo(id=352)
df = online_retail.data.ids.join(online_retail.data.features)
df = df.rename(columns={"CustomerId": "CustomerID"})
print(f"Raw shape: {df.shape}")


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
df = df.dropna(subset=["CustomerID"]).copy()
df["Description"] = df["Description"].fillna("Unknown Product")

before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed: {before - len(df)}")

df["CustomerID"] = df["CustomerID"].astype(int)
df["InvoiceNo"]  = df["InvoiceNo"].astype(str)

bad_codes = {"POST", "D", "C2", "M", "BANK CHARGES", "AMAZONFEE",
             "DOT", "CRUK", "PADS", "B"}
df = df[~df["StockCode"].astype(str).str.upper().isin(bad_codes)]

cancellations = df[df["InvoiceNo"].str.startswith("C")].copy()
df = df[~df["InvoiceNo"].str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

df["InvoiceDate"]            = pd.to_datetime(df["InvoiceDate"])
cancellations["InvoiceDate"] = pd.to_datetime(cancellations["InvoiceDate"])

df["Amount"] = df["Quantity"] * df["UnitPrice"]
df["Date"]   = df["InvoiceDate"].dt.date
df["Time"]   = df["InvoiceDate"].dt.time
df["Hour"]   = df["InvoiceDate"].dt.hour
df["DOW"]    = df["InvoiceDate"].dt.day_name()
df["Month"]  = df["InvoiceDate"].dt.month

cancellations["Amount"] = cancellations["Quantity"] * cancellations["UnitPrice"]


# ---------------------------------------------------------------------------
# 3. Customer-level feature engineering
# ---------------------------------------------------------------------------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg(
    Recency       = ("InvoiceDate", lambda s: (snapshot_date - s.max()).days),
    Frequency     = ("InvoiceNo",   "nunique"),
    GrossMonetary = ("Amount",      "sum"),
)

extras = df.groupby("CustomerID").agg(
    TotalQuantity   = ("Quantity",    "sum"),
    UniqueProducts  = ("StockCode",   "nunique"),
    FirstPurchase   = ("InvoiceDate", "min"),
    LastPurchase    = ("InvoiceDate", "max"),
)
extras["TenureDays"]     = (extras["LastPurchase"] - extras["FirstPurchase"]).dt.days
extras["SinglePurchase"] = (rfm["Frequency"] == 1).astype(int)

# Quantity-weighted unit price.
weighted_unit_price = (
    df.groupby("CustomerID")
      .apply(lambda g: g["Amount"].sum() / g["Quantity"].sum(),
             include_groups=False)
      .rename("AvgUnitPrice")
)

order_totals = df.groupby(["CustomerID", "InvoiceNo"]).agg(
    order_value = ("Amount",   "sum"),
    order_items = ("Quantity", "sum"),
).reset_index()

per_order = order_totals.groupby("CustomerID").agg(
    AvgOrderValue = ("order_value", "mean"),
    AvgOrderItems = ("order_items", "mean"),
)

cancellations["CustomerID"] = pd.to_numeric(
    cancellations["CustomerID"], errors="coerce"
)
cancellations = cancellations.dropna(subset=["CustomerID"]).copy()
cancellations["CustomerID"] = cancellations["CustomerID"].astype(int)

returns_amt = (
    cancellations.groupby("CustomerID").agg(
        ReturnInvoices = ("InvoiceNo", "nunique"),
        ReturnValueAbs = ("Amount", lambda s: float(np.abs(s).sum())),
    )
)

country = (
    df.groupby("CustomerID")["Country"]
      .agg(lambda s: s.mode().iat[0])
      .rename("Country")
)

customer = (
    rfm
      .join(extras)
      .join(weighted_unit_price)
      .join(per_order)
      .join(returns_amt)
      .join(country)
)

customer["ReturnInvoices"] = customer["ReturnInvoices"].fillna(0).astype(int)
customer["ReturnValueAbs"] = customer["ReturnValueAbs"].fillna(0.0)

customer["NetMonetary"] = customer["GrossMonetary"] - customer["ReturnValueAbs"]

customer["ReturnRate"] = customer["ReturnInvoices"] / (
    customer["Frequency"] + customer["ReturnInvoices"]
)
customer["ReturnValueRate"] = (
    customer["ReturnValueAbs"] / customer["GrossMonetary"]
)

p_winsor = customer["AvgUnitPrice"].quantile(WINSOR_Q)
customer["AvgUnitPrice_winsor"] = customer["AvgUnitPrice"].clip(upper=p_winsor)


def _signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


for col in ["Recency", "Frequency", "GrossMonetary",
            "TotalQuantity", "UniqueProducts", "AvgOrderValue"]:
    customer[f"log_{col}"] = np.log1p(customer[col])

customer["log_NetMonetary"] = _signed_log1p(customer["NetMonetary"])

# Backwards-compat aliases (older notebooks reference Monetary / log_Monetary).
customer["Monetary"]     = customer["GrossMonetary"]
customer["log_Monetary"] = customer["log_GrossMonetary"]

customer = customer.reset_index()

assert customer["CustomerID"].is_unique, "CustomerID must be unique."
assert customer["CustomerID"].dtype.kind in "iu", \
    "CustomerID must be integer."
assert not customer.isna().any().any(), \
    "Unexpected NaNs in customer feature table."


# ---------------------------------------------------------------------------
# 4. Save (paths come from config.yml)
# ---------------------------------------------------------------------------
for k in ("interim_dir", "processed_dir", "outputs_dir"):
    _abspath(DATA[k]).mkdir(parents=True, exist_ok=True)

df.to_csv(_abspath(DATA["online_retail_clean"]),     index=False)
cancellations.to_csv(_abspath(DATA["online_retail_cancelled"]), index=False)
customer.to_csv(_abspath(DATA["customer_features"]), index=False)


# ---------------------------------------------------------------------------
# 5. Data-quality summary
# ---------------------------------------------------------------------------
quality_lines = [
    "Online Retail — data quality summary",
    "=" * 60,
    f"Date range            : {df['InvoiceDate'].min()}  ->  "
    f"{df['InvoiceDate'].max()}",
    f"Transactions (pos)    : {len(df):,}",
    f"Cancellations         : {len(cancellations):,}",
    f"Unique invoices (pos) : {df['InvoiceNo'].nunique():,}",
    f"Unique products       : {df['StockCode'].nunique():,}",
    f"Unique customers      : {df['CustomerID'].nunique():,}",
    f"Countries             : {df['Country'].nunique()}",
    "",
    "Customer feature table",
    "-" * 60,
    f"Rows                  : {len(customer):,}",
    f"Columns               : {customer.shape[1]}",
    f"CustomerID unique?    : {bool(customer['CustomerID'].is_unique)}",
    f"NaN cells             : {int(customer.isna().sum().sum())}",
    "",
    "AvgUnitPrice (weighted) summary:",
    customer["AvgUnitPrice"].describe(percentiles=[.5, .9, .99]).to_string(),
    "",
    f"AvgUnitPrice winsorised at p{int(WINSOR_Q*100)} = {p_winsor:.2f}",
    "",
    "Net vs gross monetary:",
    customer[["GrossMonetary", "NetMonetary", "ReturnValueAbs"]]
        .describe().round(2).to_string(),
    "",
    f"Customers with NetMonetary <= 0  : "
    f"{int((customer['NetMonetary'] <= 0).sum())}",
    f"Customers with SinglePurchase=1  : "
    f"{int(customer['SinglePurchase'].sum())}",
]
quality = "\n".join(quality_lines)
print("\n" + quality)
_abspath(DATA["data_quality_summary"]).write_text(quality)

print("\nSaved:")
print(" ", DATA["online_retail_clean"],     df.shape)
print(" ", DATA["online_retail_cancelled"], cancellations.shape)
print(" ", DATA["customer_features"],       customer.shape)
print(" ", DATA["data_quality_summary"])
