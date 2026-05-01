"""
Online Retail (UCI id=352) — data ingest, cleaning, and feature engineering
for an unsupervised customer-segmentation project.

Outputs (in <repo>/data/):
  online_retail_clean.csv     positive-quantity, positive-price line items
  online_retail_cancelled.csv cancellation invoices kept SEPARATELY
  customer_features.csv       customer-level RFM + behavioural features
  data_quality_summary.txt    headline counts and sanity checks

Run:
    pip install ucimlrepo pandas numpy
    python dataPreprocess.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


# ---------------------------------------------------------------------------
# 1. Fetch
# ---------------------------------------------------------------------------
print("Fetching Online Retail dataset from UCI ...")
online_retail = fetch_ucirepo(id=352)
df = online_retail.data.ids.join(online_retail.data.features)

# Force the canonical key name *before* anything else touches the frame.
df = df.rename(columns={"CustomerId": "CustomerID"})

print(f"Raw shape: {df.shape}")
print("Missing per column:\n", df.isnull().sum(), sep="")


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

# Drop non-product / administrative StockCodes (postage, manual adjustments,
# bank charges, Amazon fees, bad debt, etc.).
bad_codes = {"POST", "D", "C2", "M", "BANK CHARGES", "AMAZONFEE",
             "DOT", "CRUK", "PADS", "B"}
df = df[~df["StockCode"].astype(str).str.upper().isin(bad_codes)]

# Cancellations: InvoiceNo starting with 'C'. Keep these in a SEPARATE
# table so we can compute net revenue features alongside the gross ones.
cancellations = df[df["InvoiceNo"].str.startswith("C")].copy()
df = df[~df["InvoiceNo"].str.startswith("C")]

# Positive-only transactions for the main frame.
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

df["InvoiceDate"]            = pd.to_datetime(df["InvoiceDate"])
cancellations["InvoiceDate"] = pd.to_datetime(cancellations["InvoiceDate"])

# Row-level derived columns on the positive frame.
df["Amount"] = df["Quantity"] * df["UnitPrice"]
df["Date"]   = df["InvoiceDate"].dt.date
df["Time"]   = df["InvoiceDate"].dt.time
df["Hour"]   = df["InvoiceDate"].dt.hour
df["DOW"]    = df["InvoiceDate"].dt.day_name()
df["Month"]  = df["InvoiceDate"].dt.month

# Cancellations: Amount preserves the negative sign (Quantity is negative).
cancellations["Amount"] = cancellations["Quantity"] * cancellations["UnitPrice"]


# ---------------------------------------------------------------------------
# 3. Customer-level feature engineering
# ---------------------------------------------------------------------------
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

# 3a. RFM core (gross — positive sales only).
rfm = df.groupby("CustomerID").agg(
    Recency       = ("InvoiceDate", lambda s: (snapshot_date - s.max()).days),
    Frequency     = ("InvoiceNo",   "nunique"),
    GrossMonetary = ("Amount",      "sum"),
)

# 3b. Behavioural extras.
extras = df.groupby("CustomerID").agg(
    TotalQuantity   = ("Quantity",    "sum"),
    UniqueProducts  = ("StockCode",   "nunique"),
    FirstPurchase   = ("InvoiceDate", "min"),
    LastPurchase    = ("InvoiceDate", "max"),
)
extras["TenureDays"]     = (extras["LastPurchase"] - extras["FirstPurchase"]).dt.days
extras["SinglePurchase"] = (rfm["Frequency"] == 1).astype(int)

# 3c. Quantity-weighted average unit price = sum(Amount) / sum(Quantity).
#     This is the price-per-item the customer ACTUALLY paid on average,
#     not the unweighted mean of line-level UnitPrice (which is biased
#     by lots of tiny low-priced items).
weighted_unit_price = (
    df.groupby("CustomerID")
      .apply(lambda g: g["Amount"].sum() / g["Quantity"].sum(),
             include_groups=False)
      .rename("AvgUnitPrice")
)

# 3d. Order-level features.
order_totals = df.groupby(["CustomerID", "InvoiceNo"]).agg(
    order_value = ("Amount",   "sum"),
    order_items = ("Quantity", "sum"),
).reset_index()

per_order = order_totals.groupby("CustomerID").agg(
    AvgOrderValue = ("order_value", "mean"),
    AvgOrderItems = ("order_items", "mean"),
)

# 3e. Cancellation features per customer.
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

# 3f. Dominant country per customer.
country = (
    df.groupby("CustomerID")["Country"]
      .agg(lambda s: s.mode().iat[0])
      .rename("Country")
)

# 3g. Assemble.
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

# 3h. Net monetary = positive sales minus absolute return value.
#     Most customers have 0 returns; some have small partial cancellations;
#     a handful are net-zero or slightly negative (rare, but real).
customer["NetMonetary"] = customer["GrossMonetary"] - customer["ReturnValueAbs"]

# Two return-rate variants:
#   ReturnRate         (count-based)  — sparse, mostly zero
#   ReturnValueRate    (value-based)  — better for clustering interpretation
customer["ReturnRate"] = customer["ReturnInvoices"] / (
    customer["Frequency"] + customer["ReturnInvoices"]
)
customer["ReturnValueRate"] = (
    customer["ReturnValueAbs"] / customer["GrossMonetary"]
)

# 3i. Winsorise AvgUnitPrice at the 99th percentile to stop a few extreme
#     items distorting PCA loadings on PC2 / PC3. Keep the raw column too.
p99 = customer["AvgUnitPrice"].quantile(0.99)
customer["AvgUnitPrice_winsor"] = customer["AvgUnitPrice"].clip(upper=p99)

# 3j. Log-transformed variants. Use signed log1p for NetMonetary so the rare
#     negative-net customers don't crash the pipeline.
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))

for col in ["Recency", "Frequency", "GrossMonetary",
            "TotalQuantity", "UniqueProducts", "AvgOrderValue"]:
    customer[f"log_{col}"] = np.log1p(customer[col])

customer["log_NetMonetary"] = signed_log1p(customer["NetMonetary"])

# Backwards-compat aliases for older notebooks that referenced Monetary /
# log_Monetary directly.
customer["Monetary"]     = customer["GrossMonetary"]
customer["log_Monetary"] = customer["log_GrossMonetary"]

customer = customer.reset_index()

# Validation — fail fast if anything is off.
assert customer["CustomerID"].is_unique, "CustomerID must be unique."
assert customer["CustomerID"].dtype.kind in "iu", "CustomerID must be integer."
assert not customer.isna().any().any(), \
    "Unexpected NaNs in customer feature table."


# ---------------------------------------------------------------------------
# 4. Save
# ---------------------------------------------------------------------------
out_dir = Path(__file__).resolve().parent.parent / "data"
out_dir.mkdir(exist_ok=True)

df.to_csv(out_dir / "online_retail_clean.csv",     index=False)
cancellations.to_csv(out_dir / "online_retail_cancelled.csv", index=False)
customer.to_csv(out_dir / "customer_features.csv", index=False)


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
    customer["AvgUnitPrice"]
        .describe(percentiles=[.5, .9, .99]).to_string(),
    "",
    f"AvgUnitPrice winsorised at p99 = {p99:.2f}",
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
(out_dir / "data_quality_summary.txt").write_text(quality)

print("\nSaved:")
print(" ", out_dir / "online_retail_clean.csv",     df.shape)
print(" ", out_dir / "online_retail_cancelled.csv", cancellations.shape)
print(" ", out_dir / "customer_features.csv",       customer.shape)
print(" ", out_dir / "data_quality_summary.txt")
