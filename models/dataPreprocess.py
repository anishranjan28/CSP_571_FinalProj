from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import os

from ucimlrepo import fetch_ucirepo

def pull_uci_data(id: int) -> pd.DataFrame:
    print("Fetching Online Retail dataset from the UCI ...")
    online_retail = fetch_ucirepo(id=id)

    X = online_retail.data.features
    ids = online_retail.data.ids
    df = ids.join(X)

    print(f"Raw shape: {df.shape}")
    print("Missing values per column:\n", df.isnull().sum(), sep="")
    return df

## ====================================
## Cleaning
## ====================================

def data_cleaning(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    ## Drop rows without CustomerID
    df = df.dropna(subset=["CustomerID"]).copy()

    # Fill missing product descriptions
    df["Description"] = df["Description"].fillna("Unknown Product")

    # Remove duplicate rows
    dup_count = df.duplicated().sum()
    print(f"Duplicate rows removed: {dup_count}")
    df = df.drop_duplicates().reset_index(drop=True)

    # Drop non-product / administratice StockCodes
    #   (postage, manual adjustments, bank_charges, amazon fees,  bad debt, etc)
    bad_codes = {"POST", "D", "C2", "M", "BANK CHARGES", "AMAZONFEE",
             "DOT", "CRUK", "PADS", "B"}
    df = df[~df['StockCode'].str.upper().isin(bad_codes)]

    # Cancellations: InvoiceNo starting with 'C' = cancelled order
    # Keep them in a separate frame for return behaviour features, then
    #  drop from the main transactions frame
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    cancellations = df[df['InvoiceNo'].str.startswith("C")].copy()
    df = df[~df['InvoiceNo'].str.startswith("C")]

    # Keep only strictly positive quantities and prices
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Types
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int)

    # Derived cols
    df['Amount'] = df['Quantity'] * df['UnitPrice']
    df['Date']   = df['InvoiceDate'].dt.date
    df['Time']   = df['InvoiceDate'].dt.time
    df['Hour']   = df['InvoiceDate'].dt.hour
    df['DOW']    = df['InvoiceDate'].dt.day_name()
    df['month']  = df['InvoiceDate'].dt.month

    print(f"Clean transaction shape: {df.shape}")
    print(f"Date range: {df['InvoiceDate'].min()} -> {df['InvoiceDate'].max()}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    print(f"Unique invoices : {df['InvoiceNo'].nunique()}")
    print(f"Countries       : {df['Country'].nunique()}")

    #============================================
    # FEATURE ENGINEERING
    #============================================

    # Reference data = day aafter last transaction, the standard RFM conversion
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # RFM core
    rfm = df.groupby("CustomerID").agg(
        Recency     = ("InvoiceDate", lambda s: (snapshot_date - s.max()).days),
        Frequency   = ("InvoiceNo",   "nunique"),
        Monetary    = ("Amount",      "sum"),
    )

    # extra behavioral features useful for sementation beyond pure RFM
    extras = df.groupby("CustomerID").agg(
        TotalQuantity       = ("Quantity",  "sum"),
        UniqueProducts      = ("StockCode", "nunique"),
        AvgUnitPrice        = ("UnitPrice", "mean"),
        AvgBasketValue      = ("Amount",    "mean"),
        FirstPurchase       = ("InvoiceDate", "min"),
        LastPurchase        = ("InvoiceDate", "max")
    )

    # Average order value (per invoice) and items per order
    order_totals = df.groupby(['CustomerID', 'InvoiceNo']).agg(
        order_value  = ("Amount",   "sum"),
        order_items  = ("Quantity", "sum"),
    ).reset_index()

    per_order = order_totals.groupby("CustomerID").agg(
        AvgOrderValue   = ("order_value", "mean"),
        AvgOrderItems   = ("order_items", "mean"),
    )

    # Customer tenure and inter-purchase rhythm
    extras['TenureDays'] = (extras['LastPurchase'] - extras['FirstPurchase']).dt.days

    # Return behaviour - number of cancelled invoices per customer
    cancellations["CustomerID"] = pd.to_numeric(
        cancellations['CustomerID'], errors='coerce'
    )
    returns = (
        cancellations.dropna(subset=['CustomerID'])
        .assign(CustomerID=lambda d: d["CustomerID"].astype(int))
        .groupby("CustomerID")
        .agg(ReturnInvoices=("InvoiceNo", "nunique"))
    )

    # Dominant country per customer (mode)
    country = (
        df.groupby("CustomerID")["Country"]
        .agg(lambda s: s.mode().iat[0])
        .rename("Country")
    )

    # Assemble
    customer_features = (
        rfm
        .join(extras)
        .join(per_order)
        .join(returns)
        .join(country)
    )

    customer_features["ReturnInvoices"] = customer_features["ReturnInvoices"].fillna(0).astype(int)
    customer_features["ReturnRate"] = (
        customer_features["ReturnInvoices"] / 
        (customer_features["Frequency"] + customer_features["ReturnInvoices"])
    )

    # Log-transformed variants - RFM is extremly righ skewed and clustering
    for col in ["Recency", "Frequency", "Monetary",
            "TotalQuantity", "UniqueProducts", "AvgOrderValue"]:
        customer_features[f"log_{col}"] = np.log1p(customer_features[col])

    customer_features = customer_features.reset_index()

    return df, customer_features

def runner() -> None:
    outdir = Path.cwd().parent / "data"
    os.makedirs(outdir, exist_ok=True)
    uci_data = pull_uci_data(id=352)
    df, customer_features = data_cleaning(uci_data)

    df.to_csv(outdir / "online_retail_clean.csv", index=False)
    customer_features.to_csv(outdir / "customer_features.csv", index=False)

    print("\nSaved:")
    print(" ", outdir / "online_retail_clean.csv", df.shape)
    print(" ", outdir / "customer_features.csv", customer_features.shape)

    print("\nCustomer feature preview:")
    print(customer_features.head())
    print("\nSummary stats:")
    print(customer_features[["Recency", "Frequency", "Monetary",
                            "AvgOrderValue", "TenureDays", "ReturnRate"]]
        .describe().round(2))

if __name__ == '__main__':
    runner()