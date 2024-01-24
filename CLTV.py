# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

import pandas as pd

pd.set_option('display.float_format', lambda x: '%.5f' % x)
from sklearn.preprocessing import MinMaxScaler

df_ = pd.read_excel("datafiles/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()

df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df["Quantity"] > 0)]
df.dropna(inplace=True)
df["TotalPrice"] = df["Quantity"] * df["Price"]

cltv_df = df.groupby("Customer ID").agg({"Invoice": lambda x: len(x),
                                         "Quantity": lambda x: x.sum(),
                                         "TotalPrice": lambda x: x.sum()})
cltv_df.columns = ["total_transaction", "total_unit", "total_price"]

cltv_df.shape[0]
cltv_df["avg_order_value"] = cltv_df["total_price"] / cltv_df["total_transaction"]

cltv_df["purchase_frequency"] = cltv_df["total_transaction"] / cltv_df.shape[0]

repeat_rate = cltv_df[cltv_df.total_transaction > 1].shape[0] / cltv_df.shape[0]
churn_rate = 1 - repeat_rate

cltv_df["profit_margin"] = cltv_df["total_price"] * 0.05

cltv_df["CV"] = (cltv_df["avg_order_value"] * cltv_df["purchase_frequency"]) / churn_rate

cltv_df["CLTV"] = cltv_df["CV"] * cltv_df["profit_margin"]

cltv_df.sort_values("CLTV", ascending=False)
scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_df[["CLTV"]])
cltv_df["SCALED_CLTV"] = scaler.transform(cltv_df[["CLTV"]])
cltv_df.sort_values("CLTV", ascending=False)
cltv_df[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].sort_values(by="SCALED_CLTV",
                                                                                               ascending=False).head()
cltv_df["segment"] = pd.qcut(cltv_df["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])

cltv_df[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV", "segment"]].sort_values(
    by="SCALED_CLTV",
    ascending=False).head()
cltv_df.groupby("segment")[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].agg(
    {"count", "mean", "sum"})
