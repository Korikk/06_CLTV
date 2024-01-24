import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

# pip install lifetimes


##############################################################
# GÖREV 1
##############################################################
# - 2010-2011 UK müşterileri için 6 aylık CLTV prediction yapınız.
# - Elde ettiğiniz sonuçları yorumlayıp üzerinde değerlendirme yapınız.
# - Mantıksız ya da çok isabetli olduğunu düşündüğünüz sonuçları vurgulayınız.
# - Dikkat! 6 aylık expected sales değil cltv prediction yapılmasını bekliyoruz.
#   Yani direk bgnbd ve gamma modellerini kurarak devam ediniz ve
# - cltv prediction için ay bölümüne 6 giriniz.
# 1. Data Preperation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")

df = df_.copy()
df.shape
df.head()
df.info()
df.describe().T

df = df[df["Country"] == "United Kingdom"]

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]


def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T
df["TotalPrice"] = df["Quantity"] * df["Price"]
df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

#############################################
# RFM Table
#############################################

rfm = df.groupby('Customer ID').agg({"InvoiceDate": [lambda date: (date.max() - date.min()).days,
                                                     lambda date: (today_date - date.min()).days],
                                     "Invoice": lambda num: num.nunique(),
                                     "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.columns=rfm.columns.droplevel(0)

rfm.columns=["recency_cltv_p","T","frequency","monetary"]

rfm["monetary"]=rfm["monetary"]/rfm["frequency"]

rfm.rename(columns={"monetary":"monetary_avg"},inplace=True)
# BGNBD için WEEKLY RECENCY VE WEEKLY T'nin HESAPLANMASI
## recency_weekly_p
rfm["recency_weekly_p"]=rfm["recency_cltv_p"]/7
rfm["T_weekly"]=rfm["T"]/7
#Kontrol
rfm = rfm[rfm["monetary_avg"] > 0]
#Freq>1
rfm=rfm[(rfm["frequency"]>1)]
rfm["frequency"]=rfm["frequency"].astype(int)

##############################################################
# 2. BG/NBD Modelinin Kurulması
##############################################################

bgf=BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(rfm['frequency'],
        rfm['recency_weekly_p'],
        rfm['T_weekly'])
bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                        rfm['frequency'],
                                                        rfm['recency_weekly_p'],
                                                        rfm['T_weekly']).sort_values(ascending=False).head(10)

rfm["expected_number_of_purchases"] = bgf.predict(4,
                                                  rfm['frequency'],
                                                  rfm['recency_weekly_p'],
                                                  rfm['T_weekly'])
rfm.head()
##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################

ggf=GammaGammaFitter(penalizer_coef=0.001)
ggf.fit(rfm["frequency"],rfm["monetary_avg"])

ggf.conditional_expected_average_profit(rfm["frequency"],
                                        rfm["monetary_avg"]).head(10)
#6 aylık

cltv=ggf.customer_lifetime_value(bgf,
                                 rfm['frequency'],
                                 rfm['recency_weekly_p'],
                                 rfm['T_weekly'],
                                 rfm['monetary_avg'],
                                 time=6,  # 6 aylık
                                 freq="W",  # T'nin frekans bilgisi.
                                 discount_rate=0.01).sort_values(ascending=False)
cltv.head()
cltv.shape
cltv = cltv.reset_index()
cltv.sort_values(by="clv",ascending=False).head(50)

rfm_cltv_final=rfm.merge(cltv,on="Customer ID",how="left")
rfm_cltv_final.head(20)
##############################################################
# GÖREV 2
##############################################################
# - 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
# - 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
# - Fark var mı? Varsa sizce neden olabilir?
# Dikkat! Sıfırdan model kurulmasına gerek yoktur.
# Var olan bgf ve ggf üzerinden direk cltv hesaplanabilir.

#1 aylık
cltv_1=ggf.customer_lifetime_value(bgf,
                                 rfm['frequency'],
                                 rfm['recency_weekly_p'],
                                 rfm['T_weekly'],
                                 rfm['monetary_avg'],
                                 time=1,  # 1 aylık
                                 freq="W",  # T'nin frekans bilgisi.
                                 discount_rate=0.01).sort_values(ascending=False)
cltv_1.head(10)
#12 Aylık

cltv_12=ggf.customer_lifetime_value(bgf,
                                 rfm['frequency'],
                                 rfm['recency_weekly_p'],
                                 rfm['T_weekly'],
                                 rfm['monetary_avg'],
                                 time=12,  # 1 aylık
                                 freq="W",  # T'nin frekans bilgisi.
                                 discount_rate=0.01).sort_values(ascending=False)
cltv_12.head(10)
##############################################################
# GÖREV 3
##############################################################
# 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 3 gruba (segmente) ayırınız ve
# grup isimlerini veri setine ekleyiniz. Örneğin (C, B, A)
# CLTV'ye göre en iyi yüzde 20'yi seçiniz. Ve bunlara top_flag yazınız. yüzde 20'ye 1.
# diğerlerine 0 yazınız.

# 3 grubu veri setindeki diğer değişkenler açısıdan analiz ediniz.
# 3 grup için yönetime 6 aylık aksiyon önerilerinde bulununuz. Kısa kısa.
rfm_cltv_final["segment"]=pd.qcut(rfm_cltv_final["clv"],3,labels=["C","B","A"])
rfm_cltv_final.head()


#
top_20_ids = rfm_cltv_final.sort_values(by="clv", ascending=False)[:int(rfm_cltv_final.shape[0] / 100 * 20)]["Customer ID"]
#1-0 ataması yapıldı
rfm_cltv_final["top_flag"] = rfm_cltv_final["Customer ID"].apply(lambda x: 1 if x in top_20_ids.values else 0)
rfm_cltv_final.head()

rfm_cltv_final.sort_values(by="clv", ascending=False).head()
rfm_cltv_final.sort_values(by="clv", ascending=True).head()