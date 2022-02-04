# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:04:02 2021

@author: Shaila Sarker
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
#from sklearn.preprocessing import StandardScaler
#from sklearn.manifold import TSNE
from sklearn.cluster import KMeans



df = pd.read_csv("C:\\Users\\Shaila Sarker\\Documents\\Antibio_Resist\\Feb HD.csv")

# Bad debt adjustments will be dropped from the dataset as these do not represent actual sales.
df.describe()
df.describe(include='O')
df.isnull().mean()*100
# Remove the negative values from Quantity and UnitPrice
df = df[df.SalesQty > 0]
df = df[df.NSI > 0]
df = df[pd.notnull(df['CustomerCode'])]
df.isnull().mean()

# Here we add a new column called TotalPrice.
df['TotalPrice'] = df['NSI']

import datetime as dt
NOW = dt.date(2021,4,21) #YY:MM:DD 

df['date'] = pd.DatetimeIndex(df.InvoiceDate).date 

## RFM Analysis
#Here, we will perform following operations:
#- For Recency, Calculate the number of days between present date and date of last purchase each customer.
#- For Frequency, Calculate the number of orders for each customer.
#- For Monetary, Calculate sum of purchase price for each customer.

df_recency = df.groupby(['CustomerCode'],as_index=False)['date'].max()
df_recency.columns = ['CustomerCode','Last_Purchase_Date']

df_recency['Recency'] = df_recency.Last_Purchase_Date.apply(lambda x:(NOW - x).days)

df_recency.drop(columns=['Last_Purchase_Date'],inplace=True)

FM_Table = df.groupby('CustomerCode').agg({'InvoiceNo'   : lambda x:len(x),
                                         'TotalPrice'  : lambda x:x.sum()})

FM_Table.rename(columns = {'InvoiceNo' :'Frequency',
                           'TotalPrice':'Monetary_Value'},inplace= True)

RFM_Table = df_recency.merge(FM_Table,left_on='CustomerCode',right_on='CustomerCode')

# Now we will split the metrics into segments using quantiles.
# And assign a score from 1 to 4 to Recency, Frequency and Monetary.
# One is the best/highest value, and four is the lowest/worst value.
# A final RFM score is calculated simply by combining individual RFM score numbers.

quantiles = RFM_Table.quantile(q=[0.25,0.50,0.75])
quantiles = quantiles.to_dict()

segmented_rfm = RFM_Table.copy()

def RScore(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FMScore(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1
    
segmented_rfm['R_quartile'] = segmented_rfm['Recency'].apply(RScore, args=('Recency',quantiles))
segmented_rfm['F_quartile'] = segmented_rfm['Frequency'].apply(FMScore, args=('Frequency',quantiles))
segmented_rfm['M_quartile'] = segmented_rfm['Monetary_Value'].apply(FMScore, args=('Monetary_Value',quantiles))
segmented_rfm.head()

# Best Recency score = 1: Purchased very recently. 
    
# Best Frequency score = 1: highest quantity purchases.  

# Best Monetary score = 1: spent the most.

segmented_rfm['RFM_Segment'] = segmented_rfm.R_quartile.map(str)+segmented_rfm.F_quartile.map(str)+segmented_rfm.M_quartile.map(str)
segmented_rfm['RFM_Score'] = segmented_rfm[['R_quartile','F_quartile','M_quartile']].sum(axis=1)

print("Best Customers: ",len(segmented_rfm[segmented_rfm['RFM_Segment']=='111']))
print('Loyal Customers: ',len(segmented_rfm[segmented_rfm['F_quartile']==1]))
print("Big Spenders: ",len(segmented_rfm[segmented_rfm['M_quartile']==1]))
print('Almost Lost: ', len(segmented_rfm[segmented_rfm['RFM_Segment']=='134']))
print('Lost Customers: ',len(segmented_rfm[segmented_rfm['RFM_Segment']=='344']))
print('Lost Cheap Customers: ',len(segmented_rfm[segmented_rfm['RFM_Segment']=='444']))


### Suggested marketing strategies on segmented customers:
## Best Customers- No price incentives, new products, and loyalty programs.
## Big Spenders- Market your most expensive products.
## Almost Lost- Aggresive price incentives
## Lost Customers-Don’t spend too much trying to re-acquire them.

segmented_rfm['RFM_Score'].unique()

segmented_rfm.groupby('RFM_Score').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary_Value': ['mean', 'count'] }).round(1)

# K-means gives the best result under the following conditions:
# - Data’s distribution is not skewed. 
# - Data is standardised (i.e. mean of 0 and standard deviation of 1).

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column])
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

# Plot all 3 graphs together for summary findings
plt.figure(figsize=(9, 9))

plt.subplot(3, 1, 1)
check_skew(RFM_Table,'Recency')

plt.subplot(3, 1, 2)
check_skew(RFM_Table,'Frequency')

plt.subplot(3, 1, 3)
check_skew(RFM_Table,'Monetary_Value')

plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)


# The data is highly skewed,therefore we will perform log transformations to reduce the skewness of each variable.I will add a small constant as log transformation demands all the values to be positive.
df_rfm_log = RFM_Table.copy()
df_rfm_log = np.log(df_rfm_log+1)

plt.figure(figsize=(9, 9))

plt.subplot(3, 1, 1)
check_skew(df_rfm_log,'Recency')

plt.subplot(3, 1, 2)
check_skew(df_rfm_log,'Frequency')

plt.subplot(3, 1, 3)
check_skew(df_rfm_log,'Monetary_Value')

plt.tight_layout()
plt.savefig('after_transform.png', format='png', dpi=1000)

RFM_Table.describe()
RFM_Table2 = RFM_Table
RFM_Table2.drop(columns=['CustomerCode'],inplace=True)
#Normalization
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
RFM_Normalized = pd.DataFrame(Scaler.fit_transform(RFM_Table2))
RFM_Normalized.columns = ['n_Recency', 'n_Frequency', 'n_Monetary']
RFM_Normalized.describe()                            


# Segmenting with K-Means. Identify the optimal k.
# SSE = Sum Squared Err
SSE = []
for k in range(0,10):
	kmeans = KMeans(n_clusters = k+1, random_state = 1231).fit(RFM_Normalized)
	SSE.append(kmeans.inertia_)

sns.pointplot(x=list(range(1,11)), y=SSE)
plt.show()
# Plot of Inertia against k. ‘Elbow’ at k=5, where the decrement in inertia after k=6 is insignificant, it does not worth to further complicate the model.

from yellowbrick.cluster import SilhouetteVisualizer
SilhouetteVisualizer(KMeans(5, random_state=42), RFM_Table, colors = 'yellowbrick')



model = KMeans(n_clusters=5, random_state = 1231).fit(RFM_Normalized)
centers = model.cluster_centers_
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(RFM_Normalized['n_Recency'], RFM_Normalized['n_Frequency'], RFM_Normalized['n_Monetary'], cmap="brg",
	c = model.predict(RFM_Normalized))
ax.scatter(centers[:, 0], centers[:,1], c = 'black');



RFM_ = pd.DataFrame(Scaler.inverse_transform(RFM_Normalized))
RFM_.columns = RFM_Table.columns
RFM_['CustomerCode'] = RFM_Table.index
RFM_['Cluster'] = model.labels_

melted_RFM_Normalized = pd.melt(RFM_.reset_index(),
	id_vars = ['CustomerCode', 'Cluster'],
	value_vars = ['Recency', 'Frequency', 'Monetary_Value'],
	var_name = 'Features',
	value_name = 'Value')
sns.lineplot('Features', 'Value', hue='Cluster', data = melted_RFM_Normalized)
plt.legend()

RFM_.groupby('Cluster').agg({
	'Recency':['mean', 'min', 'max'],
	'Frequency':['mean', 'min', 'max'],
	'Monetary_Value':['mean', 'min', 'max', 'count']})

Final = RFM_.groupby('Cluster').agg({
	'Recency':['mean', 'min', 'max'],
	'Frequency':['mean', 'min', 'max'],
	'Monetary_Value':['mean', 'min', 'max', 'count']})

segmented_rfm['Cluster'] = model.labels_
Final2 = segmented_rfm[['CustomerCode', 'RFM_Score', 'Cluster']]