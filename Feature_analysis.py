import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns       
from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
createFolder('Feature_image_data')
data_train = pd.read_csv("houseprice/train.csv")
#analyze the data
analyze=data_train['SalePrice'].describe()
print(analyze)
# show the sale Price
sns.distplot(data_train['SalePrice'])
plt.savefig('Feature_image_data/SalePrice.png')

# check  Skewness and Kurtosis
print("Skewness is ",data_train['SalePrice'].skew())
print("Kurtosis is ",data_train['SalePrice'].kurt())

# boxplot for CentralAir
data= data_train.loc[0:1459, ['SalePrice', 'CentralAir']]
fig1 = sns.boxplot(x='CentralAir', y="SalePrice", data=data)
plt.savefig('Feature_image_data/CentralAir.png')

#boxplot for OverallQual
data= data_train.loc[0:1459, ['SalePrice', 'OverallQual']]
fig2 = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
plt.savefig('Feature_image_data/OverallQual.png')

data= data_train.loc[0:1459, ['SalePrice', 'LotArea']]
data.plot.scatter(x='LotArea', y='SalePrice', )
plt.savefig('Feature_image_data/LotArea.png')

data_train['Utilities']=1
data= data_train.loc[0:1459, ['SalePrice', 'Utilities']]
data.plot.scatter(x='Utilities', y='SalePrice')
plt.savefig('Feature_image_data/Utilities.png')

# boxplot for Neighborhood
data= data_train.loc[0:1459, ['SalePrice', 'Neighborhood']]
f, ax = plt.subplots(figsize=(8, 6))
fig3 = sns.boxplot(x='Neighborhood', y="SalePrice", data=data)
fig3.set_xticklabels(ax.get_xticklabels(), rotation=-90)
plt.savefig('Feature_image_data/Neighborhood.png')

# boxplot for overallQual
data= data_train.loc[0:1459, ['SalePrice', 'OverallQual']]
fig4 = sns.boxplot(x='OverallQual', y="SalePrice", data=data)
plt.savefig('Feature_image_data/OverallQual.png')

#boxplot for year built
data= data_train.loc[0:1459, ['SalePrice', 'YearBuilt']]
f, ax = plt.subplots(figsize=(28, 6))
fig5 = sns.boxplot(x='YearBuilt', y="SalePrice", data=data)
fig5.set_xticklabels(ax.get_xticklabels(), rotation=-90)
plt.savefig('Feature_image_data/boxPlot_YearBuilt.png')

# scatter plot for YearBuilt  
data= data_train.loc[0:1459, ['SalePrice', 'YearBuilt']]
data.plot.scatter(x='YearBuilt', y="SalePrice")
plt.savefig('Feature_image_data/scatterPlot_YearBuilt.png')


# scatter plot for TotalBsmtSF  

data= data_train.loc[0:1459, ['SalePrice', 'TotalBsmtSF']]
data.plot.scatter(x='TotalBsmtSF', y='SalePrice')
plt.savefig('Feature_image_data/TotalBsmtSF.png')

col  = ['GarageArea', 'GarageCars']
for index in range(2):
    data = pd.concat([data_train['SalePrice'], data_train[col[index]]], axis=1)
    data.plot.scatter(x=col[index], y='SalePrice')
plt.savefig('Feature_image_data/GarageArea_GarageCars.png')
# correleation matrix
corrmattrix = data_train.corr()
f, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corrmattrix, vmax=0.8, square=True, cmap="Blues")
plt.savefig('Feature_image_data/corrmattrix.png')
corrmattrix = data_train.corr()
# correleation matrix
k  = 10 
cols = corrmattrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, \
                 square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.savefig('Feature_image_data/corrmattrix_2.png')
