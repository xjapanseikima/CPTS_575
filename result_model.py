from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

raw_data = pd.read_csv("houseprice/train.csv",
                         usecols=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd',
                                  'YearBuilt', 'SalePrice'])


x = raw_data.iloc[:, 0:6]

y = raw_data['SalePrice'].values
x_1 = preprocessing.StandardScaler()
x_1 = x_1.fit_transform(x)

a = raw_data.iloc[:, 7]
# print(a)
a = y.reshape(-1, 1)

b = preprocessing.StandardScaler()
b = b.fit_transform(a)
a, b, c, d = train_test_split(x_1, b, test_size=0.33, random_state=42)
clfs = {
    'svm': svm.SVR(),
    'RandomForestRegressor': RandomForestRegressor(n_estimators=400),
    'BayesianRidge': linear_model.BayesianRidge()
}
for clf in clfs:
    clfs[clf].fit(a, c)
    y_pred = clfs[clf].predict(b)
    print(clf + " cost:" + str(np.sum(y_pred - d) / len(y_pred)))



# print(type(raw_data))



x_1, x_2, y_1, y_2 = train_test_split(raw_data.iloc[:, 0:7], raw_data.iloc[:,7].values, test_size=0.25, random_state=42)

clf = RandomForestRegressor(n_estimators=500)
clf.fit(x_1, y_1)
y_pred = clf.predict(x_2)

print(sum(abs(y_pred - y_2)) / len(y_pred))

predictValue = (abs(y_pred)) / (y_2)
print(predictValue)
count = 0;
print("-----------------------------")
print("standard")

for i in range(len(predictValue)):
    if predictValue[i] < 1.2 and predictValue[i] > 0.8:
        count = count + 1;
print(count / len(predictValue))

rfr = clf
data_test_1 = pd.read_csv("houseprice/test.csv",
                        usecols=['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd',
                                 'YearBuilt', 'Id'])
data_test_1.iloc[:, 0:6].isnull().sum()
print(data_test_1)
average_gc = data_test_1.iloc[:,7].sum() / len(data_test_1)

cars = data_test_1.iloc[:,7].fillna(1.766118)

average_tb = data_test_1.iloc[:,3].sum() / len(data_test_1)
bsmt = data_test_1.iloc[:,3].fillna(1046.117970)

data_test_2 = pd.concat([data_test_1[['OverallQual', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']], cars, bsmt],
                        axis=1)
data_test_2.isnull().sum()
x = data_test_2.values
y_te_pred = rfr.predict(x)
print(y_te_pred)
print(y_te_pred.shape)
print(x.shape)
prediction = pd.DataFrame(y_te_pred, columns=['SalePrice'])
result = pd.concat([data_test_1.iloc[:,0], prediction], axis=1)
# result = result.drop(resultlt.columns[0], 1)
result.columns
result.to_csv('houseprice/result_predict.csv', index=False)
