

#step 1 : import library
import pandas as pd

#step 2 : import data
hv=pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/main/Hill%20Valley%20Dataset.csv')

hv.info()

hv.head()

hv.describe()

hv.columns

print(hv.columns.tolist())

hv.shape

hv['Class'].value_counts()

hv.groupby('Class').mean()

#step 3 : define y and x
y = hv['Class']
x = hv[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', ]]

x.shape

#step 4 : train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=2529)
x_train

x_train.shape, x_test.shape, y_train.shape, y_test.shape

import matplotlib.pyplot as plt

plt.plot(x.iloc[0,:])
plt.title('Valley')

plt.plot(x.iloc[1,:])
plt.title('Hill')

# step 5 : choose model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000)

#step 6 : fit the model
model.fit(x_train,y_train)

#step 7 : predict the model
y_pred = model.predict(x_test)

#step 8 : accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)