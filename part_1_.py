import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)
creditcard_df = pd.read_csv('UCI_Credit_Card.csv')
# print(creditcard_df.head)
# print(creditcard_df.info())
# print(creditcard_df.describe())
# print(creditcard_df.isnull())
sns.heatmap(creditcard_df.isnull(), yticklabels=False, cbar=False, cmap='Blues')
creditcard_df.hist(bins=25, figsize=(50, 50), color='r')
plt.subplots_adjust(wspace=1)
# plt.show()
###########################################################
creditcard_df.drop(['ID'], axis=1, inplace=True)
cc_defult_df = creditcard_df[creditcard_df['default.payment.next.month'] == 1]
cc_nondefult_df = creditcard_df[creditcard_df['default.payment.next.month'] == 0]
a = cc_defult_df['default.payment.next.month'].value_counts()
b = cc_nondefult_df['default.payment.next.month'].value_counts()
c = a.sum() + b.sum()
# print(c)
# print(a,b)
# print(a/c)
# print(b/c)

# Count the number of employees who stayed and left
# It seems that we are dealing with an imbalanced dataset
#
# print("Total =", len(creditcard_df))
#
# print("Number of customers who defaulted on their credit card payments =", len(cc_default_df))
# print("Percentage of customers who defaulted on their credit card payments =",
#       1. * len(cc_default_df) / len(creditcard_df) * 100.0, "%")
#
# print("Number of customers who did not default on their credit card payments (paid their balance)=",
#       len(cc_nodefault_df))
# print("Percentage of customers who did not default on their credit card payments (paid their balance)=",
#       1. * len(cc_nodefault_df) / len(creditcard_df) * 100.0, "%")


# print(cc_defult_df.describe())
# print(cc_nondefult_df.describe())
######################################################################

carolation = creditcard_df.corr()
# f, ax = plt.subplots(figsize=(20, 20))
# sns.heatmap(carolation, annot=True)
# plt.figure(figsize=(25, 12))
# sns.countplot(x='AGE', hue='default.payment.next.month', data=creditcard_df)
# plt.figure(figsize=(20, 20))
# # plt.subplots(311)  # 3 rows and 1 column and first figure
# sns.countplot(x='EDUCATION', hue='default.payment.next.month', data=creditcard_df)
# plt.figure(figsize=(20, 20))
# # plt.subplots(312)  # 3 rows and 1 column and 2 figure
# sns.countplot(x='SEX', hue='default.payment.next.month', data=creditcard_df)
# plt.figure(figsize=(20, 20))
# # plt.subplots(313)  # 3 rows and 1 column and 3 figure
# sns.countplot(x='MARRIAGE', hue='default.paym ent.next.month', data=creditcard_df)

# plt.figure(figsize=(12,7))
# sns.distplot(cc_nondefult_df['LIMIT_BAL'],bins=250,color='r')
# sns.distplot(cc_defult_df['LIMIT_BAL'],bins=250,color='b')
# plt.xlabel('Amount of bill statement in September, 2005 (NT dollar)')

#
# plt.figure(figsize=(12,7))
#
# sns.kdeplot(cc_nodefault_df['BILL_AMT1'], label = 'Customers who did not default (paid balance)', shade = True, color = 'r')
# sns.kdeplot(cc_default_df['BILL_AMT1'], label = 'Customers who defaulted (did not pay balance)', shade = True, color = 'b')
#
# plt.xlabel('Amount of bill statement in September, 2005 (NT dollar)')
#plt.xlim(0, 200000)

# plt.figure(figsize=(12,7))
#
# sns.kdeplot(cc_nodefault_df['PAY_AMT1'], label = 'Customers who did not default (paid balance)', shade = True, color = 'r')
# sns.kdeplot(cc_default_df['PAY_AMT1'], label = 'Customers who defaulted (did not pay balance)', shade = True, color = 'b')
#
# plt.xlabel('PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)')
# plt.xlim(0, 200000)
###########

plt.figure(figsize=[10,20])
# plt.subplot(211)
# sns.boxplot(x = 'SEX', y = 'LIMIT_BAL', data = creditcard_df, showfliers = False)
# plt.subplot(212)
# sns.boxplot(x = 'SEX', y = 'LIMIT_BAL', data = creditcard_df)
##############
plt.figure(figsize=[10,20])
# plt.subplot(211)
# sns.boxplot(x = 'MARRIAGE', y = 'LIMIT_BAL', data = creditcard_df, showfliers = False)
# # plt.subplot(212)
# sns.boxplot(x = 'MARRIAGE', y = 'LIMIT_BAL', data = creditcard_df)


# plt.show()



x_cat = creditcard_df[['SEX','EDUCATION','MARRIAGE']]
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder()
x_cat = onehotencoder.fit_transform(x_cat).toarray()
# print(x_cat.shape)
x_cat = pd.DataFrame(x_cat)
# print(x_cat)

# note that we dropped the target 'default.payment.next.month'
X_numerical = creditcard_df[['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

x_all = pd.concat([x_cat,X_numerical],axis=1)
# print(x_all)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x = scaler.fit_transform(x_all)
# print(x)
y = creditcard_df['default.payment.next.month']
# print(y)


from sklearn.model_selection import train_test_split

X_trina,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
print(X_trina.shape,X_test.shape)

import xgboost as xgb

model = xgb.XGBClassifier(objective='reg:squarederror',learning_rate=0.1,max_depth=20,n_estimators=500)
model.fit(X_trina,y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
# print(y_pred)

from sklearn.metrics import classification_report,confusion_matrix

# print('Accuracy{} %'.format(100*accuracy_score(y_pred,y_test)))

cm =confusion_matrix(y_pred,y_test)
sns.heatmap(cm,annot=True)
plt.show()

# print(classification_report(y_pred,y_test))

pram_grid =  {
    'gamma':[0.5,1,5],
    'subsample':[0.6,0,8,1],
    'colsample_bytree':[0.6,0.8,1],
    'max_depth':[3,4,5]
}

print('saad')
from xgboost import XGBClassifier

xgb_model = XGBClassifier(learning_rate=0.01,n_estimators=100,objective='binary:logistic')
from sklearn.model_selection import GridSearchCV

grid =GridSearchCV(xgb_model,pram_grid,refit=True,verbose=4)
# grid.fit(X_trina,y_train)

y_preidct=grid.predict(X_test)
print(y_preidct)
cm =confusion_matrix(y_preidct,y_test)
sns.heatmap(cm,annot=True)
# plt.show()

train_data = pd.DataFrame({'Target':y_train})
for i in range(X_trina.shape[1]):
    train_data[i] = X_trina[:,i]

val_data = pd.DataFrame({'Target':y_test})
for i in range(X_test.shape[1]):
    val_data[i] = X_test[:,i]

train_data.to_csv('train.csv',header=False,index=False)
val_data.to_csv('validation.csv',header=False,index=False)

import sagemaker
import boto3
# from sagemaker.amazon.amazon_estimator import get_image_urinp
sagemaker_session = sagemaker.Session()

bucket = 'modern_ai_course'
prefix = 'XGBoost-classifier'
key = 'XGBoost-classifier'

role = sagemaker.get_execution_role()
print(role)


























