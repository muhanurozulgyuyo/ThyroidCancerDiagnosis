from preprocessing import load_data, train_prep, test_prep
from train_model import train
import joblib
import pandas as pd
import numpy as np

df_train = load_data('data/train.csv')
train(df_train)

df_test = load_data('data/test.csv')

X_test = test_prep(df_test)

model = joblib.load('models/rf_model.pkl')

y_predict = model.predict(X_test)

y_predict_series = pd.Series(y_predict, name='Cancer')

df_predict = pd.concat([df_test['ID'], y_predict_series], axis=1).reset_index(drop=True)

df_predict.to_csv('results/predicted.csv', index=False)