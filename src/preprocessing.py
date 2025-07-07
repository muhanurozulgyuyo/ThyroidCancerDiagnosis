import pandas as pd
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def train_prep(df):
    target = df['Cancer']
    input = df.drop("Cancer", axis=1).copy()

    # 나이 라벨링
    bins = [0, 19, 39, 59, 79, 100]
    labels = ['0-19', '20-30', '40-50', '60-70', '80-']

    input['age_group'] = pd.cut(input['Age'], bins=bins, labels=labels)

    # one-hot encoding
    onehot_cols = ['Gender', 'Country', 'Race', 'Family_Background', 'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk', 'Diabetes', 'age_group']

    input = pd.get_dummies(input, columns=onehot_cols, drop_first=False)

    input.drop(['Age', 'ID'], axis=1, inplace=True)

    return input, target

def test_prep(df):
    input = df.copy()
    
    # 나이 라벨링
    bins = [0, 19, 39, 59, 79, 100]
    labels = ['0-19', '20-30', '40-50', '60-70', '80-']

    input['age_group'] = pd.cut(input['Age'], bins=bins, labels=labels)

    # one-hot encoding
    onehot_cols = ['Gender', 'Country', 'Race', 'Family_Background', 'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Weight_Risk', 'Diabetes', 'age_group']

    input = pd.get_dummies(input, columns=onehot_cols, drop_first=False)

    input.drop(['Age', 'ID'], axis=1, inplace=True)

    return input