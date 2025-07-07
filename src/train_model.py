from preprocessing import load_data, train_prep
from sklearn.ensemble import RandomForestClassifier
import joblib

def train(df):
    X_train, y_train = train_prep(df)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'models/rf_model.pkl')