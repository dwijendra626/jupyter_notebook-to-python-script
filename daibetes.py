from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# Split the dataset into train and test set
def split_data(df):
    X = df.drop('Y', axis=1).values
    y = df['Y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    
    return data

# Train model and return the model
def train_model(data,args):
    reg_model = Ridge(**args)
    reg_model.fit(data["train"]["X"], data["train"]["y"])

    return reg_model

# Evaluate model metrics
def get_model_metrics(reg_model,data):
    preds = reg_model.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    metrics = {"mse": mse}

    return metrics

def main():

    # Load data
    sample_data = load_diabetes()
    df = pd.DataFrame(data=sample_data.data, columns=sample_data.feature_names)
    df['Y'] = sample_data.target

    # call the above functions
    data = split_data(df)
    
    args = args = {
            "alpha": 0.5
            }
    reg_model = train_model(data,args)

    metrics = get_model_metrics(reg_model,data)

    model_name = "sklearn_regression_model.pkl"

    joblib.dump(value=reg_model, filename=model_name)

if __name__ == '__main__':
    main()

