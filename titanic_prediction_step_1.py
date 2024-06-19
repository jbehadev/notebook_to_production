import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, [2,4,5,6,7,9,10,11]]
y = dataset.iloc[:, 1:2]

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Apply the fit_transform method on the instance of ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [1,6,7])], remainder='passthrough')
X_encoded = ct.fit_transform(X.values).toarray()

# placeholder if encoding is ever needed
y_encoded = y.values

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y_encoded, test_size = 0.2, random_state = 0)

from xgboost import XGBClassifier
classifier = XGBClassifier(n_estimators=500, learning_rate=0.01)
classifier.fit(X=X_train_encoded, y=y_train_encoded)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_test_encoded, y = y_test_encoded, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# need to change column indexes because of column index selector
dataset = pd.read_csv('test.csv')
X = dataset.iloc[:, [1,3,4,5,6,8,9,10]]
X_encoded = ct.transform(X.values).toarray()
pd.DataFrame(
    {
        'PassengerId': dataset.iloc[:, 0],
        'Survived': classifier.predict(X_encoded)
    }
).to_csv('submission.csv', index=False)

