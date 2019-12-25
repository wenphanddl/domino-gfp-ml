import os
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

path_wd = os.getenv("PATH_WD")
path_data = os.path.join(path_wd, "data")
path_artifacts = os.path.join(path_wd, "artifacts")

pd_adult_X_train = pd.read_csv(os.path.join(path_data, "adult_X_test.csv"))

ct = ColumnTransformer([("scaling", StandardScaler(), ['age', 'hours-per-week']),
     ("onehot", OneHotEncoder(sparse=False),
     ['workclass', 'education', 'gender', 'occupation'])])

ct.fit(pd_adult_X_train)

with open(os.path.join(path_artifacts, "coltransformer.pkl"), "wb") as f:
    pickle.dump(ct, f)
    f.close()
