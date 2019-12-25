import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

path_wd = os.getenv("PATH_WD")
path_data = os.path.join(path_wd, "data")
path_artifacts = os.path.join(path_wd, "artifacts")

list_data = ["adult_X_train", "adult_X_test", "adult_y_train", "adult_y_test"]
dict_data = {}

for item in list_data:
    dict_data[item] = pd.read_csv(os.path.join(path_data, item+".csv"))

with open(os.path.join(path_artifacts, "coltransformer.pkl"), "rb") as f:
    ct = pickle.load(f)
    f.close()

pd_adult_X_train_trans = ct.transform(dict_data["adult_X_train"])

model_logreg = LogisticRegression()
model_logreg.fit(pd_adult_X_train_trans, dict_data["adult_y_train"])

pd_adult_X_test_trans = ct.transform(dict_data["adult_X_test"])
score = model_logreg.score(pd_adult_X_test_trans, dict_data["adult_y_test"])
print("Score: " + str(score))

with open(os.path.join(path_artifacts, "model_logreg.pkl"), "wb") as f:
    pickle.dump(model_logreg, f)
    f.close()
