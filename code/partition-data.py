import os
import pandas as pd

path_wd = os.getenv("PATH_WD")
path_data = os.path.join(path_wd, "data")
path_adult = os.path.join(path_data, "adult.csv")

pd_adult = pd.read_csv(path_adult, header=None, index_col=False,
    names=['age', 'workclass', 'fnlwgt', 'education',  'education-num',
           'marital-status', 'occupation', 'relationship', 'race', 'gender',
           'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
           'income'])

from sklearn.model_selection import train_test_split

pd_features = pd_adult.drop("income", axis=1)
pd_target = pd_adult[["income"]]

dict_data = {}

dict_data["adult_X_train"], dict_data["adult_X_test"], dict_data["adult_y_train"], dict_data["adult_y_test"] = train_test_split(
    pd_features, pd_target, random_state=0)

for key in dict_data.keys():
    pd_data = dict_data[key]
    pd_data.to_csv(os.path.join(path_data, str(key)+".csv"), index=False)
