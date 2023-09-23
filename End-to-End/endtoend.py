from sklearn.utils.extmath import squared_norm
from pandas.io.formats.style import format_table_styles
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.fixes import sklearn
from importlib.abc import InspectLoader
import os
import tarfile
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy import stats
import joblib


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
housing.head()

print(housing)
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set))
print(len(test_set))

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing.plot(kind="scatter", x="longitude", y="latitude")
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True,
)
plt.legend()

corr_matrix = housing.corr()
print(corr_matrix)
corr_matrix["median_house_value"].sort_values(ascending=False)

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
##print(corr_matrix)
corr_matrix["median_house_value"].sort_values(ascending=False)

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels= strat_train_set["median_house_value"].copy()
print(type(housing))
print(len(housing))
print(housing)

print(type(housing_labels))
print(len(housing_labels))
print(housing_labels)

"""housing.dropna(subset=["total_bedrooms"])
housing.drop("total_bedrooms",axis=1)
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)"""

imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)



X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
print("-------------------------------------------------------------------------------")
print(housing_tr)

housing_cat = housing[["ocean_proximity"]]
print("-------------------------------------------------------------------------------")
print(type(housing_cat.head(10)))
print(housing_cat.head(10))
print("-------------------------------------------------------------------------------")
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print("-------------------------------------------------------------------------------")
print(housing_cat_encoded)
print(type(housing_cat_encoded))
print("-------------------------------------------------------------------------------")
print(housing_cat_encoded[:10])
print("-------------------------------------------------------------------------------")
print(ordinal_encoder.categories_)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(type(housing_cat_1hot))

housing_cat_1hot.toarray()

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
  def __init__(self, add_bedrooms_per_room = True):
    self.add_bedrooms_per_room = add_bedrooms_per_room
  def fit(self, X, y=None):
    return self #nothing else to do
  def transform(self, X):
    rooms_per_household = X[:,rooms_ix] / X[:, households_ix]
    population_per_household = X[:, population_ix] / X[:, households_ix]
    if self.add_bedrooms_per_room:
      bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
      return np.c_[X, rooms_per_household, population_per_household,
                  bedrooms_per_room]
    else:
      return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
print("-------------------------------------------------------------------------------")
print(type(housing_extra_attribs))
print(len(housing_extra_attribs))
print(housing_extra_attribs)
print("-------------------------------------------------------------------------------")

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attrib_adder', CombinedAttributesAdder()),
    ('std_scale', StandardScaler()),
])
housing_num_tr = num_pipeline.fit_transform(housing_num)

print("-------------------------------------------------------------------------------")
print(type(housing_num_tr))
print(len(housing_num_tr))
print(housing_num_tr.shape)
print(housing_num_tr)
print("-------------------------------------------------------------------------------")

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
print("-------------------------------------------------------------------------------")
print(type(num_attribs))
print(len(num_attribs))
print(num_attribs)
print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")
print(type(cat_attribs))
print(len(cat_attribs))
print(cat_attribs)
print("-------------------------------------------------------------------------------")

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)
print("-------------------------------------------------------------------------------")
print(type(housing_prepared))
print(len(housing_prepared))
print(housing_prepared.shape)
print(housing_prepared)
print("-------------------------------------------------------------------------------")

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Labels: ", list (some_labels))
print("-------------------------------------------------------------------------------")

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)
print("-------------------------------------------------------------------------------")

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)
print("-------------------------------------------------------------------------------")

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
print(tree_rmse_scores)
print("-------------------------------------------------------------------------------")

def display_scores(scores):
  print("Scores: ", scores)
  print("Mean: ", scores.mean())
  print("Standard deviation: ", scores.std())

  print(display_scores)
  print("-------------------------------------------------------------------------------")

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring = "neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
print(display_scores(lin_rmse_scores))
print("-------------------------------------------------------------------------------")

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print(type(grid_search))
print(grid_search)
print("-------------------------------------------------------------------------------")

print(grid_search.best_estimator_)
print("-------------------------------------------------------------------------------")

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
  print(np.sqrt(mean_score),params)

final_model = grid_search.best_estimator_

x_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)
final_predictions = final_model.predict(x_test_prepared)

final_model2 = RandomForestRegressor(n_estimators = 30, max_features = 6)
final_model2.fit(housing_prepared, housing_labels)
final_predictions2 = final_model2.predict(x_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

final_mse2 = mean_squared_error(y_test, final_predictions2)
final_rmse2 = np.sqrt(final_mse2)

print("-------------------------------------------------------------------------------")
print(final_mse)
print(final_rmse)
print("-------------------------------------------------------------------------------")
print(final_mse2)
print(final_rmse2)
print("-------------------------------------------------------------------------------")

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))

                                              #LinearRegression
final_model2_1 = LinearRegression()
final_model2_1.fit(housing_prepared, housing_labels)
final_predictions2_1 = final_model2_1.predict(x_test_prepared)

final_mse2_1 = mean_squared_error(y_test, final_predictions2_1)
final_rmse2_1 = np.sqrt(final_mse2_1)
print(final_rmse2_1)
print("-------------------------------------------------------------------------------")

confidence = 0.95
squared_errors2 = (final_predictions2_1 - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors2) - 1,
                                               loc=squared_errors2.mean(),
                                               scale=stats.sem(squared_errors2)))

                                              #DecisionTreeRegressor

final_model2_2 = DecisionTreeRegressor(random_state=42)
final_model2_2.fit(housing_prepared, housing_labels)
final_predictions2_2 = final_model2_2.predict(x_test_prepared)

final_mse2_2 = mean_squared_error(y_test, final_predictions2_2)
final_rmse2_2 = np.sqrt(final_mse2_2)
print(final_rmse2)
print("-------------------------------------------------------------------------------")


confidence = 0.95
squared_errors_2 = (final_predictions2_2 - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors_2) - 1,
                                               loc=squared_errors_2.mean(),
                                               scale=stats.sem(squared_errors_2)))

final_model = grid_search
joblib.dump(final_model, "my_model.pkl")
""" print("-------------------------------------------------------------------------------")
print(final_model)
print(type(final_model))
print("-------------------------------------------------------------------------------")
print(final_model2)
print(type(final_model2))
print("-------------------------------------------------------------------------------")
print(final_model2_1)
print(type(final_model2_1))
print("-------------------------------------------------------------------------------")
print(final_model2_2)
print(type(final_model2_2))"""