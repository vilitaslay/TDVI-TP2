import pandas as pd
import numpy as np
import gc
import sklearn
import time 
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import uniform
import xgboost as xgb


random_state = 256
np.random.seed(random_state)

# Load the competition data
comp_data = pd.read_csv("./data/competition_data.csv")

# Split into training and evaluation samples
train_data = comp_data[comp_data["ROW_ID"].isna()]

eval_data = comp_data[comp_data["ROW_ID"].notna()]
del comp_data
gc.collect()

#Definimos X e y
y = train_data["conversion"]
X = train_data.drop(columns=["conversion", "ROW_ID"]) 
X = X.select_dtypes(include='number')

# Train a random forest model on the train data
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  train_size = 0.7,
                                                  random_state = random_state,
                                                  stratify = y)
del train_data
gc.collect()

params = {'colsample_bytree': 0.75,
               'gamma': 0.5,
               'learning_rate': 0.065,
               'max_depth': 7,
               'min_child_weight': 3,
               'n_estimators': 1500,
               'reg_lambda': 0.5,
               'subsample': 0.80,
               }


xgModel = xgb.XGBClassifier(objective = "binary:logistic", seed = random_state, eval_metric = "auc", **params)
xgModel.fit(X_train, y_train)

#vemos la importancia de las variables
bst = xgModel.get_booster()

weight = bst.get_score(importance_type='weight')
gain = bst.get_score(importance_type='gain')

sorted_weight = sorted(weight.items(), key = lambda x: x[1])
sorted_gain = sorted(gain.items(), key = lambda x: x[1])

comp_data = pd.read_csv("./data/competition_data.csv")

variables_importantes = []
for i in sorted_weight:
    variables_importantes.append(i[0])

variables_importantes.append("ROW_ID")
variables_importantes.append("conversion")
comp_data = comp_data[variables_importantes]

# Split into training and evaluation samples
train_data = comp_data[comp_data["ROW_ID"].isna()]

eval_data = comp_data[comp_data["ROW_ID"].notna()]
del comp_data
gc.collect()
#Definimos X e y
y = train_data["conversion"]
X = train_data.drop(columns=["conversion", "ROW_ID"]) 
X = X.select_dtypes(include='number')

# Train a random forest model on the train data
X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  train_size = 0.7,
                                                  random_state = random_state,
                                                  stratify = y)
del train_data
gc.collect()

xgModel = xgb.XGBClassifier(objective = "binary:logistic", seed = random_state, eval_metric = "auc", **params)
xgModel.fit(X_train, y_train)

y_pred = xgModel.predict_proba(X_val)[:, xgModel.classes_ == True]
print(y_pred.shape)
print("ROC test score: ", roc_auc_score(y_val, y_pred))


#búsqueda de parámetros

# params = {'max_depth': list(range(1, 40)),
#           'learning_rate': uniform(scale = 0.2),
#           'gamma': uniform(scale = 2),
#           'reg_lambda': uniform(scale = 5),        # Parámetro de regularización.
#           'subsample': uniform(0.5, 0.5),          # Entre 0.5 y 1.
#           'min_child_weight': uniform(scale = 5),
#           'colsample_bytree': uniform(0.75, 0.25), # Entre 0.75 y 1.
#           'n_estimators': list(range(1, 1000))
#          }

# start = time.time()
# best_score = 0
# best_estimator = None
# iterations = 100
# for g in ParameterSampler(params, n_iter = iterations, random_state = random_state):
#     clf_xgb = xgb.XGBClassifier(objective = 'binary:logistic', seed = random_state, eval_metric = 'auc', **g)
#     clf_xgb.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = False)

#     y_pred = clf_xgb.predict_proba(X_val)[:, 1] # Obtenemos la probabilidad de una de las clases (cualquiera).
#     auc_roc = sklearn.metrics.roc_auc_score(y_val, y_pred)
#     # Guardamos si es mejor.
#     if auc_roc > best_score:
#         print(f'Mejor valor de ROC-AUC encontrado: {auc_roc}')
#         best_score = auc_roc
#         best_grid = g
#         best_estimator = clf_xgb

# end = time.time()
# print('ROC-AUC: %0.5f' % best_score)
# print('Grilla:', best_grid)


# xgModel = xgb.XGBClassifier(objective = "binary:logistic", seed = random_state, eval_metric = "auc", **best_grid)
# xgModel.fit(X_train, y_train)

# # Predict on the evaluation set
eval_data = eval_data.drop(columns=["conversion"])
eval_data = eval_data.select_dtypes(include='number')

y_preds_eval = xgModel.predict_proba(eval_data.drop(columns =["ROW_ID"]))[:, xgModel.classes_ == 1].squeeze()


# Make the submission file
submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds_eval})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("./data/basic_model_xgboost2.csv", sep=",", index=False)
