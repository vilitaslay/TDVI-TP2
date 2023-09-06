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
from datetime import datetime


random_state = 256
np.random.seed(random_state)

# Load the competition data
comp_data = pd.read_csv("C:/Users/44482978/Desktop/uni/td6/competition_data.csv")

# Pre procesamiento de datos
columns_to_drop=["etl_version","accepts_mercadopago","site_id","boosted","benefit",] 
comp_data.drop(columns=columns_to_drop, inplace=True) #eliminamos las columnas indicadas x kaggle y aquellas donde hay un único valor
comp_data.drop(columns="date", inplace=True) #eliminamos esta columna dado que contiene la misma informacion que "print_server_timestamp"
comp_data.drop(columns="uid",inplace=True)
comp_data.drop(columns="deal_print_id",inplace=True)
#comp_data.drop(columns="product_id",inplace=True)
comp_data.drop(columns="main_picture",inplace=True)

##Hacemos one hot encoding
encoder = OneHotEncoder(sparse=False)
one_hot_encoded = encoder.fit_transform(comp_data[['platform']])
one_hot_df = pd.DataFrame(one_hot_encoded)
comp_data = pd.concat([comp_data, one_hot_df], axis=1)
comp_data.drop(columns=['platform'], inplace=True)

#convertimos la columna de warranty
def map_warranty(value):
    if isinstance(value, str) and "Sin garantía" in value:
        return 0
    elif isinstance(value, str) and "12" in value:
        return 12
    elif isinstance(value, str) and "6" in value:
        return 6
    elif isinstance(value, str) and "30" in value:
        return 1
    elif isinstance(value, str) and "90" in value:
        return 3
    elif isinstance(value, str) and "1" in value:
        return 12
    elif isinstance(value, str) and "180" in value:
        return 6
    elif pd.notna(value):
        return 0
    else:
        return None

comp_data['warranty'] = comp_data['warranty'].apply(map_warranty)
pd.set_option('display.max_rows', None)


print("HOLA1")

#convertimos la fecha

comp_data['print_server_timestamp'] = pd.to_datetime(comp_data['print_server_timestamp'])

comp_data['month'] = comp_data['print_server_timestamp'].dt.month
comp_data['day'] = comp_data['print_server_timestamp'].dt.day
comp_data['hour'] = comp_data['print_server_timestamp'].dt.hour

comp_data.drop(columns=['print_server_timestamp'], inplace=True)


# ##convertimos la columna tags
# unique_categories = comp_data['tags'].unique()
# for i in range(len(unique_categories)):
#     unique_categories[i] = str(unique_categories[i])
#     unique_categories[i] = unique_categories[i][1:]
#     unique_categories[i] = unique_categories[i][:-1]
#     print(unique_categories[i])


# encoder2 = OneHotEncoder(sparse=False)
# one_hot_encoded2 = encoder.fit_transform(comp_data[['tags']])
# one_hot_df2 = pd.DataFrame(one_hot_encoded2)
# one_hot_df2.columns = unique_categories
# comp_data = pd.concat([comp_data, one_hot_df2], axis=1)
# comp_data.drop(columns=['tags'], inplace=True)
# print(comp_data.head())

# unique_elements  = set()
# for tag in comp_data['tags']:
#     tag.split(delimiter=',')
#     for element in tag:
#         print(element)
#         unique_elements.add(element)
       
    
# unique_elements = set(element for tag in comp_data['tags'] for element in tag)
# print(unique_elements)
# for element in unique_elements:
#     comp_data[element] = comp_data['tags'].apply(lambda x: 1 if element in x else 0)

# comp_data.drop(columns=['tags'], inplace=True)

# print(comp_data.columns)


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

print("HOLA2")


params = {'colsample_bytree': 0.7864670868346251,
               'gamma': 1.8475531774069638,
               'learning_rate': 0.06630474845427192,
               'max_depth': 8,
               'min_child_weight': 3.811309774095629,
               'n_estimators': 895,
               'reg_lambda': 3.5820673752824694,
               'subsample':  0.80412605586238,
               }

print("HOLA3")

#búsqueda de parámetros

# params = {'max_depth': list(range(1, 30)),
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


xgModel = xgb.XGBClassifier(objective = "binary:logistic", seed = random_state, eval_metric = "auc", **params)
xgModel.fit(X_train, y_train)

print("HOLA4")


y_pred = xgModel.predict_proba(X_val)[:, xgModel.classes_ == True]
print("ROC test score: ", roc_auc_score(y_val, y_pred))



# # Predict on the evaluation set
eval_data = eval_data.drop(columns=["conversion"])
eval_data = eval_data.select_dtypes(include='number')

y_preds_eval = xgModel.predict_proba(eval_data.drop(columns =["ROW_ID"]))[:, xgModel.classes_ == 1].squeeze()


#Make the submission file

submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds_eval})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("./data/basic_model_xgboost2.csv", sep=",", index=False)
