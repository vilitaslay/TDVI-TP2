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


params = {'colsample_bytree': 0.75,
               'gamma': 0.5,
               'learning_rate': 0.065,
               'max_depth': 7,
               'min_child_weight': 3,
               'n_estimators': 1500,
               'reg_lambda': 0.5,
               'subsample': 0.80,
               }

print("HOLA3")


xgModel = xgb.XGBClassifier(objective = "binary:logistic", seed = random_state, eval_metric = "auc", **params)
xgModel.fit(X_train, y_train)

print("HOLA4")


y_pred = xgModel.predict_proba(X_val)[:, xgModel.classes_ == True]
print(y_pred.shape)
print("ROC test score: ", roc_auc_score(y_val, y_pred))



# # Predict on the evaluation set
eval_data = eval_data.drop(columns=["conversion"])
eval_data = eval_data.select_dtypes(include='number')

y_preds_eval = xgModel.predict_proba(eval_data.drop(columns =["ROW_ID"]))[:, xgModel.classes_ == 1].squeeze()


#Make the submission file

submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds_eval})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("./data/basic_model_xgboost2.csv", sep=",", index=False)
