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
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import math



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

# Create an imputer for numeric features (assuming they are in X_train)

numeric_imputer = SimpleImputer(strategy="mean")
comp_data_numeric = comp_data.select_dtypes(include='number').drop(columns=["ROW_ID"])
comp_data.loc[:, comp_data_numeric.columns] = numeric_imputer.fit_transform(comp_data_numeric)

# Create an imputer for categorical features (excluding "conversion")
categorical_imputer = SimpleImputer(strategy="most_frequent")
comp_data_categorical = comp_data.select_dtypes(exclude='number')
comp_data.loc[:, comp_data_categorical.columns] = categorical_imputer.fit_transform(comp_data_categorical)


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


# Entrenamiento y evaluación del modelo Bagging
base_model = DecisionTreeClassifier()
bag = BaggingClassifier(base_model, n_estimators=500, n_jobs=-1, random_state=random_state, verbose=1)
bag.fit(pd.concat([X_train, X_val], axis=0),
        pd.concat([y_train, y_val], axis=0))
preds_test_bag = bag.predict_proba(X_val)[:, bag.classes_ == True]
print("ROC AUC Score - Bagging:", roc_auc_score(y_val, preds_test_bag)) # 0.9617250236919546
print("HOLA3")

# Entrenamiento y evaluación del modelo Random Forest
rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=random_state, verbose=1, oob_score=True)
rf.fit(pd.concat([X_train, X_val], axis=0),
       pd.concat([y_train, y_val], axis=0))
preds_test_rf = rf.predict_proba(X_val)[:, rf.classes_ == True]
print("ROC AUC Score - Random Forest:", roc_auc_score(y_val, preds_test_rf)) # 0.9506521522270438
print("HOLA4")

# Performance oob
preds_oob_rf = rf.oob_decision_function_[:, rf.classes_ == True]
print("OOB ROC AUC Score - Random Forest:", roc_auc_score(pd.concat([y_train, y_val]), preds_oob_rf)) # 0.916999750777964

# Importancia de atributos con random forest
# def plot_importance(model, n_vars):
#     # Sort the DataFrame by 'Importance' column in descending order
#     imp_df = pd.DataFrame({"Variable": model.feature_names_in_, "Importance": model.feature_importances_})
#     imp_df = imp_df.sort_values(by='Importance', ascending=False)

#     # Take only the top 10 rows
#     top_imp_df = imp_df.head(n_vars).copy()

#     # Scale the importance values to have the max as 100
#     max_importance = top_imp_df['Importance'].max()
#     top_imp_df['Scaled_Importance'] = (top_imp_df['Importance'] / max_importance) * 100

#     # Create the horizontal bar plot
#     plt.figure(figsize=(10, 6))
#     plt.barh(top_imp_df['Variable'], top_imp_df['Scaled_Importance'], color='skyblue')
#     plt.xlabel('Scaled Importance (Max = 100)')
#     plt.ylabel('Variable')
#     plt.title('Top 10 Feature Importance')
#     plt.gca().invert_yaxis()
#     plt.show()

# plot_importance(rf, 10)

# # Predict on the evaluation set
eval_data = eval_data.drop(columns=["conversion"])
eval_data = eval_data.select_dtypes(include='number')

y_preds_eval = rf.predict_proba(eval_data.drop(columns =["ROW_ID"]))[:, rf.classes_ == 1].squeeze()

print("HOLA5")

#Make the submission file

submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds_eval})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("./data/basic_model_forest.csv", sep=",", index=False)
