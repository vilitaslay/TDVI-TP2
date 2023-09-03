import pandas as pd
import numpy as np
import gc
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
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


params = {'colsample_bytree': 0.75,
               'gamma': 0.5,
               'learning_rate': 0.075,
               'max_depth': 8,
               'min_child_weight': 1,
               'n_estimators': 1200,
               'reg_lambda': 0.5,
               'subsample': 0.75,
               }
del train_data
gc.collect()

xgModel = xgb.XGBClassifier(objective = "binary:logistic", seed = random_state, eval_metric = "auc", **params)
xgModel.fit(X_train, y_train)

y_pred = xgModel.predict_proba(X_val)[:, xgModel.classes_ == True]
print(y_pred.shape)
print("ROC test score: ", roc_auc_score(y_val, y_pred))

# pipeline = make_pipeline(OneHotEncoder(handle_unknown ='ignore', sparse = False), xgb.XGBClassifier(objective = "binary:logistic", seed = random_state, eval_metric = "auc"))
# pipeline.fit(X_train, y_train)



# # Predict on the evaluation set
eval_data = eval_data.drop(columns=["conversion"])
eval_data = eval_data.select_dtypes(include='number')

y_preds_eval = xgModel.predict_proba(eval_data.drop(columns =["ROW_ID"]))[:, xgModel.classes_ == 1].squeeze()


# # Make the submission file
submission_df = pd.DataFrame({"ROW_ID": eval_data["ROW_ID"], "conversion": y_preds_eval})
submission_df["ROW_ID"] = submission_df["ROW_ID"].astype(int)
submission_df.to_csv("./data/basic_model_xgboost.csv", sep=",", index=False)
