import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
BMI 550: Applied BioNLP
Assignment 2: PREDICTING FREEZING OF GAIT FROM FALL REPORTS

SUMMARY:
This script accepts a single .xlsx file, specifically the "TEXT" column, and generates:
--> For each row, [CUI +/- negation] predicted labels for each phrase from within the "TEXT" column.


OUTPUT FORMAT: 
the system will output a text file that contains, tab separated, the id of a post, 
    the symptom expression (or entire negated symptom expression), the CUI for the symptom, 
    and a flag indicating negation (i.e., 1 if the symptom expression is a negated one, 0 otherwise).

    Future look: look into more advanced feature selection techniques (see DLATK: https://dlatk.wwbp.org/install.html)

@author Chase Fensore
email: chase.fensore@emory.edu

'''

import pandas as pd
import numpy as np
from collections import defaultdict
import re
import nltk
# from nltk.tokenize import sent_tokenize
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import argparse
from datetime import datetime
import json
from pymongo import MongoClient
import sklearn.ensemble as sken
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection as skms
import sklearn.neighbors as skknn
import sklearn.neural_network as sknn
from sklearn.naive_bayes import GaussianNB
import sklearn.metrics as skm
import sklearn.tree as sktree
import tqdm
import urllib.parse
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
# from sklearn.inspection import permutation_importance # For feature importance
import shap
import os

from evalHelper import read_json, evaluate_results, get_train_test


os.chdir('/Users/chasefensore/BMI550-NLP/2_assignment')

MODEL_PARAMS = {
    'nb':{
        'model': GaussianNB(),
        'params': {}
    },
    "dt": {
        'model': sktree.DecisionTreeClassifier(min_samples_split=5,
                                               min_samples_leaf=5),
        'params': {'criterion': ['gini', 'entropy'],
                   'max_depth': range(5, 8)}
    },
    "knn": {
        'model': skknn.KNeighborsClassifier(),
        'params': {'n_neighbors': range(3,10)}
    },
    "mlp": {
        "model": sknn.MLPClassifier(),
        "params": {
            "hidden_layer_sizes": [(2, ), (2,2), (4, ), (4, 2)],
            "max_iter": [1000]
        }
    },
    "rf": {
         'model': sken.RandomForestClassifier(),
         'params': {'max_depth': [5,6,7,8],
                    'min_samples_leaf': [5, 10],
                    'n_estimators': [5, 10, 25]}
    },
   
    "xgboost": {
        "model": xgb.XGBClassifier(),
        "params": {"max_depth": [6,7,8],
                   "n_estimators": [10, 20, 30, 40],
                   "learning_rate":[0.01, 0.1],
                   "eval_metric":["logloss"]}
    }
}

# TODO: try bin classification w/ LogisticRegression + GPT-2 transformer: https://www.kaggle.com/code/unofficialmerve/scikit-learn-with-transformers/notebook




def json_extract_values(obj):
    """
        Returns a Python list of all values from a json object, discarding the keys.
    """
    if isinstance(obj, dict):
        values = []
        for key, value in obj.items():
            values.extend(json_extract_values(value))
        return values
    elif isinstance(obj, list):
        return obj
    else:
        return []
    


    # TODO: change ACC script to do "ALL" properly. Then cp & paste here (eval on sgs: ALL, female, male, MDP-UPDRS stage...)



def main():
    parser = argparse.ArgumentParser()
    # mongo information
    username = urllib.parse.quote_plus('fensorechase')
    password = urllib.parse.quote_plus('7pzNiMi7dD!d@Ab')
    parser.add_argument("-mongo_url", default = 'mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.10.5') 
    parser.add_argument("-mongo_db",
                        default="aequitas")
    parser.add_argument("-mongo_col",
                        default="A-BioNLP",
                        help="collection_type") # Used to be subgroups_baseline
    # default information
    # Smaller sample, removed NAs: clean_tract_v2_ahrq.csv: includes select built env, mobility, air quality AHRQ data
    # AHRQ baseline year, removed NAs: clean_tract_v2_ahrq_baseline.csv
    # AHRQ all 30k samples, mean from 2009-2020: clean_tract_v2_ahrq_allsamps.csv
    # AHRQ all 30k samples, baseline year, imputation: clean_tract_v2_ahrq_all_samps_baseline.csv
    # Old input: final_clean_tract_ahrq_AQIimp1_baseline.csv
    parser.add_argument("-train_file", 
                        default="./data/train_feats.csv", 
                        help="data file") 
    parser.add_argument("-test_file", 
                        default="./data/test_feats.csv", 
                        help="data file") 
    
    parser.add_argument("-base_feat",
                        default="./data/feat_base.json",
                        help="base_features")                   
    parser.add_argument("-feat_file", 
                        default="./data/feat_column.json",
                       help="model_features")
    parser.add_argument("-subgroup_file", 
                    default="./data/FAST_subgroup_cols.json",
                    help="subgroups_to_test") # Only gender subgroup testing conducted here.
    parser.add_argument("-endpoint",
                        default="fog_q_class")

    parser.add_argument("--feats", nargs='+', default=[
       "demo",
       "clinical",
       "simple_lexical",
       "POS"

        ])
   
    args = parser.parse_args()

    # setup mongo
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]
    # raw_mcol = mdb["10_7_ACC_raw_agnostic"] # Used to be: "raw_baseline"
    # shap_mcol = mdb["9_20_shap_xgb_MLH"] # To save shap for XGBoost.

    df_train = pd.read_csv(args.train_file)
    df_test = pd.read_csv(args.test_file)

    base_feat = read_json(args.base_feat)
    feat_info = read_json(args.feat_file)
    subgroups_bins = read_json(args.subgroup_file)

    # Determine the feature sets
    # TEXT FEAT: "fall_description"
    # TARGET FEAT: "fog_q_class"
    # Generate feat sets from "fall_description"

    # GENERATE FEATURES: 
    # Then, save features in file: feats_train.csv
    # ... & test feats: feats_test.csv


    feat_cols = {}
    for ft in args.feats:
        colset = set()
        # check if it's a base feature, if so update
        if ft in base_feat:
            colset.update(base_feat[ft])
        else:
            for ftbase in feat_info[ft]:
                colset.update(base_feat[ftbase])
        feat_cols[ft] = list(colset)

    # Determine subgroups within each subgroup bin:
    subgroups = json_extract_values(subgroups_bins)

    for i in tqdm.tqdm(range(1, 2), desc="test-split"):
        #train_df, test_df, train_y, test_y = get_train_test(df, i, label=args.endpoint)
        
        train_df = df_train
        test_df = df_test
        train_y = train_df[args.endpoint]
        test_y = test_df[args.endpoint]
        # Reset test_df indices for subgroup indexing
        test_df = test_df.reset_index()


        for fname, fcolumns in tqdm.tqdm(feat_cols.items(),
                                         desc="feats", leave=False):
            base_res = {
                "file": args.train_file,
                "feat": fname,
                "endpoint": args.endpoint,
                "fold": i
            }

            # for both train and test get only those columns
            train_x = train_df[fcolumns]

             # Apply feature preprocessing: StandardScaler
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x)
            


            # train the logistic on train
            
            #---- TODO: add grid search for Logistic: 
            # "C" : [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 5]
            # L2 reg is default. 

            logr = LogisticRegression(penalty="none", max_iter=1000, solver='lbfgs') # Used to be 2000 for 9-6, often didn't converge.
            # Try adjusting logistic hyperparams: maybe add penalty, change solver from lbfgs...
            # Maybe not converging for logistic because of multicollinearity?
            logr.fit(train_x, train_y)
            

            ################## Begin subgroups test
            # Loop through test eval for each subgroup
            for curr_subgroup in subgroups:
                # Get indices of rows that match curr_subgroup
                cs_ind = test_df.loc[test_df[curr_subgroup]==1].index
                # For subgroups, select only current 'sg' from test_df & test_y.
                sg_test_df = test_df[test_df[curr_subgroup] == 1]
                sg_test_y = test_y.iloc[cs_ind] # use column indices for test_y, bc no subgroup cols here
                
                # Get only desired feature cols
                test_x = sg_test_df[fcolumns]
                test_x = scaler.transform(test_x)
                # get the test encounter id
                #test_idx = sg_test_df["Encounter"]
  
                accuracy, microF1, macroF1 = evaluate_results(logr, test_x, sg_test_y)

                perf_res = {
                    "model": "logr",
                    "ts": datetime.now(),
                    "accuracy": accuracy,
                    "microF1": microF1,
                    "macroF1": macroF1,
                    "subgroup": curr_subgroup,
                    "test_samp_size": len(sg_test_y)
                }
                mcol.insert_one({**base_res, **perf_res})
                #tmp = dict(zip(test_idx, y_hat))
                #raw_res = {
                #    "model": "logr",
                #    "pred": json.dumps(tmp)
                #}
                #raw_mcol.insert_one({**base_res, **raw_res})


            for mname, mk_dict in tqdm.tqdm(MODEL_PARAMS.items(),
                                            desc="models", leave=False):
                gs = skms.GridSearchCV(mk_dict["model"],
                                       mk_dict["params"],
                                       cv=5,
                                       n_jobs=4,
                                       scoring='accuracy')
                gs.fit(train_x, train_y)  
                # Loop through test eval for each subgroup
                for curr_subgroup in subgroups:
                    # Get indices of rows that match curr_subgroup
                    cs_ind = test_df.loc[test_df[curr_subgroup]==1].index
                    # For subgroups, select only current 'sg' from test_df & test_y.
                    sg_test_df = test_df[test_df[curr_subgroup] == 1]
                    sg_test_y = test_y.iloc[cs_ind] # use column indices for test_y, bc no subgroup cols here
                    
                    # Get only desired feature cols
                    test_x = sg_test_df[fcolumns]
                    # get the test encounter id
                    #test_idx = sg_test_df["Encounter"]
                    accuracy, microF1, macroF1 = evaluate_results(gs, test_x, sg_test_y)
                    perf_res = {
                        "model": mname,
                        "ts": datetime.now(),
                        "accuracy": accuracy,
                        "microF1": microF1,
                        "macroF1": macroF1,
                        "subgroup": curr_subgroup,
                        "test_samp_size": len(sg_test_y)
                    }
                    mcol.insert_one({**base_res, **perf_res})
                    

                    # Get SHAP for XGBoost
                    '''
                    if mname == "xgboost":
                        model = gs.best_estimator_ # Best XGBoost model.
                        explainer = shap.Explainer(model)
                        shap_values = explainer(np.ascontiguousarray(test_x))
                        shap_importance = shap_values.abs.mean(0).values
                        sorted_idx = shap_importance.argsort()
                        ordered_shaps = shap_importance[sorted_idx]
                        names_ordered_shaps = np.array(fcolumns)[sorted_idx]
                        # Save Ordered shaps & names.
                        xg_shap_res = {
                            "model": mname,
                            "ts": datetime.now(),
                            "auc": auc,
                            "aps": aps,
                            "subgroup": curr_subgroup,
                            "test_samp_size": len(sg_test_y),
                            "shap_ordered_names": names_ordered_shaps.tolist(),
                            "shap_ordered_importance": ordered_shaps.tolist()
                        }
                        shap_mcol.insert_one({**base_res, **xg_shap_res})
                    '''
                        

    mclient.close()


if __name__ == '__main__':
    main()

