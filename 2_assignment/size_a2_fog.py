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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.preprocessing import sequence, text


from evalHelper import read_json, evaluate_results, get_train_test


os.chdir('/Users/chasefensore/BMI550-NLP/2_assignment')

MODEL_PARAMS = {
   
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
    

def frac(dataframe, fraction, other_info=None):
    """
    Returns fraction of data
    From: https://stackoverflow.com/questions/52386560/randomly-selecting-different-percentages-of-data-in-python
    """
    return dataframe.sample(frac=fraction)


# Function to parse and convert the string representation of arrays to numerical features
# For turning n-grams into numerical columns.
# Function to parse and convert the string representation of arrays to numerical features
def parse_array_string(array_str):
    # Remove leading/trailing spaces and brackets, then split by spaces
    values = str(array_str).strip('[]\n').split()

    # Check if 'NaN' is in the list of values
    if 'NaN' in values:
        print("Array contains 'NaN':", values)
        print(array_str)
        print(values)
    if 'nan' in values:
        print("Array contains 'nan':", values)
        print(array_str)
        print(values)
    # Convert the string values to integers
    numerical_values = [float(val) for val in values]

    return numerical_values
# Change ACC script to do "ALL" properly. Then cp & paste here (eval on sgs: ALL, female, male, MDP-UPDRS stage...)






def main():
    parser = argparse.ArgumentParser()
    # mongo information
    username = urllib.parse.quote_plus('fensorechase')
    password = urllib.parse.quote_plus('7pzNiMi7dD!d@Ab')
    parser.add_argument("-mongo_url", default = 'mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.10.5') 
    parser.add_argument("-mongo_db",
                        default="aequitas")
    parser.add_argument("-mongo_col",
                        default="B-BioNLP",
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
    parser.add_argument("-train_folds", 
                        default="./data/train_folds.csv", 
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
       
       "demo_clinical_simple_lexical"
       
        ])
    
    """
    "age_at_enrollment",
       "Sex",
       "race",

       "demo",
       "clinical",
       "simple_lexical",
       "POS",

       "demo_and_clinical",
       "demo_clinical_simple_lexical",
       "demo_clinical_simple_lexical_POS"
    
    """
    #     "word_ngram"
    # "char_ngram",
    # "tfidf_ngram"
   
    args = parser.parse_args()

    # setup mongo
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]
    # raw_mcol = mdb["10_7_ACC_raw_agnostic"] # Used to be: "raw_baseline"
    # shap_mcol = mdb["9_20_shap_xgb_MLH"] # To save shap for XGBoost.

    #df_train = pd.read_csv(args.train_file)
    #df_test = pd.read_csv(args.test_file)
    #df = pd.concat([df_train, df_test], axis=0)
    df_full = pd.read_csv(args.train_folds)

    # Select percentage of training set:



    # Only eval holdout_test_df on BEST model among the 10 folds. 
    # Best model from CV: chosen according to accuracy.
    holdout_test_df = pd.read_csv(args.test_file)

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

    train_pct_samps = [1, 0.8, 0.6, 0.4, 0.2]

    for j in tqdm.tqdm(range(1,6), desc="train-size"):
        df = frac(df_full, train_pct_samps[j-1])


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

        for i in tqdm.tqdm(range(1, 6), desc="test-split"):
            train_df, test_df, train_y, test_y = get_train_test(df, i, label=args.endpoint)
            
            # Reset test_df indices for subgroup indexing
            test_df = test_df.reset_index()


            for fname, fcolumns in tqdm.tqdm(feat_cols.items(),
                                            desc="feats", leave=False):
                base_res = {
                    "file": args.train_folds,
                    "feat": fname,
                    "endpoint": args.endpoint,
                    "fold": i
                }

                # for both train and test get only those columns
                train_x = train_df[fcolumns]


                #VECTORIZE N-GRAMS: word-level 
                vectorizer = CountVectorizer(ngram_range=(1, 3), analyzer="word", tokenizer=None, preprocessor=None,
                                                    max_features=1000)

                # if any embedding features present, chop off.
                # Then, exapand to numerical cols for TRAIN.
                # Then, add numerical cols back into train_x.
                if "pp_fall_description" in fcolumns:
                    train_x.fillna('', inplace=True)
                    word_training_vectors = vectorizer.fit_transform(train_x['pp_fall_description']).toarray()

                    #desired_shape = (word_training_vectors.shape[0], 1000)  # Assuming you want a fixed shape of (number_of_samples, 1000)
                    #X_padded = np.zeros(desired_shape)
                    #X_padded[:, :word_training_vectors.shape[1]] = word_training_vectors
                    #word_training_vectors = X_padded

                    # Convert the NumPy array into a DataFrame
                    word_training_vectors = pd.DataFrame(word_training_vectors, columns=[f'Column_{i}' for i in range(word_training_vectors.shape[1])])

                    # Concatenate the new DataFrame with the original DataFrame along the columns axis
                    # train_x = pd.concat([train_x, word_training_vectors], axis=1)
                    # Drop orig string rep of vectors:
                    # train_x = train_x.drop(columns=['pp_fall_description'])
                    train_x = word_training_vectors


                # Impute, then standardize
                # Identify numerical columns (Train, test same names)
                numerical_columns = train_x.select_dtypes(include=[np.number]).columns


                # TRAIN:
                # Iterate through each column and fill NaN values with column median
                for column in numerical_columns:
                    median = train_x[column].median()
                    train_x[column].fillna(median, inplace=True)

                # Then for all feats, apply feature preprocessing: StandardScaler
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

                    # if any embedding features present, chop off.
                    # Then, exapand to numerical cols for TEST.
                    # Then, add numerical cols back into test_x.

                    #VECTORIZE N-GRAMS: word-level 


                    if "pp_fall_description" in fcolumns:
                        test_x.fillna('', inplace=True)
                        word_test_vectors = vectorizer.transform(test_x['pp_fall_description']).toarray()

                        # Convert the NumPy array into a DataFrame
                        word_test_vectors = pd.DataFrame(word_test_vectors, columns=[f'Column_{i}' for i in range(word_test_vectors.shape[1])])

                        # Concatenate the new DataFrame with the original DataFrame along the columns axis
                        #test_x = pd.concat([test_x, word_test_vectors], axis=1)
                        # Drop orig string rep of vectors:
                        #test_x = test_x.drop(columns=['pp_fall_description'])
                        test_x = word_test_vectors


                    # TEST:
                    # Iterate through each column and fill NaN values with column median
                    for column in numerical_columns:
                        median = test_x[column].median()
                        test_x[column].fillna(median, inplace=True)
                
                    # Then, scale all feature columns:
                    test_x = scaler.transform(test_x)
                    # get the test encounter id
                    #test_idx = sg_test_df["Encounter"]
    
                    accuracy, microF1, macroF1 = evaluate_results(logr, test_x, sg_test_y)

                    perf_res = {
                        "model": "logr",
                        "set": "validation",
                        "train_pct": train_pct_samps[j-1],
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

                    ################################
                    # holdout_test_df
                    # Now, eval model on HOLDOUT set:
                    # Get indices of rows that match curr_subgroup
                    cs_ind = holdout_test_df.loc[holdout_test_df[curr_subgroup]==1].index
                    # For subgroups, select only current 'sg' from holdout_test_df & y from holdout_test_df.
                    sg_test_df = holdout_test_df[holdout_test_df[curr_subgroup] == 1]
                    sg_test_y = holdout_test_df[args.endpoint].iloc[cs_ind] # use column indices for test_y, bc no subgroup cols here
                    
                    # Get only desired feature cols
                    test_x = sg_test_df[fcolumns]


                    #VECTORIZE N-GRAMS: word-level 
                    
                    # if any embedding features present, chop off.
                    # Then, exapand to numerical cols for TEST.
                    # Then, add numerical cols back into test_x.
                    if "pp_fall_description" in fcolumns:
                        # Replace "nan" with empty string, since "nan" fall descriptions are present in holdout set.
                        test_x.fillna('', inplace=True)
                        word_test_vectors = vectorizer.transform(test_x['pp_fall_description']).toarray()

                        # Convert the NumPy array into a DataFrame
                        word_test_vectors = pd.DataFrame(word_test_vectors, columns=[f'Column_{i}' for i in range(word_test_vectors.shape[1])])

                        # Concatenate the new DataFrame with the original DataFrame along the columns axis
                        #test_x = pd.concat([test_x, word_test_vectors], axis=1)
                        # Drop orig string rep of vectors:
                        #test_x = test_x.drop(columns=['pp_fall_description'])
                        test_x = word_test_vectors



                    # Impute, then standardize
                    # Identify numerical columns (Train, test same names)
                    numerical_columns = test_x.select_dtypes(include=[np.number]).columns

                    # HOLDOUT:
                    # Iterate through each column and fill NaN values with column median
                    for column in numerical_columns:
                        median = test_x[column].median()
                        test_x[column].fillna(median, inplace=True)

                    # Then, scale all feature columns:
                    test_x = scaler.transform(test_x)
                    # get the test encounter id
                    #test_idx = sg_test_df["Encounter"]
    
                    accuracy, microF1, macroF1 = evaluate_results(logr, test_x, sg_test_y)

                    perf_res = {
                        "model": "logr",
                        "set": "holdout",
                        "train_pct": train_pct_samps[j-1],
                        "ts": datetime.now(),
                        "accuracy": accuracy,
                        "microF1": microF1,
                        "macroF1": macroF1,
                        "subgroup": curr_subgroup,
                        "test_samp_size": len(sg_test_y)
                    }
                    mcol.insert_one({**base_res, **perf_res})



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

                        # if any embedding features present, chop off.
                        # Then, exapand to numerical cols for TEST.
                        # Then, add numerical cols back into test_x.


                        #VECTORIZE N-GRAMS: word-level 
                        

                        if "pp_fall_description" in fcolumns:
                            test_x.fillna('', inplace=True)

                            word_test_vectors = vectorizer.transform(test_x['pp_fall_description']).toarray()

                            # Convert the NumPy array into a DataFrame
                            word_test_vectors = pd.DataFrame(word_test_vectors, columns=[f'Column_{i}' for i in range(word_test_vectors.shape[1])])

                            # Concatenate the new DataFrame with the original DataFrame along the columns axis
                            #test_x = pd.concat([test_x, word_test_vectors], axis=1)
                            # Drop orig string rep of vectors:
                            #test_x = test_x.drop(columns=['pp_fall_description'])
                            test_x = word_test_vectors


                    
                        # Impute, then standardize
                        # Identify numerical columns (Train, test same names)
                        numerical_columns = test_x.select_dtypes(include=[np.number]).columns


                        # TEST:
                        # Iterate through each column and fill NaN values with column median
                        for column in numerical_columns:
                            median = test_x[column].median()
                            test_x[column].fillna(median, inplace=True)

                        # Then, scale all feature columns:
                        test_x = scaler.transform(test_x)

                        # Then, cross-validate:
                        accuracy, microF1, macroF1 = evaluate_results(gs, test_x, sg_test_y)
                        perf_res = {
                            "model": mname,
                            "set": "validation",
                            "train_pct": train_pct_samps[j-1],
                            "ts": datetime.now(),
                            "accuracy": accuracy,
                            "microF1": microF1,
                            "macroF1": macroF1,
                            "subgroup": curr_subgroup,
                            "test_samp_size": len(sg_test_y)
                        }
                        mcol.insert_one({**base_res, **perf_res})
                        

                        # Holdout evaluation:
                        ################################
                        # holdout_test_df
                        # Now, eval model on HOLDOUT set:
                        # Get indices of rows that match curr_subgroup
                        cs_ind = holdout_test_df.loc[holdout_test_df[curr_subgroup]==1].index
                        # For subgroups, select only current 'sg' from holdout_test_df & y from holdout_test_df.
                        sg_test_df = holdout_test_df[holdout_test_df[curr_subgroup] == 1]
                        sg_test_y = holdout_test_df[args.endpoint].iloc[cs_ind] # use column indices for test_y, bc no subgroup cols here
                        
                        # Get only desired feature cols
                        test_x = sg_test_df[fcolumns]

    
                        # if any embedding features present, chop off.
                        # Then, exapand to numerical cols for TEST.
                        # Then, add numerical cols back into test_x.
                        if "pp_fall_description" in fcolumns:
                            test_x.fillna('', inplace=True)
                            word_test_vectors = vectorizer.transform(test_x['pp_fall_description']).toarray()

                            # Convert the NumPy array into a DataFrame
                            word_test_vectors = pd.DataFrame(word_test_vectors, columns=[f'Column_{i}' for i in range(word_test_vectors.shape[1])])

                            # Concatenate the new DataFrame with the original DataFrame along the columns axis
                            #test_x = pd.concat([test_x, word_test_vectors], axis=1)
                            # Drop orig string rep of vectors:
                            #test_x = test_x.drop(columns=['pp_fall_description'])
                            test_x = word_test_vectors


                        # HOLDOUT:
                        # Iterate through each column and fill NaN values with column median
                        for column in numerical_columns:
                            median = test_x[column].median()
                            test_x[column].fillna(median, inplace=True)

                        # Then, scale all feature columns:
                        test_x = scaler.transform(test_x)
                        # get the test encounter id
                        #test_idx = sg_test_df["Encounter"]
        
                        accuracy, microF1, macroF1 = evaluate_results(gs, test_x, sg_test_y)

                        perf_res = {
                            "model": mname,
                            "set": "holdout",
                            "train_pct": train_pct_samps[j-1],
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

