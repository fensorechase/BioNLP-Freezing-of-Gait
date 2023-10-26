import argparse
import datetime
import pandas as pd
from pymongo import MongoClient
import urllib.parse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-output", help="output file",
                        default="size_results_fog.csv") 
    # subgroups: results_clean_tract_ahrq_AQIimp1_baseline_SGs_2.csv
    # Sep subgroups: results_clean_tract_ahrq_AQIimp1_baseline_sep_SGs.csv
    username = urllib.parse.quote_plus('fensorechase')
    password = urllib.parse.quote_plus('7pzNiMi7dD!d@Ab')
    parser.add_argument("-mongo_url", default = 'mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+1.10.5')
    # 'mongodb+srv://%s:%s@cluster0.ysobo3u.mongodb.net/' % (username, password)) 
    parser.add_argument("-mongo_db",
                        default="aequitas")
    parser.add_argument("-mongo_col",
                        default="B-BioNLP",
                        help="collection_type") # For subgroup results, set default="subgroups_baseline", for entire dataset, default="baseline"
    args = parser.parse_args()

    # setup the mongo stuff
    mclient = MongoClient(args.mongo_url)
    mdb = mclient[args.mongo_db]
    mcol = mdb[args.mongo_col]

    pipe_list = [{
    "$group":
            {
                "_id":
                {
                    "feat": "$feat",
                    "model": "$model",
                    "train_pct": "$train_pct",
                    "file": "$file",
                    "set": "$set",
                    "endpoint": "$endpoint",
                    "subgroup": "$subgroup"
                },
                "accuracy":
                {
                    "$avg": "$accuracy"
                },
                "acc_sd":
                {
                    "$stdDevSamp": "$accuracy"
                },
                "microF1":
                {
                    "$avg": "$microF1"
                },
                "microF1_sd":
                {
                    "$stdDevSamp": "$microF1"
                },
                "macroF1":
                {
                    "$avg": "$macroF1"
                },
                "macroF1_sd":
                {
                    "$stdDevSamp": "$macroF1"
                },
                "n_runs": 
                {
                    "$sum": 1
                },
                "test_samp_size": 
                {
                    "$avg": "$test_samp_size"
                }
            }
    },
    {"$project":
            {
                "model": "$_id.model",
                "train_pct": "$_id.train_pct",
                "feat": "$_id.feat",
                "file": "$_id.file",
                "set": "$_id.set",
                "endpoint": "$_id.endpoint",
                "accuracy": "$accuracy",
                "accuracy_sd": "$accuracy_sd",
                "microF1": "$microF1",
                "microF1_sd": "$microF1_sd",
                "macroF1": "$macroF1",
                "macroF1_sd": "$macroF1_sd",
                "n_runs": "$n_runs",
                "_id": 0,
                "subgroup": "$_id.subgroup",
                "test_samp_size": "$test_samp_size"
            }
            }
    ]


    tmp = list(mcol.aggregate(pipe_list))
    tmp_df = pd.DataFrame.from_records(tmp)
    #print(tmp_df)
    mclient.close()

    tmp_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

