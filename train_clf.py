import warnings
warnings.filterwarnings('ignore')

from src.model_analysis import *
from src.utils import *
import argparse
import time

CLI = argparse.ArgumentParser()
CLI.add_argument("--dataset_names", nargs="*", type=str, default=["HE", "WH", "DV"]) 
CLI.add_argument("--feature_names", nargs="*", type=str, default=["bert","glove", "fastText", 'tf', 'tfidf', 'roberta'])
args = CLI.parse_args()

#+------------------------------------------------------------------+
#| GLOBAL FEATURES                                                  |
#+------------------------------------------------------------------+


CLASSIFIERS = get_classifiers()
FEATURES = args.feature_names 
DATASET_NAMES = args.dataset_names
print(DATASET_NAMES)
METRICS = ["accuracy","f1_score", "auc","fned", "fped", "subgroup","bpsn","bnsp"]

SEP = os.path.sep
IDENTITY_SET = read_identity_terms(f"dataset{SEP}UB{SEP}identity.txt")

def train_clf(dataset,df_bias_test):
    path = f"results{SEP}{dataset}{SEP}classifiers{SEP}"   
    print(path)
    if not check_exist(path):
        create_dir(path)
    
    df_results = pd.DataFrame()

    if not check_exist('results/df_time_manage_clf.csv'):
        df_time_manage = pd.DataFrame()

    else: 
        df_time_manage = pd.read_csv('results/df_time_manage_clf.csv', index_col=[0])

    for feature in FEATURES:

        #load data
        path_ft = f"results{SEP}{dataset}{SEP}features{SEP}{feature}{SEP}"
        path_class = f"dataset{SEP}{dataset}{SEP}preprocess{SEP}"

        #load the data
        class_train = pd.read_csv(f"{path_class}train.csv")["label"]
        class_test = pd.read_csv(f"{path_class}test.csv")["label"]

        #load unbiased dataset features
        if feature in ['tf', 'tfidf']:
            path_ub_ft = f"results{SEP}UB{SEP}features{SEP}{feature}{SEP}test_{dataset}.pickle"
            bias_test = load_pickle(path_ub_ft)
            print(bias_test.shape)           
        else:
            path_ub_ft = f"results{SEP}UB{SEP}features{SEP}{feature}{SEP}test.pickle"
            bias_test = load_pickle(path_ub_ft)
            print(bias_test.shape)      

        #save predictions
        y_pred = {}
        y_proba = {}
        

        for clf in CLASSIFIERS.keys():
            print("CLASSIFIER:", clf)
            start = time.time()
            model = copy.deepcopy(CLASSIFIERS[clf])
            #set best params
            best_params = load_pickle(f"{path}{clf}_{feature}_best_params_v2.pickle")
            model.set_params(**best_params)

            
            #load features 
            train = load_pickle(f"{path_ft}train.pickle")
            test = load_pickle(f"{path_ft}test.pickle")

            #fit the classifier   
            train = train.to_numpy() if isinstance(train, pd.DataFrame) else train    
            test = test.to_numpy() if isinstance(test, pd.DataFrame) else test      
            model.fit(train, class_train)
            

            print(f"Time: {time.time() - start}")
            new_row = {'feature': feature,'classifier': clf, 'time': time.time() - start}
            df_time_manage = df_time_manage.append(new_row, ignore_index=True)
            df_time_manage.to_csv('results/df_time_manage_clf.csv')
            bias_test = bias_test.to_numpy() if isinstance(bias_test, pd.DataFrame) else bias_test

            #predict 
            y_pred = model.predict(test)
            y_proba = model.predict_proba(test)
            y_pred_bias = model.predict(bias_test)

            #Calculate the metrics
            clf_name = f"{clf}_{feature}"
            new_row = calculate_all_metrics(clf_name, df_bias_test, IDENTITY_SET,None, class_test, y_pred, y_proba, y_pred_bias)
            df_results = df_results.append(new_row, ignore_index=True)
            df_results.to_csv(f"{path}metrics_v3.csv")    

    # save average value 
    df_avg = pd.DataFrame()
    output = df_results.groupby(['model'], as_index=False).agg({m:['mean','std'] for m in METRICS})
    df_avg["model"] = output["model"]

    for col in METRICS:
        df_avg[f"{col}_mean"] = output[col][['mean']].round(3)
        df_avg[f"{col}_std"] = output[col][['std']].round(3)  
    df_avg.to_csv(f"{path}metrics_average_v3.csv")

def train_clf_folds(dataset,df_bias_test, folds=5):
    path = f"results{SEP}{dataset}{SEP}classifiers{SEP}"   
    print(path)
    if not check_exist(path):
        create_dir(path)
    
    df_results = pd.DataFrame()   

    for feature in FEATURES:

        #load data
        path_ft = f"results{SEP}{dataset}{SEP}features{SEP}{feature}{SEP}"
        path_class = f"dataset{SEP}{dataset}{SEP}preprocess{SEP}"

        #load the labels dictionary [0-4]
        class_train = load_pickle(f"{path_class}train_labels.pickle")
        class_test = load_pickle(f"{path_class}test_labels.pickle")

        #load unbiased dataset features
        if feature not in ['tf', 'tfidf']:            
            path_ub_ft = f"results{SEP}UB{SEP}features{SEP}{feature}{SEP}test.pickle"
            bias_test = load_pickle(path_ub_ft)
            print(bias_test.shape)      

        #save predictions
        y_pred = {}
        y_proba = {}
        

        for clf in CLASSIFIERS.keys():
            start = time.time()
            print("CLASSIFIER:", clf)
            
            model = copy.deepcopy(CLASSIFIERS[clf])
            #set best params
            best_params = load_pickle(f"{path}{clf}_{feature}_best_params_v2.pickle")
            model.set_params(**best_params)

            for f in range(1, folds+1):
                #load features 
                train = load_pickle(f"{path_ft}train{SEP}F{f}.pickle")
                test = load_pickle(f"{path_ft}test{SEP}F{f}.pickle")

                #load UB dataset if feature is tf or tfidf
                if feature in ['tf', 'tfidf']:            
                    path_ub_ft = f"results{SEP}UB{SEP}features{SEP}{feature}{SEP}test_{dataset}_F{f}.pickle"
                    bias_test = load_pickle(path_ub_ft) 
                    print(bias_test.shape)

                #fit the classifier  
                train = train.to_numpy() if isinstance(train, pd.DataFrame) else train
                test = test.to_numpy() if isinstance(test, pd.DataFrame) else test           
                model.fit(train, class_train[f-1])
                
                bias_test = bias_test.to_numpy() if isinstance(bias_test, pd.DataFrame) else bias_test
                #predict 
                y_pred = model.predict(test)
                y_proba = model.predict_proba(test)
                y_pred_bias = model.predict(bias_test)

                #Calculate the metrics
                clf_name = f"{clf}_{feature}"
                new_row = calculate_all_metrics(clf_name, df_bias_test, IDENTITY_SET,f, class_test[f-1], y_pred, y_proba, y_pred_bias)
                df_results = df_results.append(new_row, ignore_index=True)
            df_results.to_csv(f"{path}metrics_v3.csv")    

    # save average value 
    df_avg = pd.DataFrame()
    output = df_results.groupby(['model'], as_index=False).agg({m:['mean','std'] for m in METRICS})
    df_avg["model"] = output["model"]

    for col in METRICS:
        df_avg[f"{col}_mean"] = output[col][['mean']].round(3)
        df_avg[f"{col}_std"] = output[col][['std']].round(3) 
    df_avg.to_csv(f"{path}metrics_average_v3.csv")


def main():
    print("MAIN")
    set_names = ["train","test"]

    #Read data
    print("Read  data")
    
    # Read unbiased dataset
    path_ub_data = f"dataset{SEP}UB{SEP}preprocess{SEP}test.pickle"
    df_bias_test = load_pickle(path_ub_data)
    print(df_bias_test.shape)  

    #Adds a boolean column for each subgroup to the DataFrame.
    add_subgroup_columns_from_text(df_bias_test, text_column="text", subgroups=IDENTITY_SET)    
    
    print("CLASSIFICATION")
    
    for dataset_name in DATASET_NAMES:
        if dataset_name =="HE":
            #for i in range(5):
            train_clf(dataset_name,df_bias_test)
        else:        
            train_clf_folds(dataset_name, df_bias_test, folds=5)       


if __name__ == "__main__":
    main()
