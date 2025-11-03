import warnings
warnings.filterwarnings('ignore')
import time 

from src.utils import *
from src.model_analysis import *

import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument("--dataset_names", nargs="*", type=str, default=[ "HE","WH", "DV"])#,
CLI.add_argument("--feature_names", nargs="*", type=str, default=["fastText","bert","glove" , 'tf', 'tfidf', 'roberta']) 
args = CLI.parse_args()

#+------------------------------------------------------------------+
#| GLOBAL FEATURES                                                  |
#+------------------------------------------------------------------+


CLASSIFIERS = get_classifiers()
FEATURES = args.feature_names 
DATASET_NAMES = args.dataset_names
print(DATASET_NAMES)

SEP = os.path.sep
IDENTITY_SET = read_identity_terms(f"dataset{SEP}bias_data{SEP}identity.txt")

def param_select_folds(dataset,feature, folds=5):
    path = f"results{SEP}{dataset}{SEP}classifiers{SEP}"   

    if not check_exist(path):
        create_dir(path)

    #load data
    path_ft = f"results{SEP}{dataset}{SEP}features{SEP}{feature}{SEP}"
    path_class = f"dataset{SEP}{dataset}{SEP}preprocess{SEP}"
    train = {}
    val = {}


    for f in range(1, folds+1):
        train[f-1] = load_pickle(f"{path_ft}train{SEP}F{f}.pickle")
        val[f-1] = load_pickle(f"{path_ft}val{SEP}F{f}.pickle")

    #load labels
    class_train = load_pickle(f"{path_class}train_labels.pickle")
    class_val = load_pickle(f"{path_class}val_labels.pickle")

    # select the best params for each classifier
    for clf in CLASSIFIERS.keys():
        print("CLASSIFIER:", clf)
        start = time.time()
        #fit the classifier with all folds
        
        estimators = fit_all_folds(train, class_train, classifier=clf, folds=folds)
        #select the best params
        best_params = best_estimator_folds(estimators, val, class_val, folds=folds)

        #save best params 
        file_path = f"{path}{clf}_{feature}_best_params_v2.pickle"
        save_pickle(file_path, best_params)
        print("\n Time:", time.time() - start)

def param_select(dataset,feature):
    path = f"results{SEP}{dataset}{SEP}classifiers{SEP}"   

    if not check_exist(path):
        create_dir(path)

    #load data
    path_ft = f"results{SEP}{dataset}{SEP}features{SEP}{feature}{SEP}"
    path_class = f"dataset{SEP}{dataset}{SEP}preprocess{SEP}"
    
    ext = ".pickle" if feature != "bert" else ""

    train = load_pickle(f"{path_ft}train{ext}")
    val = load_pickle(f"{path_ft}val{ext}")

    #load labels
    
     
    class_train =  pd.read_csv(f"{path_class}train.csv")["label"]               
    class_val = pd.read_csv(f"{path_class}val.csv")["label"]  

    # select the best params for each classifier
    for clf in CLASSIFIERS.keys():
        print("CLASSIFIER:", clf)
        start = time.time()
        #fit the classifier with all folds
        estimators = fit_all(train, class_train, classifier=clf)

        #select the best params
        best_params = best_estimator(estimators, val, class_val)

        #save best params 
        file_path = f"{path}{clf}_{feature}_best_params_v2.pickle"
        print(best_params)
        save_pickle(file_path, best_params)
        print("\n Time:", time.time() - start)


def main():
    print("MAIN")
    for dt in DATASET_NAMES:
        print("dataset:", dt)
        for feature in FEATURES:
            print(feature)
            if dt in ["WH", "DV"]:
                param_select_folds(dt,feature)
            else: 
                param_select(dt,feature)

                
if __name__ == "__main__":
    main()
    		
