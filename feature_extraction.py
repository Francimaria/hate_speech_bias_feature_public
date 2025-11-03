#imports 
import warnings
warnings.filterwarnings('ignore')

import os
import time
import argparse

import scipy

from src.utils import *
from src.model_analysis import *


CLI = argparse.ArgumentParser()
CLI.add_argument("--dataset_names", nargs="*", type=str, default=["HE","WH","DV"]) 
CLI.add_argument("--feature_extraction", nargs="*", type=bool, default=True)
CLI.add_argument("--feature_names", nargs="*", type=str, default=['tf', 'tfidf',"fastText","bert","glove","roberta"])  

args = CLI.parse_args()

#+------------------------------------------------------------------+
#| GLOBAL FEATURES                                                  |
#+------------------------------------------------------------------+

#["bert", "glove", "fastText"]
FEATURES = args.feature_names 

DATASET_NAMES = args.dataset_names
print(DATASET_NAMES)

SEP = os.path.sep

IDENTITY_SET = read_identity_terms(f"dataset{SEP}UB{SEP}identity.txt")


def get_features():
    feature_extraction = {}

    if len(list(set(FEATURES) & set(["bert", "glove", "fastText", "tf", "tfidf", "falcon", "roberta"]))) < 1:
        print("Invalid value!")
        return {}
        
    if "glove" in FEATURES: 
        from zeugma.embeddings import EmbeddingTransformer

        glove = EmbeddingTransformer('glove-twitter-200') 
        feature_extraction["glove"] = glove

    if "fastText" in FEATURES: 
        from zeugma.embeddings import EmbeddingTransformer

        fastText = EmbeddingTransformer('fasttext-wiki-news-subwords-300')
        feature_extraction["fastText"] = fastText
     
    if "bert" in FEATURES:
        from src.feature_extraction_transformers import FeatureExtractionTransformer
        from transformers import BertTokenizer, BertModel   

        model = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model)
        bert_model = BertModel.from_pretrained(model)

        bert = FeatureExtractionTransformer(base_tokenizer=tokenizer, base_model=bert_model)
        feature_extraction["bert"] = bert 

    if "roberta" in FEATURES:
        from src.feature_extraction_transformers import FeatureExtractionTransformer
        from transformers import RobertaModel, RobertaTokenizer   

        model = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(model)
        roberta_model = RobertaModel.from_pretrained(model)
        roberta = FeatureExtractionTransformer(base_tokenizer=tokenizer, base_model=roberta_model)
        
        feature_extraction["roberta"] = roberta  
        

    if "tf" in FEATURES: 
        from sklearn.feature_extraction.text import CountVectorizer

        feature_extraction["tf"] = CountVectorizer(analyzer="word", max_features=2000)

    if "tfidf" in FEATURES: 
        from sklearn.feature_extraction.text import TfidfVectorizer
        feature_extraction["tfidf"] = TfidfVectorizer(max_features=2000)         
    
    return feature_extraction

def run_feature_extraction(feature, X, fit=False):
    if fit:
        feature.fit(X)
    
    x_pln = feature.transform(X)
    
    if scipy.sparse.issparse(x_pln):
        # convert sparse matrix to dataframe
        x_pln = pd.DataFrame.sparse.from_spmatrix(x_pln)
        #x_pln = x_pln.toarray()
    return x_pln

def feature_extraction(X_train, X, set_name, dataset_name="HE", save=False, features={}):
    """"
    X: dictionary with each set ex.:  
    X = { "train": X_train,"test": X_test,"val": X_val,
        "bias_train": X_bias_train,"bias_test": X_bias_test,"bias_val": X_bias_val}
    """
   
    #textual content to be transformed

    if not check_exist('results/df_time_manage.csv'):
        df_time_manage = pd.DataFrame()

    else: 
        df_time_manage = pd.read_csv('results/df_time_manage.csv', index_col=[0])
  
   
    print("dataset:", dataset_name)

    for name, feature in features.items():
        print(name)
        start = time.time()
        path = f"results{SEP}{dataset_name}{SEP}features{SEP}{name}{SEP}"
        if save:
            if not check_exist(path):
                create_dir(path)
        if dataset_name in ["UB","HE"]:
            X = X.to_frame() if isinstance(X,pd.Series) else X
            if name in ["tf", "tfidf"]:
                if dataset_name == "UB":
                    feature.fit(X_train["HE"]["text"])
                    x_pln = run_feature_extraction(feature, X["text"])
                    save_pickle(f"{path}{set_name}_HE.pickle", x_pln)
                    
                    for _dataset_name in ["WH", "DV"]:
                        fold = X_train[_dataset_name].keys()
                        for f in fold:
                            print("fold", f)
                            data = pd.DataFrame(X_train[_dataset_name][f])
                            feature.fit(data["text"])
                            x_pln = run_feature_extraction(feature, X["text"])
                            if save:
                                save_pickle(f"{path}{set_name}_{_dataset_name}_F{f+1}.pickle", x_pln)
                else:
                    feature.fit(X_train["text"])
                    x_pln = run_feature_extraction(feature, X["text"])
                    if save:
                        save_pickle(f"{path}{set_name}.pickle", x_pln)
            else:
                x_pln = run_feature_extraction(feature, X["text"])
                if save:
                    save_pickle(f"{path}{set_name}.pickle", x_pln)

        else:
            
            fold = X.keys()
            for f in fold:
                X[f] = X[f].to_frame() if isinstance(X[f],pd.Series) else X[f]
                if name in ["tf", "tfidf"]:
                    feature.fit(X_train[f]["text"])
                    x_pln = run_feature_extraction(feature, X[f]["text"])
                    if save:
                        save_pickle(f"{path}{set_name}.pickle", x_pln)
                else:
                    x_pln = run_feature_extraction(feature, X[f]["text"])  
                path_set = f"{path}{set_name}{SEP}"
                if not check_exist(path_set):
                    create_dir(path_set)
                if save:
                    save_pickle(f"{path_set}F{f+1}.pickle", x_pln)
    
        print(f"Time: {time.time() - start}")
        new_row = {'feature': name, 'time': time.time() - start, 'set_name': set_name}
        df_time_manage = df_time_manage.append(new_row, ignore_index=True)
        df_time_manage.to_csv('results/df_time_manage.csv')
 

def main():
    check_gpu()
    train,test, val = {},{},{}
    ext = ".pickle"
    #Read data
    for dataset_name in DATASET_NAMES:
        print(dataset_name)
        path_data = f"dataset/{dataset_name}/preprocess/"

        if dataset_name == "UB":
            # Read unbiased dataset
            bias_test = load_pickle(f"{path_data}test{ext}")
        else:
            if dataset_name == "HE":
                train[dataset_name] = pd.read_csv(f"{path_data}train.csv")
                test[dataset_name ]= pd.read_csv(f"{path_data}test.csv")
                val[dataset_name] = pd.read_csv(f"{path_data}val.csv")
            else:
                train[dataset_name] = load_pickle(f"{path_data}train{ext}")
                test[dataset_name ]= load_pickle(f"{path_data}test{ext}")
                val[dataset_name] = load_pickle(f"{path_data}val{ext}")
             
    # Feature extraction
    features = get_features()
    for i in range(5):
        if args.feature_extraction:  
            print("Feature extraction")      
            for dataset_name in args.dataset_names:
                if dataset_name == "UB":
                    feature_extraction(X_train=train, X=bias_test, set_name="test",dataset_name=dataset_name, save=False, features=features)            
                else:
                    feature_extraction(X_train=train[dataset_name], X=train[dataset_name], set_name="train",dataset_name=dataset_name, save=False, features=features)
                    feature_extraction(X_train=train[dataset_name], X=test[dataset_name],set_name="test", dataset_name=dataset_name, save=False, features=features)
                    feature_extraction(X_train=train[dataset_name], X=val[dataset_name], set_name="val", dataset_name=dataset_name,save=False, features=features)
    
if __name__ == "__main__":
    main()
