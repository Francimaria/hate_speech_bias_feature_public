import pickle
import os
import pandas as pd 
import numpy as np
import re
from IPython.display import clear_output

from sklearn.model_selection import train_test_split, StratifiedKFold

import tensorflow as tf
import torch

import warnings
warnings.filterwarnings('ignore')

SEP = os.path.sep

def load_pickle(file_path):   
    """
    Load object from specified pickle file
    Parameters
    ----------
    file_path : str or path object
        String or path object to load the object
    """
    if check_exist(file_path):
        file_to_read = open(file_path, "rb")
        return pickle.load(file_to_read)
    return None
    
def save_pickle(file_path, data):
    """
    Saves ```data``` object into specified pickle file
    Parameters
    ----------
    file_path : str or path object
        String or path object to save the object
    data : Any
        Object to save
    """
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def check_exist(file_name):
    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        return False
    return True

def create_dir(file_name):
    #Create the directory
    os.makedirs(file_name) 

def save_predicted_values(path, column_name, values):
    fold_path = "/".join(path.split("/")[:-1])
    if len(fold_path) > 0  and not check_exist(fold_path):    
        create_dir(fold_path)
    df = pd.read_csv(path) if check_exist(path) else pd.DataFrame()
    df = df.drop(df.filter(regex="Unnamed").columns, axis=1)
    df[column_name] = values
    df.to_csv(path)
    print("saved!")

def get_values(df, columns=['text', 'label']):  
    return [df[col] for col in columns]


def get_dataset(dataset_name, path=None):    
    dataset_path = f"dataset{SEP}{dataset_name}{SEP}preprocess{SEP}" if path is None else path
    train = pd.read_csv(f"{dataset_path}train.csv")
    test = pd.read_csv(f"{dataset_path}test.csv")
    val = pd.read_csv(f"{dataset_path}val.csv")
    return train,test,val


def get_unbiased_dataset(path=None): 

    # read unbias dataset
    unbiased_path = f"dataset{SEP}bias_data{SEP}preprocess{SEP}" if path is None else path
    # bias_train = pd.read_csv(f"{unbiased_path}train.csv")
    bias_test = pd.read_csv(f"{unbiased_path}test.csv")
    bias_val = pd.read_csv(f"{unbiased_path}val.csv")

    return bias_test, bias_val

def get_unbiased_features(ft_name="bert",path=None): 

    # read unbias dataset
    unbiased_path = f"results{SEP}bias_data{SEP}features{SEP}{ft_name}{SEP}" if path is None else path
   
    bias_test = np.array(load_pickle(unbiased_path+"bias_test"))
    bias_val = np.array(load_pickle(unbiased_path+"bias_val"))

    return bias_test, bias_val

def check_gpu():
    # Check if GPU is found
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        clear_output(wait=False)
        print('GPU device not found')
    else:
        # specify the GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        torch.cuda.get_device_name(0)
        clear_output(wait=False)
        print("=="*10)
        print("Found GPU at: {}".format(device_name))


def read_identity_terms(identity_terms_path):
    with open(identity_terms_path) as f:
        return [term.strip() for term in f.readlines()]
    

def replaceName(text, name, new_name):
    newText = re.sub(r"\b{}\b".format(name), "{}".format(new_name), text) 
    return str(newText)

def _select_set_text(data, data_selected, term="women", identity_path="dataset/UB/identity.txt"):
    df_new = pd.DataFrame()
    
    text_list = data_selected.text.values    
                           
    # read the identity set 
    identity_set = read_identity_terms(identity_path)
                    
    for text in text_list:
        #select the same text with all identity terms
        text = [replaceName(text, term, term_new) for term_new in identity_set]
        aux = data.loc[data["text"].isin(text)]
        df_new = pd.concat([df_new,aux])        
    return df_new

def get_result_metric(metric, df, classifiers = ["LR","DT","SVM","XGB","MLP","RF"], std = False):
    #get the name of the clf and feature extractor
    # clf, ext = df.model.split("_")
    df_results = pd.DataFrame(columns=["model","GloVe","FastText","BERT"])
    for clf in classifiers:
        aux_c = df[df["model"].str.contains(clf)][["model",f"{metric}_mean", f"{metric}_std"]]
        
        #mean and std
        if std:
            aux_c[clf] = aux_c[[f"{metric}_mean", f"{metric}_std"]].apply(lambda x : '{:.3f} $\pm$ {:.3f}'.format(float(x[0]),float(x[1])), axis=1) 
        else:
            aux_c[clf] =  aux_c[f"{metric}_mean"]
        aux_c = aux_c[["model", clf]]
        # print(aux_c)
        # aux_c.columns = ["model"] + [c.split("_")[1] for c in aux_c.iloc[0].to_list()]
        new_row = {"model": [clf], 
                   "GloVe": aux_c[aux_c["model"] == f"{clf}_glove"][clf].values,
                   "FastText": aux_c[aux_c["model"] == f"{clf}_fastText"][clf].values,
                   "BERT": aux_c[aux_c["model"] == f"{clf}_bert"][clf].values}
        
        df_results = pd.concat([df_results, pd.DataFrame(new_row)], ignore_index=True)
    
    #transpose results
    df_results = df_results.T
    cols = df_results.iloc[0]
    df_results.columns = cols
    return df_results[1:].reset_index()



