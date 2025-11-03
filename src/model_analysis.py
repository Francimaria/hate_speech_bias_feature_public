#Models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from gensim.models import Word2Vec
import scipy

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import model_bias_analysis

import numpy as np
import pandas as pd

import re
import copy

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


def get_classifiers(random_state=42):
    classifiers = {
        'LR': LogisticRegression(random_state=random_state, multi_class='auto', solver='liblinear'),
        'DT': DecisionTreeClassifier(random_state=random_state),
        'SVM': SVC(random_state=random_state, probability=True),
        'XGB': xgb.XGBClassifier(learning_rate=0.01, random_state=random_state),
        'MLP': MLPClassifier(random_state=random_state,hidden_layer_sizes=(100,), batch_size=20, max_iter=20, verbose=100, solver='adam'),
        'RF': RandomForestClassifier(criterion='gini', n_jobs=-1, random_state=random_state)
        }
    return classifiers
  
def get_parameters():
	params = {
        'LR': {
            'penalty': ['l1', 'l2']
        },
        'DT': {
            'criterion': ['gini', 'entropy']
        },
        'SVM':{
            'kernel': ['linear', 'rbf']
        },
        'RF': {
            'n_estimators': [50, 100]
        },
        'XGB': {
            'n_estimators': [50, 100]
        },
        'MLP': {
            'activation': ['relu', 'logistic']
        }
	}
    
	return params

        
def get_param_classifier(classifier):
	params = get_parameters()
	return list(ParameterGrid(params[classifier]))

def fit_params(classifier, params, train, class_train):
	clf = classifier.set_params(
	    params[classifier]
	)
	return clf.fit(train, class_train)


def fit_all(train, class_train, classifier=None):
	estimators = []
	params_clf = get_param_classifier(classifier)
	total_params = len(params_clf)
	k = 1
	classifiers_ = get_classifiers()

	for params in params_clf:
		classifiers = copy.deepcopy(classifiers_)
		clf = classifiers[classifier]
		clf.set_params(**params)
		X = train.to_numpy() if isinstance(train, pd.DataFrame) else train
		estimators.append(clf.fit(X, class_train))
		print("Done {} of {}".format(k, total_params))
		k += 1
	return estimators

def best_estimator(estimators, val, class_val):
	best_score = 0
	best_estimator = None
	val = val.to_numpy() if isinstance(val, pd.DataFrame) else val
	for e in estimators:		
		y_pred = e.predict(val)
		score = f1_score(class_val, y_pred, average='macro')#e.score(val, class_val)
		print("\n SCORE: ", score)
		if best_score < score:
			best_estimator = e
			best_score = score
	return best_estimator.get_params()


def fit_all_folds(train, class_train, classifier=None, folds=5):
    estimators = {}
    params_clf = get_param_classifier(classifier)
    total_params = len(params_clf)
    
    classifiers_ = get_classifiers()
    for f in range(folds):
        print("FOLD:", f)
        estimators_param = []		
        k = 1
        for params in params_clf:
            classifiers = copy.deepcopy(classifiers_)
            clf = classifiers[classifier]
            clf.set_params(**params)
            X = train[f].to_numpy() if isinstance(train[f], pd.DataFrame) else train[f]
            estimators_param.append(clf.fit(X, class_train[f]))
            print("Done {} of {}".format(k, total_params))
            k += 1
        print(estimators_param)
        estimators[f] = estimators_param
    return estimators

def best_estimator_folds(estimators, val, class_val, folds=5):
	
    scores = np.zeros((folds, len(estimators[0])))
    for f in range(folds):
        val[f] = val[f].to_numpy() if isinstance(val[f], pd.DataFrame) else val[f]	
        for idx, e in enumerate(estimators[f]):
            y_pred = e.predict(val[f])
            scores[f][idx] = f1_score(class_val[f], y_pred, average='macro')#e.score(val, class_val)
    print(np.mean(scores, axis=0))
    idx_best_est = np.argmax(np.mean(scores, axis=0))
    best_estimator = estimators[0][idx_best_est]
    print(best_estimator)
    return best_estimator.get_params()
        
def add_subgroup_columns_from_text(df, text_column="text", subgroups=[], expect_spaces_around_words=True):
    """Adds a boolean column for each subgroup to the data frame.
            New column contains True if the text contains that subgroup term.
            Args:
            df: Pandas dataframe to process.
            text_column: Column in df containing the text.
            subgroups: List of subgroups to search text_column for.
            expect_spaces_around_words: Whether to expect subgroup to be surrounded by
                spaces in the text_column.  Set to False to for languages which do not
                use spaces.
            """
    for term in subgroups:
        if expect_spaces_around_words:
            df[term] = df[text_column].apply(
                    lambda x: bool(re.search('\\b' + term + '\\b', x,
                                        flags=re.UNICODE | re.IGNORECASE)))
        else:
            df[term] = df[text_column].str.contains(term, case=False)


def get_bias_metrics(debias_dataset, identity_terms, model_name, scores):
        madlibs = pd.DataFrame()
        # debias dataset used to evalatuate the model
        madlibs["text"] = debias_dataset["text"]
        madlibs["label"] = debias_dataset["label"]

        family_name = model_name
        scores = pd.DataFrame(scores).replace(2,1)
        madlibs[family_name] = scores

        madlibs_terms = list(identity_terms)
        madlibs[madlibs_terms] = debias_dataset[madlibs_terms]

        all_model_families_names = [[family_name]]

        #threshold of 0.5
        _model_eers_madlibs = 0.5

        fnr = model_bias_analysis.per_subgroup_fnr_diff_from_overall(madlibs,madlibs_terms, 
                                                                     all_model_families_names, _model_eers_madlibs, False)
      #model_bias_analysis.per_subgroup_fnr_diff_from_overall(madlibs, madlibs_terms, all_model_families_names, _model_eers_madlibs, False)
        fpr = model_bias_analysis.per_subgroup_tnr_diff_from_overall(madlibs, madlibs_terms, all_model_families_names, _model_eers_madlibs, False)
        # # Get AUC family metrics
        madlibs_results = pd.DataFrame()
        madlibs['label_bool'] = madlibs.apply(lambda row: row.label == 1, axis=1)
        #model_bias_analysis.
        madlibs_results = model_bias_analysis.compute_bias_metrics_for_model(madlibs, madlibs_terms, family_name, 'label_bool', False)

        return fnr.fnr_equality_difference[0], fpr.tnr_equality_difference[0], madlibs_results



def calculate_all_metrics(clf_name, debias_dataset, identity_terms, fold, y_true, y_pred, y_proba, y_pred_bias):
    # calculate classification accuracy
    acc = accuracy_score(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
       
    f_score = f1_score(y_true, y_pred, average="macro")

    fned, fped, madlibs_results = get_bias_metrics(debias_dataset, identity_terms, clf_name, y_pred_bias)
    subgroup = np.mean(madlibs_results[f"{clf_name}_subgroup_auc"])
    bpsn = np.mean(madlibs_results[f"{clf_name}_bpsn_auc"])
    bnsp = np.mean(madlibs_results[f"{clf_name}_bnsp_auc"])

    fold_name = fold
    if fold != None:
         fold_name =  f"F{fold}"
  
    results = {"model": clf_name,"Fold":fold_name, "accuracy": acc, "f1_score": f_score, "auc": auc,
               "fned": (fned/len(identity_terms)), "fped": (fped/len(identity_terms)), "subgroup": subgroup,"bpsn": bpsn,"bnsp": bnsp}
        
    # # Save predictions
    # if save:
    #     save_predicted_values(f"{path}{set_names[1]}.csv", column_name, y_pred)
    #     save_predicted_values(f"{path}{set_names[1]}_bias.csv", column_name, y_pred_bias)

    #     for i in range(len(y_probahat[0])):
    #         save_predicted_values(f"{path}{set_names[1]}_proba.csv", f"{column_name}-{str(i)}", y_probahat[:,i])
    #     for i in range(len(yprobahat_bias[0])):
    #         save_predicted_values(f"{path}{set_names[1]}_bias_proba.csv",  f"{column_name}-{str(i)}", yprobahat_bias[:,i])
    return results