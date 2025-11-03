**Gender bias propagation on hate speech detection models: an analysis at feature-level**
========================================================================================================================

**Unbiased dataset**:
========================================================================================================================

The idea is to build sentences changing only the identity term, for instance, "**Women** should be protected" and "**Men** should be protected".
We define several templates filled with the terms described in bias_data. Thus, each identity term occurs in the same context. 

We save the dataset as pickle file.

<!-- The synthetic test set created comprises 1,248 samples, of which 648 are non-hateful, and 600 are hateful, and all identity terms appear in the same contexts.  -->

## 1. Running local

### Requirements and installation:

This code requires Python >= 3.6.5, Zeugma, Scikit-learn, Pandas, NLTK, Matplotlib. Environment can be installed using the following command:

```bash
$ cd path_fold
```

```bash
$ pip install -r requirements.txt
```

### Running feature extraction
```bash
$ python feature_extraction.py
```

* For each dataset

```bash
$ python feature_extraction.py --dataset_names WH
```

```bash
$ python feature_extraction.py --dataset_names HE
```

### Running train classifiers

```bash
$ python train_clf.py
```

* For each dataset

```bash
$ python train_clf.py --dataset_names WH
```

```bash
$ python train_clf.py --dataset_names HE
```


### Running param select

The idea is to evaluate a set of parameter combinations in the proposed strategy and select the setting that presents the best performance based on unintended bias metrics using the validation set.

```bash
$ python train_clf_param_select.py
```

* For each dataset

```bash
$ python train_clf_param_select.py --dataset_names WH
```

```bash
$ python train_clf_param_select.py --dataset_names HE
```

## 2. Running with sbatch on the cluster

1. running code

```bash
$ sbatch run_job.sh
```

2. see result

```bash
$ cat job_output.txt
```

or 

```bash
$ tail -f job_output.txt
```




