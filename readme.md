# Assessor feedback mechanism
## 1. Introduction
Recently assessor models have been proposed to explain Artificial System behavior on a given task. In this paper, we propose a methodology which we call "Assessor feedback mechanism" to leverage the power of assessors to correct AI outputs. Assume we have several AI robots to deliver goods around a city, we do not know which robots can do a certain job successfully until employ one of them and get their results. In this case, we would propose to use another AI model called assessor model to assess them on a particular job. Instead learning ML model a success/failure probability introduced by authors of this assessors on a given task:

$$\hat{R}(r|x_j, s_i)\approx Pr(R(x_j, s_i)=r)$$

where $x_j$ - instance features is being to predicted by an ML model with emergent behavior and other related factors to the prediction process characterized by $s_i$ (simple the emergent behavior of the model). 

We would train assessor models on AI models' errors on their (AI model) prediction to correct their errors. For example, for regression task, let's define some notations: a model set $s_i \in S$, and their errors $e_j^i=y_j-\hat{y}_j^i$ ($x_i$ object predicted by model $s_j$ with prediction result $\hat{y}_j^i$, and the prediction error is $e_j^i$) on a set of instances $x_j \in X$ with target values $y_j$. Now, our assessor model is going to learn errors $e_j \in E$. Now, the following assessor model predict the error on new input instance $x_{n+1}$ and system $s_{m+1}$ (a new system can be unseen by the assessor model yet).

$$\hat{e}=\hat{E}(x_j, s_i)$$

To choose the best model on test or production mode, we first predict a error produced by an AI model when predicting a given instance (we predict the error using the assessor model), and we then find a model $s_i$ which produces the lowest error among all exists AI models on a new input $x_j$, and then we predict object $x_j$ by system $s_i$, and finally correct a error $e_j^i$ produced by $s_i$ on instance $x_j$.

$$\hat{y}_j^i=Model_i(x_i)+e_j^i$$

By using the same framework, we achieve doing two staff:

- assess model error on a particular instance if low (or desirable), we can employ a system with the lowest error;
- we can correct the ML model prediction by adding the error produced by the assessor model.
## 2. Datasets

### 2.1 Regression dataset
#### 2.1.1 Blog Feedback
All instances were extracted from Hungarian blog posts to predict the number of comments for a published post, and each instance characterized by 280 features reproduced from raw HTML files. It contains 61 separated files: only one file is for training, and the rest are for testing the model. To get more information refer to [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/304/blogfeedback).
### 2.2 Logistic regression
#### 2.2.1 REJAFADA
REJAFADA (Retrieval of Jar Files Applied to Dynamic Analysis) aims to be used, as benchmark, to check the quality of the detection of Jar malware. It consists of 1996 instances with binary classification: benign and malware. Each object is described by 6825 integer features. To get more information refer to [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset).
#### 2.2.2 Secondary Mushroom
It aims to be used for binary classification into edible and poisonous. It consists of about 61000 objects, each object is characterized by 20 features. We removed columns which has too many missing values, after that we also removed rows with consisting missing values. Eventually, the number dropped to 1412 objects. We also encoded categorical features by one-hot encoding. To get more information refer to [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/860/rejafada).
#### 2.2.3 Adult income
Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset. To get more information refer to [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/2/adult).
#### 2.2.4 Email-Spam
It consists of 4601 email for spam detection, and each object characterized by 57 features extracted from emails. To get more information refer to [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/94/spambase).

### 2.3 Softmax
#### 2.3.1 Maternal Health Risk
It consists of 1014 instances, and each instance described by 6 integer features to decide the risk level of a patient: Low, Middle, High levels. We reproduced the categorical (target) string values as: Low level is 0, middle level is 1, and high level is 2 in the dataset folder. To get more information refer to [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/863/maternal+health+risk).
#### 2.3.2 Students dropout and academic success
A dataset created from a higher education institution (acquired from several disjoint databases) related to students enrolled in different undergraduate degrees, such as agronomy, design, education, nursing, journalism, management, social service, and technologies. The dataset includes information known at the time of student enrollment (academic path, demographics, and social-economic factors) and the students' academic performance at the end of the first and second semesters. The data is used to build classification models to predict students' dropout and academic success. The problem is formulated as a three category classification task, in which there is a strong imbalance towards one of the classes. To get more information refer to [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success).
#### 2.3.3 Dry Bean
Images of 13,611 grains of 7 different registered dry beans were taken with a high-resolution camera. A total of 16 features; 12 dimensions and 4 shape forms, were obtained from the grains. To get more information refer to [archive.ics.uci.edu](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset).


## 3. Run the experiments

### 3.1 Setup the environment

### 3.2 Run experiment scripts

There exist three files whose names started with "exp": _reg.py, _log_reg.py, and _softmax.py for stand experiment on regression tasks, logistic regression for binary classification, and softmax for multi-classification.

#### 3.2.1 Regression
To get the regression results, please run exp_reg.py with following script:

```
python3 ./exp_reg.py --n_systems 60 --error_interval 10 --removable_accuracy 0.4
```
Where:
- n_systems - integer, the number of models to learn their behavior, it must be more than 5 to run TSNE
- error_interval - float, the maximal error interval to assert ml models. Refer to the paper for more information.
- removable_accuracy - float, it aims to be used to drop ml models whose accuracy less than it.

#### 3.2.2 Logistic regression
To get the regression results, please run exp_reg.py with following script:

```
python3 ./exp_log_reg.py --dataset_name rejafada --n_systems 30 --a_acc 0.55 b_acc 0.85 ass_type nn --seed 42
```
Where:
- dataset_name - the name of available datasets to implement the experiment on it. Another dataset can be included in our format.
- n_systems - integer, the number of models to learn their behavior, it must be more than 5 to run TSNE
- a_cc - float, it aims to be used to drop ml models whose accuracy less than it.
- b_cc - float, it aims to be used to drop ml models whose accuracy greater than it.
- ass_type - str, it aims to be used to choose assessor model type. 'nn' - Neural Network, 'svm' - SVM.
- seed - int, it aims to be used to split the chosen dataset into train, assessor (for training the assessor) and test sets.

## 4. Licenses

We leverage 5 datasets downloaded and reproduced from [archive.ics.uci.edu](https://archive.ics.uci.edu) to show our findings based on the proposed methodology, so each dataset has an own license (mainly [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode)) respectively, and there exist links for each them to find more information. All other open-source libraries, including tensorflow, numpy, scikit-learn have their licenses too. The code also is under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode) license.

## 5. If you use our code or contribution, please cite us in your works.

```
@article {musulmon23,
title = 'Assessor feedback mechanism',
authors = 'Musulmon, Lolaev and Anand, Paul',
}

```