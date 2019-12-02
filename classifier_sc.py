import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

basic_params_dict = {
    
    'X':['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
    'y':['species_id'],
    'seed': 123,
    'test_size': 0.3
}
model_type=1

classifier_config_dict = {

    # Classifiers
   
    'RandomForestClassifier': RandomForestClassifier(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'LogisticRegression':LogisticRegression(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    
  }

regression_config_dict={
     # Regressions
    'LinearRegression':LinearRegression(),
    'RandomForestRegressor':RandomForestRegressor()
}

gridsearch_params_dict={
    # Hyperparameters
    'RandomForestClassifier':{'n_estimators': range(5,20,2),
              'max_features' : ['auto', 'sqrt'],
              'max_depth' : [10,20,30,40],
              'min_samples_split':[2,5,10],
              'min_samples_leaf':[1,2,4]},
    'DecisionTreeClassifier':{'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]},
    'LogisticRegression':{"C":np.logspace(-3,3,7), "penalty":["l1","l2"]},
    'SVM':[{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}],
    'KNN':{'n_neighbors': [5,7,9,11,13,15]},
    'LinearRegression':{'fit_intercept':[True,False], 'normalize':[True,False], 'copy_X':[True, False]},
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'KNNClassifier':{ 'n_neighbors': [5,10,15,20], 'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']},
    'RandomForestRegressor' : {'max_features' : ['auto', 'sqrt'], 'min_samples_split' : [2,5,10], 'n_estimators' : range(5,20,2),'min_samples_leaf' : [1,2,4]}
}