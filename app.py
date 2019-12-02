
from flask import Flask, request, jsonify, render_template
import pickle
from classifier_sc import classifier_config_dict,basic_params_dict,gridsearch_params_dict,regression_config_dict,model_type
import time
import pandas as pd               
import numpy as np
import pickle
import sys

import plotly
from plotly.data import iris
import sklearn
from sklearn.model_selection import train_test_split   #splitting data
from pylab import rcParams
from sklearn.linear_model import LinearRegression         #linear regression
from sklearn.metrics.regression import mean_squared_error #error metrics
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
##
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import ast
dataset=pd.read_csv("dataset.csv")
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/performGCV',methods=['POST'])
def performGCV():
    
    dataset=pd.read_csv("dataset.csv")
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]
    model_type=int(int_features[0])
    basic_params_dict['X']=ast.literal_eval(int_features[1])
    basic_params_dict['y']=ast.literal_eval(int_features[2])
    basic_params_dict['test_size']=float(int_features[3])
  
    if model_type==1:
        b=BaseModelHelper(basic_params_dict,classifier_config_dict,gridsearch_params_dict,model_type)
    else:
        b=BaseModelHelper(basic_params_dict,regression_config_dict,gridsearch_params_dict,model_type)
       
    metrics= b.model_build()
      
    return render_template('output.html', prediction_text=metrics.to_html())
   


@app.route('/uploader', methods = ['POST'])
def upload_file():
   print("Hello")
   if request.method == 'POST':
   
      file = request.files['file']
      file.save('dataset.csv')
      return render_template("index.html")
# Create a class to perform base model operations
class BaseModelHelper:
    def __init__(self,base_param,base_model,base_gridsearch_params,model_type):
        self.base_param = base_param
        self.base_model = base_model
        self.base_gridsearch_params=base_gridsearch_params
        #Initialize X
        self.X = self.base_param['X']
        #Initialize y
        self.y = self.base_param['y']
        #set random seed
        self.random_state = self.base_param['seed']
        #Set test_size
        self.test_size = self.base_param['test_size']
        #set base model params
        self.base_model = self.base_model
        self.base_gridsearch_params=self.base_gridsearch_params
        self.model_type=model_type
     
    #Function to standardize columns
    def normalize_columns(self):
        
        X=dataset[self.X]
        #Scale the values
        scaler = StandardScaler()
        scaler.fit(X)

        # Scale and center the data
        fdf_normalized = scaler.transform(X)

        # Create a pandas DataFrame
        fdf_normalized = pd.DataFrame(data=fdf_normalized, index=X.index, columns=X.columns)
        return fdf_normalized
    
    #Function to perform train test split
    def train_test_split_base(self,X_norm):
        self.X=X_norm
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,dataset[self.y],random_state=self.random_state,test_size=self.test_size)
    
  
    
    #Building model
    def model_build(self):
        X_norm = self.normalize_columns()
        self.train_test_split_base(X_norm)
            
        
        metrics=[]
        for key in self.base_model:
            Time=None
           
            model=self.base_model[key]
            model.fit(self.X_train,self.y_train)
            y_pred_test = model.predict(self.X_test)
            
                      
            #gridsearch cv
            #random_grid = self.grid_params(str(key))
            random_grid = self.base_gridsearch_params[str(key)]
            
            rf_gs = GridSearchCV(model, random_grid, cv = 3, n_jobs=-1, verbose=2)

            rf_gs.fit(self.X_train,self.y_train)
            y_pred_test = rf_gs.predict(self.X_test)
            Time = time.process_time()
            if self.model_type==1:
                train_accuracy=accuracy_score(self.y_train, rf_gs.predict(self.X_train))
                test_accuracy=accuracy_score(self.y_test, y_pred_test)
                #columns = ['Name','Train Accuracy', 'Test Accuracy', 'Parameters','Time']
            else:
                train_accuracy=mean_absolute_error(self.y_train, rf_gs.predict(self.X_train))
                test_accuracy=mean_absolute_error(self.y_test, y_pred_test)
                #columns = ['Name','Train MAE', 'Test MAE', 'Parameters','Time']
                
            
            
            metrics.append([key,train_accuracy,test_accuracy,rf_gs.best_params_,Time])
            
            if self.model_type==1:
                results=pd.DataFrame(metrics, columns =['Name','Train Accuracy', 'Test Accuracy', 'Parameters','Time']) 
            else:
                results=pd.DataFrame(metrics, columns= ['Name','Train MAE', 'Test MAE', 'Parameters','Time']) 
        return results
    
    
if __name__ == "__main__":
    app.run(debug=True)