import pandas as pd
import random
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

def readData (df):
    r,c = df.shape
    y = df[['class']]
    X = df.iloc[:,df.columns !='class']
    features = X.columns.tolist()  

    acc_RF= list()

    time_RF =list()
    
    for i in range (10):
        Train_x, Test_x, Train_y, Test_y = train_test_split(X,  y, stratify=y, test_size=0.4, random_state = 42) 

        start = datetime.now()
        
        rf_model = RandomForestClassifier(n_estimators=10,max_features= int(math.sqrt(c))+1)

        rf_model.fit(Train_x,Train_y.values.ravel())
        pred_y = rf_model.predict(Test_x)
        end = datetime.now() - start
        time_RF.append(end)
        
        acc_RF.append(metrics.accuracy_score(Test_y, pred_y))
        
        print ("Thoi gian Random forest: ", np.mean(time_RF))
    results =[]

    results.append(acc_RF)

    names = ['Random forest']
    fig = plt.figure()
    plt.boxplot(results, labels=names)
    plt.ylabel('Accuracy') 
    plt.show()

def run():
    df = pd.read_csv('/home/totoo/Project/data-mining/Brain.csv' )  
    df.columns.values[0] = "class"   
    readData(df)    
    

if __name__ == "__main__":
    run()