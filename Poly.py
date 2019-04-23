import numpy as np
import pandas as pd
from sklearn import svm
from os.path import join
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
import pickle

import matplotlib.pyplot as plt

##################################################

def poly_svm(value_range):
    X = scale(data.iloc[:, 2:])
    X = pd.DataFrame(X)
    results = []
    for C in value_range:
        kf = KFold(n_splits=5, random_state=0, shuffle=False)
        scores = []
        for train_index, test_index in kf.split(X):
            
            print("---> Processing")
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            clf = SVC(C=C, kernel='poly', coef0=0, degree=3, gamma='auto')
            model = clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            
            scores.append(score)
            
        print("C: [{}], Score: [{}]".format(C, np.mean(scores)))
        results.append([C, scores])
    return results

##################################################

if __name__ == '__main__':
    data = pd.read_csv("./wholesale-customers.csv")
    X = data.iloc[:, 2:]
    y = data.iloc[:, 0]
    
    value_range = np.linspace(1, 50, 20)
    
    poly_results = poly_svm(value_range)
    
    with open('poly_results.pickle', 'wb') as file:
        pickle.dump(poly_results, file)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# END
    