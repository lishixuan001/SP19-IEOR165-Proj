import numpy as np
import pandas as pd
from sklearn import svm
from os.path import join
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import pickle

import matplotlib.pyplot as plt

##################################################

def linear_svm(value_range):
    results = []
    for C in value_range:
        kf = KFold(n_splits=5, random_state=0, shuffle=False)
        scores = []
        coeffs = []
        for train_index, test_index in kf.split(X):
            
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            clf = SVC(C=C, kernel='linear', gamma='auto')
            model = clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            coeff = clf.coef_
            
            scores.append(score)
            coeffs.append(coeff)
            
        print("C: [{}], Score: [{}]".format(C, np.mean(scores)))
        results.append([C, coeffs, scores])
    return results

##################################################

if __name__ == '__main__':
    data = pd.read_csv("./wholesale-customers.csv")
    X = data.iloc[:, 2:]
    y = data.iloc[:, 0]

    value_range = np.linspace(0.1, 10, 10)
    
    linear_results = linear_svm(value_range)
    
    with open('linear_results.pickle', 'wb') as file:
        pickle.dump(linear_results, file)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# END
    