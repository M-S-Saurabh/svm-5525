from sklearn.model_selection import cross_validate, StratifiedKFold
import numpy as np
from itertools import product
import pickle

from Data import get_mfeat_data
from SVM import multiclassSVM

CV_SPLITS = 10
C_VALS = [ 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
SIGMA_VALS = [ 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]

def train_test_SVM(dataset, kernel):
    if dataset == 'hw2_data_2020':
        raise IOError('Not Implemented')
    elif dataset == 'mfeat':
        (X_train, y_train), (X_test, y_test) = get_mfeat_data()

    # SVM training
    print("\nTraining SVM with {} kernel...".format(kernel))
    global SIGMA_VALS, C_VALS
    if kernel == 'Linear':
        SIGMA_VALS = [None]
    validation_error = []; validation_std = []
    training_error = []; training_std =[]
    for C, sigma in product(C_VALS, SIGMA_VALS):
        svm = multiclassSVM(kernel)
        fit_params = {'C':C, 'sigma':sigma}
        skf = StratifiedKFold(n_splits=CV_SPLITS)
        skf.get_n_splits(X_train, y_train)
        results = cross_validate(svm, X_train, y_train, 
                                cv=skf, fit_params=fit_params, 
                                scoring='accuracy', return_train_score=True)
        validation_rates = 1 - results['test_score']
        training_rates = 1 - results['train_score']
        validation_error.append( validation_rates.mean() )
        validation_std.append( validation_rates.std() )
        training_error.append( training_rates.mean() )
        training_std.append( training_rates.std() )

    # Select C and Sigma which gave the best validation error rate
    fit_params = {}
    index = np.argmin(validation_error)
    if kernel == 'Linear':
        fit_params['C'] = C_VALS[index]
        fit_params['sigma'] = None
    elif kernel == 'RBF':
        fit_params['C'] = C_VALS[index // len(SIGMA_VALS) ]
        fit_params['sigma'] = SIGMA_VALS[index % len(C_VALS)]

    # Train model with this C and Sigma on the entire training set.
    svm = multiclassSVM(kernel)
    svm.fit(X_train, y_train, **fit_params)
    # with open('multi-model.pkl', 'wb') as pkl:
    #     pickle.dump(svm,pkl)
    test_err = svm.error_rate(X_test, y_test)
    
    # Print out the validation, training and testing error rates.
    round_to = 3
    print("\nMulti-SVM with {} kernel".format(kernel))
    print("="*50)
    print("Tested C values:", C_VALS)
    print("Tested Sigma values:", SIGMA_VALS)
    print("Mean validation error:", np.round(validation_error, round_to))
    print("Training error std:", np.round(validation_std, round_to))
    print("~")
    print("Mean training error:", np.round(training_error, round_to))
    print("Training error std:", np.round(training_std, round_to))
    print("~")
    print("Selected C value:", fit_params['C'])
    print("Selected Sigma value:", fit_params['sigma'])
    print("Test error rate:", test_err)

def multi_SVM(dataset: str) -> None:
    np.random.seed(0)
    # Can run both Linear and RBF but takes too long. Suggested: comment out the one you dont want to run.
    train_test_SVM(dataset, "Linear")
    train_test_SVM(dataset, "RBF")
    
if __name__ == "__main__":
    CV_SPLITS = 10
    C_VALS = [ 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
    SIGMA_VALS = [1000]
    multi_SVM('mfeat') # See last function in this file.

