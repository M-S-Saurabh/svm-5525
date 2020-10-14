from sklearn.model_selection import cross_validate
import numpy as np
from itertools import product

from SVM import kernelSVM
from Data import get_hw_data

CV_SPLITS = 10
C_VALS = [ 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]
SIGMA_VALS = [ 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]

def kernel_SVM(dataset: str) -> None:
    if dataset == 'hw2_data_2020':
        (X_train, y_train), (X_test, y_test) = get_hw_data()
    elif dataset == 'mfeat':
        raise IOError('Not Implemented')

    # RBF SVM training
    print("\nTraining SVM with RBF kernel...")
    svm = kernelSVM('RBF')
    validation_error = []; validation_std = []
    training_error = []; training_std =[]
    for C, sigma in product(C_VALS, SIGMA_VALS):
        fit_params = {'C':C, 'sigma':sigma}
        results = cross_validate(svm, X_train, y_train, 
                                cv=CV_SPLITS, fit_params=fit_params, 
                                scoring='accuracy', return_train_score=True)
        validation_rates = 1 - results['test_score']
        training_rates = 1 - results['train_score']
        validation_error.append( validation_rates.mean() )
        validation_std.append( validation_rates.std() )
        training_error.append( training_rates.mean() )
        training_std.append( training_rates.std() )

    # Select C and Sigma which gave the best validation error rate
    index = np.argmin(validation_error)
    fit_params['C'] = C_VALS[index // len(SIGMA_VALS) ]
    fit_params['sigma'] = SIGMA_VALS[index % len(C_VALS)]

    # Train model with this C and Sigma on the entire training set.
    svm = kernelSVM('RBF')
    svm.fit(X_train, y_train, **fit_params)
    test_err = svm.error_rate(X_test, y_test)
    
    # Print out the validation, training and testing error rates.
    round_to = 3
    print("="*25)
    print("Tested C values:", C_VALS)
    print("Tested Sigma values:", SIGMA_VALS)
    print("Mean validation error:", np.round(validation_error, round_to))
    print("Validation error std:", np.round(validation_std, round_to))
    print("~")
    print("Mean training error:", np.round(training_error, round_to))
    print("Training error std:", np.round(training_std, round_to))
    print("~")
    print("Selected C value:", fit_params['C'])
    print("Selected Sigma value:", fit_params['sigma'])
    print("Test error rate:", test_err)

if __name__ == "__main__":
    CV_SPLITS = 10
    C_VALS = [100, 500, 1000, 1200]
    SIGMA_VALS = [ 1, 10, 50, 100]
    # Only runs RBF kernel
    kernel_SVM('hw2_data_2020')
