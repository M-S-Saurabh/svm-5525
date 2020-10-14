import pickle
import numpy as np

from Data import get_mfeat_data
from SVM import multiclassSVM

(X_train, y_train), (X_test, y_test) = get_mfeat_data()

with open('multi-model.pkl', 'rb') as pkl:
    model = pickle.load(pkl)

train_preds = model.predict(X_train)
print("Train: Correct predictions:{} / {}".format(np.count_nonzero(train_preds == y_train), len(y_train)))

test_preds = model.predict(X_test)
print("Test: Correct predictions:{} / {}".format(np.count_nonzero(test_preds == y_test), len(y_test)))