import numpy as np
import os

# HW data
def get_hw_data():
    data_filename = './hw2_data_2020.csv'
    hw_data = np.genfromtxt(data_filename, delimiter=',')
    np.random.shuffle(hw_data)
    X, y = hw_data[:,:-1], hw_data[:,-1].astype(int)
    y[y == 0] = -1
    return split_test_data(X, y)

# mfeat data
def get_mfeat_data():
    # Importing data
    data_folder = 'mfeat'
    filelist = os.listdir(data_folder)
    ignore_files=['mfeat.info', '.', '..']
    mfeat_data = []; 
    for filename in filelist:
        if (filename in ignore_files) or (not filename.startswith('mfeat')): continue
        mfeat_data.append( np.genfromtxt( os.path.join(data_folder, filename) ) )
    X = np.hstack(mfeat_data)

    # Labelling (First 200 are class 0, next 200 are 1 and so on)
    num_classes = 10
    class_count = 200
    y = np.repeat(np.arange(num_classes), class_count)

    # Splitting
    return class_wise_split(X, y)
    

# Negative to use full training data
NUM_TRAIN_SAMPLES = 2000

def split_test_data(X, y, train_percent=80):
    # print("Class distribution:",np.bincount(y))
    N, D = X.shape
    cutoff = train_percent*N//100
    if NUM_TRAIN_SAMPLES < 0 or NUM_TRAIN_SAMPLES >= N:
        X_train, y_train = X[:cutoff], y[:cutoff]
    else:
        X_train, y_train = X[:NUM_TRAIN_SAMPLES], y[:NUM_TRAIN_SAMPLES]
    # print("Training dataset shape:", X_train.shape, y_train.shape)
    X_test, y_test = X[cutoff:], y[cutoff:]
    # print("Testing dataset shape:", X_test.shape, y_test.shape)
    return (X_train, y_train), (X_test, y_test)

def class_wise_split(X, y, train_percent=80):
    N, D = X.shape
    num_classes = len(np.unique(y))
    class_count = N // num_classes

    class_indices = np.random.choice(class_count, train_percent*class_count//100, replace=False)
    class_train_mask = np.zeros(class_count).astype(bool)
    class_train_mask[class_indices] = True

    dataset_train_mask = np.tile(class_train_mask, num_classes)
    dataset_test_mask = np.invert(dataset_train_mask)

    X_train, y_train = X[dataset_train_mask], y[dataset_train_mask]
    print("Training dataset shape:", X_train.shape, y_train.shape)
    shuffler = np.random.permutation(len(y_train))
    X_train, y_train = X_train[shuffler], y_train[shuffler]

    X_test, y_test = X[dataset_test_mask], y[dataset_test_mask]
    print("Testing dataset shape:", X_test.shape, y_test.shape)
    shuffler = np.random.permutation(len(y_test))
    X_test, y_test = X_test[shuffler], y_test[shuffler]

    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    print("Testing data files...")
    get_mfeat_data()
