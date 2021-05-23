import numpy as np
from cvxopt import matrix, solvers
import time

class kernelSVM:
    def __init__(self, kernel='Linear'):
        super().__init__()
        self.kernel = kernel
    # The following two methods are implemented so that Sklearn cross_val functions can support this class
    def get_params(self, deep=True):
        return {'kernel': self.kernel}

    def set_params(self, **params):
        self.kernel = params['kernel']
        return self

    @staticmethod
    def linearKernel(X1, X2):
        return X1 @ X2.T

    @staticmethod
    def rbfKernel(X1, X2, sigma, allocate=False):
        N, D = X1.shape
        if (D < 10 and N < 7000) or allocate:
            # Allocates too much memory N * N * D
            G = ( (X1[:,None,:] - X2[None,:,:])**2.0 ).sum(axis=-1) # ||x - x1||^2
        else:
            G = np.zeros((X1.shape[0], X2.shape[0]))
            for i, x1 in enumerate(X1):
                G[i] = np.sum((X2 - x1)**2.0, axis=-1)
        G = np.exp(G / (-2 * sigma**2))
        return G

    def gram_matrix(self, X1, X2):
        if self.kernel == 'Linear':
            G = self.linearKernel(X1, X2)
        elif self.kernel == 'RBF':
            start_time = time.time()
            G = self.rbfKernel(X1, X2, self.sigma)
            # print("Time taken for computing gram matrix:",time.time()-start_time)
        return G

    def fit(self, X, y, C=1, sigma=None, thres=1e-5):
        N, D = X.shape
        self.sigma = sigma
        K = self.gram_matrix(X, X)
        P = matrix( np.outer(y, y) * K )
        q = matrix( -1 * np.ones(N) )
        G = matrix( np.vstack(( -1*np.eye(N), np.eye(N) )) )
        h = matrix( np.concatenate([ np.zeros(N), np.full(N,C) ]) )
        A = matrix( y, (1,N), 'd')
        b = matrix( 0.0 )

        # solve QP problem
        solution = solvers.qp(P, q, G, h, A, b)
        lambda_star = np.array(solution['x']).flatten()

        # indices where lambda > 0 (support vectors)
        indices = np.where(lambda_star > thres)[0] 
        self.lambda_star = lambda_star[indices]
        self.X_sv = X[indices]
        self.y_sv = y[indices]
        K_sv = K[indices[:,None], indices] # Gram matrix of only the support vectors
        print("number of support vectors:{}".format(len(indices)))

        # Intercept
        self.b = np.sum(self.y_sv) - np.sum(self.lambda_star * self.y_sv * K_sv)
        self.b /= len(self.lambda_star)
        return

    def predict(self, X):
        K = self.gram_matrix(X, self.X_sv)
        preds = (self.lambda_star * self.y_sv * K).sum(axis=-1) + self.b
        return np.sign(preds)

    def error_rate(self, X, y_true):
        preds = self.predict(X)
        error_rate = np.count_nonzero(preds != y_true) / len(y_true)
        return error_rate

class multiclassSVM(kernelSVM):
    def __init__(self, kernel='Linear'):
        super().__init__()
        self.kernel = kernel

    def relabel(self, y, label):
        y_new = y.copy()
        y_new[y == label] = 1
        y_new[y != label] = -1
        return y_new

    def fit(self, X, y, C=1, sigma=None):
        self.models = []
        for label in range(y.max()+1):
            y_cls = self.relabel(y, label)

            # Train svm for this class
            if np.count_nonzero(y_cls == 1) == 0:
                print("here")
            super().fit(X, y_cls, C, sigma, thres=1e-10)
            X_sv = self.X_sv.copy()
            y_sv = self.y_sv.copy()
            lambda_star = self.lambda_star.copy()
            b = self.b
            
            # Store the model
            self.models.append({
                'X_sv': X_sv,
                'y_sv': y_sv,
                'lambda_star': lambda_star,
                'b': b
            })
        return

    def predict(self, X):
        scores = []
        for model in self.models:
            if len(model['lambda_star']) == 0:
                scores.append(np.zeros(X.shape[0]))
                continue
            K = self.gram_matrix(X, model['X_sv'])
            scores.append((model['lambda_star'] *model['y_sv'] * K).sum(axis=-1) + model['b'])
        scores = np.vstack(scores)
        preds = np.argmax(scores, axis=0)
        return preds


if __name__ == "__main__":
    # Testing RBF gram matrix function.
    print("Testing RBF gram matrix function")
    X1 = np.random.randint(10, size=(8,3))
    X2 = np.random.randint(10, size=(5,3))
    print(kernelSVM.rbfKernel(X1, X2, 10, True) == kernelSVM.rbfKernel(X1,X2, 10, False))