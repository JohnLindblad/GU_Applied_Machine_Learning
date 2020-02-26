# B, C: Sparse, postponed w scaling. not BLAS SVM.
from aml_perceptron import LinearClassifier
from scipy.linalg.blas import ddot, dscal, daxpy
from scipy.sparse import csr_matrix
import numpy as np
import torch

class PegasosSVC_smoothedHinge(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
    """

    def __init__(self, n_iter=20, lam=0.5):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter
        self.lam = lam

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        t=0

        # Pegasos algorithm:
        for i in range(self.n_iter):
            for x, y in zip(X, Ye):
                
                #computing the new learning rate
                t = t+1
                eta = 1/(self.lam*t)

                # Compute the output score for this instance.
                score = x.dot(self.w)

                # Update the weights
                if y*score <= 0:
                    self.w += eta
                elif (y*score > 0) and (y*score < 1):
                    self.w -= (y*score-1)

class PegasosRudimentaryClassifier(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
    """

    def __init__(self, n_iter=20, lam=0.5, type='SVC'):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter
        self.lam = lam

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        t=0

        # Pegasos algorithm:
        for i in range(self.n_iter):
            for x, y in zip(X, Ye):
                
                #computing the new learning rate
                t = t+1
                eta = 1/(self.lam*t)

                # Compute the output score for this instance.
                score = x.dot(self.w)

                # Update the weights
                if type=='SVC':
                    if y*score < 1:
                        self.w = (1-eta*self.lam)*self.w + (eta*y)*x
                    else:
                        self.w = (1-eta*self.lam)*self.w
                elif type=='LR':
                    self.w = (1-eta*self.lam)*self.w + eta*(y/(1+np.exp(y*score)))*x

class Pegasos_Sparse(LinearClassifier):
    '''
    Our linear classifier implementation, made for using sparse matrices.
    Since the BLAS functions don't work with the sparse matrices from scipy, we wrote this separately.
    We're postponing scaling self.w in both of them, since that's pretty separate.
    '''

    def __init__(self, n_iter=20, lam=0.5, type='SVC'):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter
        self.lam = lam
        if type=='SVC':
            self.update_weights = self.update_weights_SVC
        elif type=='LR':
            self.update_weights = self.update_weights_LR

    @classmethod
    def add_sparse_to_dense(cls, x, w, factor):
        """
        Adds a sparse vector x, scaled by some factor, to a dense vector.
        This can be seen as the equivalent of w += factor * x when x is a dense
        vector.
        """
        w[x.indices] += factor * x.data

    @classmethod
    def sparse_dense_dot(cls, x, w):
        """
        Computes the dot product between a sparse vector x and a dense vector w.
        """
        return np.dot(w[x.indices], x.data)
    
    def update_weights_SVC(self, eta, x, w, y, score, a):
        if y*score < 1:
            self.add_sparse_to_dense(x, w, factor=eta*y/a)
    
    def update_weights_LR(self, eta, x, w, y, score, a):
        self.add_sparse_to_dense(x, w, factor=eta*(y/(1+np.exp(y*score)))/a)
    
    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        assert isinstance(X, csr_matrix)

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        t=0
        a=1
        # Pegasos algorithm:
        for _ in range(self.n_iter):
            for x, y in zip(X, Ye):
                # Compute the new learning rate
                t = t+1
                eta = 1/(self.lam*t)

                # Compute the output score for this instance
                score = a*self.sparse_dense_dot(x, self.w)

                # Scale a
                a *= (1-eta*self.lam)

                # Update the weights
                self.update_weights(eta, x, self.w, y, score, a)
                    
            # Catch up on scaling w, reset a
            dscal(a, self.w)
            a=1

class Pegasos_BLAS(LinearClassifier):
    '''
    Our linear classifier implementation, using the BLAS functions. 
    We're also postponing scaling self.w.
    It's mostly copy/paste from the reference code.
    '''

    def __init__(self, n_iter=20, lam=0.5, type='SVC'):
        """
        The constructor takes a type parameter, which can be either SVC or LR. 
        This only affects the loss function.
        """
        self.n_iter = n_iter
        self.lam = lam
        if type=='SVC':
            self.update_weights = self.update_weights_SVC
        elif type=='LR':
            self.update_weights = self.update_weights_LR
    
    def update_weights_SVC(self, eta, x, w, y, score, a):
        if y*score < 1:
            daxpy(x, w, a=eta*y/a)
    
    def update_weights_LR(self, eta, x, w, y, score, a):
        daxpy(x, w, a=eta*(y/(1+np.exp(y*score)))/a)

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        t=0
        a = 1
        # Pegasos algorithm:
        for i in range(self.n_iter):
            for x, y in zip(X, Ye):
                # Compute the new learning rate
                t = t+1
                eta = 1/(self.lam*t)

                # Compute the output score for this instance
                score = a*ddot(x, self.w)

                # Scale a
                a *= (1-eta*self.lam)

                # Update the weights
                self.update_weights(eta, x, self.w, y, score, a)
            
            # Catch up on scaling w, reset a
            dscal(a, self.w)
            a=1

class Torch_Classifier:
    '''
    Like the Pegasos classifiers, we're using the same class for both SVC and LR. 
    Change between them by passing type='SVC' or type='LR' to the constructor.
    '''
    def __init__(
        self, 
        n_iter=15, 
        eta=0.1, 
        batch_size=10, 
        opt=None, 
        type='SVC', 
        use_cuda=False, 
        verbose=False):
        """
        n_iter:     number of iterations (epochs) to train for
        eta:        The learning rate used for the default Adam optimizer if opt=None
        batch_size: size of batches used.
        opt:        function that initializes an optimizer from torch.optim.
        type:       SVC or LR
        use_cuda:   Whether to use cuda. Ignored if CUDA is not available.
        verbose:    Whether to print the loss every iteration.
        """
        self.n_iter = n_iter
        self.eta = eta
        self.batch_size = batch_size
        self.opt = opt
        self.type=type
        self.verbose = verbose
        # Set torch device
        if use_cuda and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        if type != 'SVC' and type != 'LR':
            raise Exception("Unknown classifier type: {}".format(type))

    def find_classes(self, Y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of
        classes is not 2, an error is raised.
        """
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class = classes[1]
        self.negative_class = classes[0]

    def encode_outputs(self, Y):
        """
        A helper function that converts all outputs to +1 or -1.
        """
        return np.array([1 if y == self.positive_class else -1 for y in Y])

    def fit(self, X, Y):
        
        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        X = X.toarray()

        Ye = self.encode_outputs(Y)

        n_instances, n_features = X.shape
        assert X.shape[0] == Ye.shape[0]
        
        # we need to "wrap" the NumPy arrays X and Y as PyTorch tensors
        Xt = torch.tensor(X, dtype=torch.float, device=self.device)
        Yt = torch.tensor(Ye, dtype=torch.float, device=self.device)

        # initialize the weight vector to all zeros
        self.w = torch.zeros(n_features, requires_grad=True, dtype=torch.float, device=self.device)
        self.history = []

        if self.opt is None:
            optimizer = torch.optim.Adam([self.w], lr=self.eta)
        else:
            optimizer = self.opt(self.w)

        tol = 1e-5
        iteration = 0
        
        converged = False
        # While not enough iterations or not converged
        while iteration < self.n_iter and not converged:
            iteration += 1
            total_loss = 0
            
            for batch_start in range(0, n_instances, self.batch_size):
                batch_end = batch_start + self.batch_size

                # pick out the batch again, as in the other notebook
                Xbatch = Xt[batch_start:batch_end, :]
                Ybatch = Yt[batch_start:batch_end]
            
                # mv = matrix-vector multiplication in Torch
                G = Xbatch.mv(self.w)
                score = Ybatch*G

                if self.type=='SVC':
                    # Calculate hinge loss
                    loss_batch = torch.sum(1-score[score<1]) / n_instances
                elif self.type=='LR':
                    # Calculate log loss
                    loss_batch = torch.sum(torch.log(1+torch.exp(-score))) / n_instances
                    
                # we sum up the loss values for all the batches.
                # the item() here is to convert the tensor into a single number
                total_loss += loss_batch.item()
                
                # reset all gradients
                optimizer.zero_grad()                  

                # compute the gradients for the loss for this batch
                loss_batch.backward()

                # for SGD, this is equivalent to w -= learning_rate * gradient as we saw before
                optimizer.step()

            self.history.append(total_loss)

            # Determine if converged
            if iteration >= 2:
                converged = self.history[len(self.history)-2] - self.history[len(self.history)-1] < tol
            
            if self.verbose:
                print(f"Iteration: {iteration:3.0f}, loss: {total_loss:9.6f}")
        print('Minibatch final loss: {:.4f}'.format(total_loss))
    
    def predict(self, X):
        '''
        We know, we weren't supposed to care about the accuracy but we just couldn't resist.
        '''
        Xt = torch.tensor(X.toarray(), dtype=torch.float, device=self.device)
        pred = Xt.mv(self.w)
        result = [self.positive_class if p > 0 else self.negative_class for p in pred]
        return result
