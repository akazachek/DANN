import numpy as np

class DANN(object):

    def __init__(self, learning_rate = 0.05, hidden_layer_size = 20, lambda_penalty = 0.5, max_iter = 200, seed = 123):
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.lambda_penalty = lambda_penalty
        self.max_iter = max_iter
        self.seed = seed

    def sigm(self, a):
        """
            sigmoid function
        """
        return 1. / (1.+np.exp(-a)) 

    def softmax(self, a):
        """
            softmax function
        """
        return np.exp(a) / np.sum(np.exp(a))

    """

        self.W is the (D x m) weight matrix for the hidden layer.
        self.b is the intercept vector for the hidden layer.
        self.V is the (L x D) weight matrix for the predictive layer.
        self.c is the intercept vector for the predictive layer.
        self.u is the (1 x D) weight vector for the domain regressor.
        self.z is the intercept scalar for the domain regressor.

        note that self.b and self.c both must be sliced prior to adding
        as their np arrays are akin to row-vectors.

        X is the matrix consisting of n many inputs [[x_n],...,[x_n]].
        note that this means it must be transposed when performing most
        operators to get each x_i into column-vector form.
        
    """

    def hidden_layer(self, X):
        """
            returns the matrix of (1 x D) vectors under hidden layer (G_f).
        """
        return self.sigm(self.W @ X.T + self.b[:,np.newaxis]).T

    def predictive_layer(self, Y):     
        """
            returns the matrix of (1 x L) vectors under predictive layer (G_y).
        """
        return self.softmax(self.V @ Y.T + self.c[:,np.newaxis]).T

    def predict_label(self, X):
        """
            computes composition of layers (G_y o G_f) and returns the
            matrix of one-hot (1 x L) vectors representing predicted labels.
        """
        predictive_layer = self.predictive_layer(self.hidden_layer(X))
        return np.argmax(predictive_layer, axis = 0)

    def domain_regressor(self, X):
        """
            returns probability of inputs belonging to the target (see \ell_d).
        """
        hidden_layer = self.hidden_layer(X).T
        return self.sigm(np.dot(self.u, hidden_layer) + self.z)

    def predict_domain(self, X):
        """
            returns binary value corresponding to whether inputs are 
            predicted to belong to the source (see function domain_regressor).
        """
        domain_prob = self.domain_regressor(X)
        return np.array(domain_prob < 0.5, dtype = int)