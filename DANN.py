import numpy as np
from itertools import repeat

class DANN(object):

    def __init__(self, learning_rate = 0.05, hidden_layer_size = 20, lambda_penalty = 0.5, max_iter = 200, seed = 123):
        self.learning_rate = learning_rate
        self.hidden_layer_size = hidden_layer_size
        self.lmbda = lambda_penalty
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

        Y is the vector [y_1,...,y_n], where y_i is the number corresponding
        to the label for x_i.
        
    """

    def hidden_layer(self, X):
        """
            returns the matrix of (1 x D) vectors under hidden layer (G_f).
        """
        return self.sigm(self.W @ X.T + self.b[:,np.newaxis]).T

    def predictive_layer(self, X):     
        """
            returns the matrix of (1 x L) vectors under predictive layer (G_y).
        """
        return self.softmax(self.V @ X.T + self.c[:,np.newaxis]).T

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

    def train(self, X, Y, X_target):

        np.random.seed(self.seed)

        # take set here to get number of unique labels
        num_labels = len(set(Y))
        num_input = np.shape(X)[0]
        num_target = np.shape(X_target)[0]

        # matrix of one-hot vectors corresponding to labels.
        # that is, given an input pair (x_i, y_i), the associated
        # one-hot vector is the (1xL) vector consisting of 0
        # everywhere except for a 1 in the y_i-th place.
        one_hots = np.zeros((num_input, num_labels))
        one_hots[:,Y] = 1.

        for _ in repeat(None, self.max_iter):
            for i in range(num_input):

                # take transposes now for ease of transcription
                W = self.W
                b = self.b.T
                V = self.V
                u = self.u.T
                lm = self.lambda_penalty
                
                x_i = X[i,:].T
                e_i = one_hots[i,:].T
                hidden_vector = self.hidden_layer(x_i.T).T
                predicted_vector = self.predictive_layer(hidden_vector.T).T

                ### compute standard stochastic gradients

                d_c = -(e_i - predicted_vector)
                d_V = d_c @ hidden_vector.T
                d_b = (V.T @ d_c) * hidden_vector * (1. - hidden_vector)
                d_W = d_b @ x_i.T

                ### incorporate domain regressor gradients

                # using source data
                domain_prob = self.domain_regressor(x_i.T)
                d_z = lm * (1. - domain_prob)
                d_u = d_z * hidden_vector
                tmp = d_z * self.u * hidden_vector * (1. - hidden_vector)
                d_b += tmp
                d_W += tmp @ x_i.T
                # using target data
                j = np.random.randit(0, num_target - 1)
                t_x_j = X_target[j,:].T
                t_hidden_vector = self.hidden_layer((W @ t_x_j + b).T).T
                t_domain_prob = self.domain_regressor(t_hidden_vector.T)
                d_z -= lm * t_domain_prob
                d_u -= d_z * t_hidden_vector
                tmp = -d_z * u * t_hidden_vector * (1. - t_hidden_vector)
                d_b += tmp
                d_W += tmp @ t_x_j.T

                ### update weights and intercepts

                mu = self.learning_rate
                self.W -= mu * d_W
                self.V -= mu * d_V
                self.b -= mu * d_b
                self.c -= mu * d_c

                self.u += mu * d_u
                self.z += mu * d_z