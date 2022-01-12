import numpy as np
from itertools import repeat

class DANN(object):

    def __init__(self, learning_rate = 0.05, hidden_layer_size = 25, lambda_penalty = 1., max_iter = 100, seed = 123):
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
        self.u is the D-dimensional weight vector for the domain regressor.
        self.z is the intercept scalar for the domain regressor.

        X is the matrix consisting of n many inputs [[x_n],...,[x_n]].

        Y is the vector [y_1,...,y_n], where y_i is the number corresponding
        to the label for x_i.
        
    """

    def hidden_layer(self, x):
        """
            returns the D-dimensional vector under hidden layer (G_f).
        """
        return self.sigm(self.W @ x + self.b)

    def predictive_layer(self, x):     
        """
            returns the L-dimensional vector under predictive layer (G_y).
        """
        return self.softmax(self.V @ x + self.c)

    def predict_label(self, x):
        """
            computes composition of layers (G_y o G_f) and returns the
            one-hot L-dimensional vector representing predicted label.
        """
        predictive_layer = self.predictive_layer(self.hidden_layer(x))
        return np.argmax(predictive_layer)

    def predict_labels(self, X):
        """
            computes composition of layers (G_y o G_f) and returns matrix
            of of one-hot (1 x L) vectors representing predicted labels.
        """
        # note that transposes are required as this takes in a matrix, not a vector.
        # moreover, the intercepts must be translated to row vectors via newaxis.
        hidden_vectors = np.apply_along_axis(self.sigm, 0, self.W @ X.T + self.b[:,np.newaxis])
        predicted_vectors = np.apply_along_axis(self.softmax, 0, self.V @ hidden_vectors + self.c[:,np.newaxis])
        return np.argmax(predicted_vectors, 0)

    def domain_regressor(self, x):
        """
            returns probability of inputs belonging to the target (see \ell_d).
        """
        hidden_layer = self.hidden_layer(x)
        return self.sigm(np.dot(self.u, hidden_layer) + self.z)

    def predict_domain(self, x):
        """
            returns binary value corresponding to whether inputs are 
            predicted to belong to the source (see function domain_regressor).
        """
        domain_prob = self.domain_regressor(x)
        return np.array(domain_prob < 0.5, dtype = int)

    def random_init(self, num_row, num_col):
        """
            initializes a random weight or intercept matrix
        """
        eps = np.sqrt(6. / (num_row + num_col))
        return eps * (2 * np.random.rand(num_row, num_col) - 1.)

    def train(self, X, Y, X_target):

        # take set here to get number of unique labels
        num_labels = len(set(Y))
        num_input, num_features = np.shape(X)
        num_target = np.shape(X_target)[0]

        # initialize random weights and intercepts
        np.random.seed(self.seed)
        self.W = self.random_init(self.hidden_layer_size, num_features)
        self.b = np.zeros(self.hidden_layer_size)
        self.V = self.random_init(num_labels, self.hidden_layer_size)
        self.c = np.zeros(num_labels)
        self.u = np.zeros(self.hidden_layer_size)
        self.z = 0.

        # matrix of one-hot vectors corresponding to labels.
        # that is, given an input pair (x_i, y_i), the associated
        # one-hot vector is the (1xL) vector consisting of 0
        # everywhere except for a 1 in the y_i-th place.
        one_hots = np.zeros((num_input, num_labels))
        for i in range(num_input): one_hots[i,Y[i]] = 1.
        # surely this can be vectorized ?

        # shorthands for transcription of algorithm    
        lm = self.lambda_penalty
        mu = self.learning_rate

        for _ in repeat(None, self.max_iter):
            for i in range(num_input):

                # shorthands for transcription of algorithm    
                V = self.V
                u = self.u

                x_i = X[i,:]
                e_i = one_hots[i,:]
                hidden_vector = self.hidden_layer(x_i)
                predicted_vector = self.predictive_layer(hidden_vector)

                ### compute standard stochastic gradients

                d_c = -(e_i - predicted_vector)
                d_V = np.outer(d_c, hidden_vector)
                d_b = (V.T @ d_c) * hidden_vector * (1. - hidden_vector)
                d_W = np.outer(d_b, x_i)

                ### incorporate domain regressor gradients

                # using source data
                domain_prob = self.domain_regressor(x_i)
                d_z = lm * (1. - domain_prob)
                d_u = d_z * hidden_vector
                tmp = d_z * u * hidden_vector * (1. - hidden_vector)
                d_b += tmp
                d_W += np.outer(tmp, x_i)

                # using target data
                j = np.random.randint(num_target)
                t_x_j = X_target[j,:]
                t_hidden_vector = self.hidden_layer(t_x_j)
                t_domain_prob = self.domain_regressor(t_x_j)

                delta_d_z = lm * t_domain_prob
                d_z -= delta_d_z
                d_u -= delta_d_z * t_hidden_vector
                tmp = -delta_d_z * u * t_hidden_vector * (1. - t_hidden_vector)
                d_b += tmp
                d_W += np.outer(tmp, t_x_j)

                ### update weights and intercepts

                self.W -= mu * d_W
                self.V -= mu * d_V
                self.b -= mu * d_b
                self.c -= mu * d_c

                self.u += mu * d_u
                self.z += mu * d_z