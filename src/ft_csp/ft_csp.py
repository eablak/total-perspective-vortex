import numpy as np
from scipy.linalg import eigh, pinv


class CSP:
    
    def __init__(self, X, y, n_components):
        
        self.X = X
        self.y = y
        self.n_components = n_components
        self.classes = None
        self.n_epochs = X.shape[0]
        self.n_channels = X.shape[1]
        self.n_times = X.shape[2]
        self.filters = None
        self.patterns = None


    # step 1: covariance matrix
    def cov_matrix(self):
        """
            self.X has 3D data -> (trial, channel, time)

            np.cov(m, y=None, rowvar=True, bias=False, ddof=None)
            * m: input data (1D or 2D array)

            because of np.cov needs 2D input we will decrease the dimensions from (n_epochs, n_channels, n_times) to (n_channels, n_samples)

            to be able to do this process:
                1- transpose the matrix with given axeses
                2- reshape it for desired dimensions
                3- calculate covariance

        """

        self.classes = np.unique(self.y)
        covs = []
        
        for label in self.classes:
            x_class = self.X[self.y == label]
            x_class = np.transpose(x_class, [1, 0, 2])
            x_class = x_class.reshape(self.n_channels, -1)
            cov = np.cov(x_class, ddof=1)
            covs.append(cov)

        return covs


    # step 2: eigen decomposition
    def gevp(self, covs):

        """
            solving generalized eigenvalue problem:
                1- get the eigenvalues and eigenvectors
                2- sort eigenvalues by distance from 0.5 and reorder eigenvector columns with using those indices 
                    (the aim for here, when you order indicates only from smaller to larger you are making basic ascending order but when you order them distance from 0.5 you will get values near to 0.5 which is mean if your distance close to 0.5 the component seperates two classes almost perfectly. and we want discriminative values most so we use this approach)
                3- select best n_components
                4- extract filters and patterns (spatial filters: transposed eigenvectors -> (64 channels, 4 columns to 4 column, 64 channel), patterns: pseudoinverse eigenvectors -> calculate the generalized inverse of a matrix)
        """

        evalue, evect = eigh(covs[0], covs[1])
        
        sorted_evalue = np.argsort(np.abs(evalue - 0.5))[::-1]
        evect = evect[:, sorted_evalue]

        evect = evect[:, :self.n_components]

        self.filters = evect.T
        self.patterns = pinv(evect)


    def fit(self):
        
        covs = self.cov_matrix()
        self.gevp(covs)
    
        
            