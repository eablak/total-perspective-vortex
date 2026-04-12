import numpy as np

class CSP:
    
    def __init__(self, X, y, n_components):
        
        self.X = X
        self.y = y
        self.n_components = n_components
        self.classes = 0
        self.n_epochs = X.shape[0]
        self.n_channels = X.shape[1]
        self.n_times = X.shape[2]


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


    def fit(self):
        
        covs = self.cov_matrix()
    
        
            