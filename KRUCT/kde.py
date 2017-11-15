import numpy as np

class gaussian_kde(object):
    def __init__(self, cov_mat, dataset = None, weights = None, band_width = 1):
        self.band_width = band_width
        self.cov_mat = np.array(cov_mat) * band_width**2
        self.inv_cov_mat = np.linalg.inv(self.cov_mat) / band_width**2
        self.band_width = band_width
        
        self.weights = weights
        if self.weights is not None:
            self.weights = np.array(weights)
            if len(self.weights.shape) != 1:
                raise ValueError("Weights must be 1 dimension.")
        
        self.dataset = dataset
        if self.dataset is not None:
            self.dataset = np.atleast_2d(dataset)
            if not self.dataset.size > 1:
                raise ValueError("`dataset` input should have multiple elements.")

            self.d, self.n = self.dataset.shape
        
        #self._norm_factor = np.sqrt(np.linalg.det(2*np.pi*self.cov_mat)) * self.n

    def set_dataset(self, dataset):
        self.dataset = np.atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape
        
    def set_weights(self, weights):
        self.weights = weights
        if self.weights is not None:
            self.weights = np.array(weights)
            if len(self.weights.shape) != 1:
                raise ValueError("Weights must be 1 dimension.")
        
        
    def evaluate(self, points):
        points = np.atleast_2d(points)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = np.reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        result = np.zeros((m,), dtype=float)

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:, i, np.newaxis] - points
                tdiff = np.dot(self.inv_cov_mat, diff)
                energy = np.sum(diff*tdiff,axis=0) / 2.0
                if self.weights is None:
                    result = result + np.exp(-energy)
                else:
                    result = result + np.exp(-energy) * self.weights[i]
        else:
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:, i, np.newaxis]
                tdiff = np.dot(self.inv_cov_mat, diff)
                energy = np.sum(diff * tdiff, axis=0) / 2.0
                if self.weights is None:
                    result[i] = np.sum(np.exp(-energy), axis=0)
                else:
                    result[i] = np.sum(np.exp(-energy) * self.weights, axis=0)
                
        #result = result / self._norm_factor

        return result
        