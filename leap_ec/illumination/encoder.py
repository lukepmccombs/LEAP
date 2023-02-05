import numpy as np
import abc
from typing import Hashable, List
from scipy.spatial import KDTree

class CellEncoder(abc.ABC):
    """An encoder that discretizes a feature space into cells.

    The resultant cells must be hashable.
    """
    
    @abc.abstractmethod
    def encode_cell(self, features) -> Hashable:
        """Encodes a feature descriptor into a discrete, hashable value

        :param features: the feature to be encoded
        """
        pass

class GridEncoder(CellEncoder):
    """An encoder that discretizes the feature space into fixed width cells over a region.
    """
    
    def __init__(self, a_min, a_max, shape) -> None:
        """Initializes a GridEncoder
        
        The total number of cells for this encoder is the product of the elements of `shape`.

        :param a_min: the minimal vertex of the box bounding the region
        :param a_max: the maximal vertex of the box bounding the region
        :param shape: the dimensions of the grid
        """
        self.a_min = np.array(a_min)
        self.extent = np.array(a_max) - self.a_min
        self.shape = shape
    
    def encode_cell(self, features):
        coord = (np.array(features) - self.a_min) / self.extent
        coord *= self.shape
        
        return tuple(np.clip(coord, 0, self.shape).astype(int))
        
class CVTEncoder(CellEncoder):
    """An encoder that divides the feature space into voronoi cells.
    
    This encoder is analogous to kmeans clustering, with each cluster denoting a cell.
    Centers can be user defined, or automatically generated from uniform distributions.
    """
    
    def __init__(self, centers):
        """Initializes a CVTEncoder
        
        The total number of cells for this encoder is the length of `centers`.

        :param centers: the centers of the clusters used to assign cells
        """
        self.centers = np.array(centers)
        self.kdt_ = KDTree(self.centers)
    
    def encode_cell(self, features):
        feat_arr = np.array(features)
        _, clus_id = self.kdt_.query(feat_arr)
        return clus_id
    
    @staticmethod
    def _converge_centers(centers, samples) -> List:
        """Performs convergence of the provided `centers` on `samples`

        :param centers: the initial cluster centroids
        :param samples: the data points sampled from the encoded region

        :return: the converged centroids
        """
        
        cluster_ids = np.zeros(len(samples))
        while True:
            # Utilizes scipy's KDTree implementation to calculate nearest neighbors quickly
            kdt = KDTree(centers)
            _, new_cluster_ids = kdt.query(samples)
            
            # Sum up and count the new clusters
            new_centers = np.zeros(centers.shape)
            new_cluster_counts = np.zeros(len(centers))
            for clus_id, sample in zip(new_cluster_ids, samples):
                new_centers[clus_id] += sample
                new_cluster_counts[clus_id] += 1
            
            # Do some safety accounting, clusters with no individuals are unchanged
            zero_centers = new_cluster_counts == 0
            new_centers[zero_centers] = centers[zero_centers]
            new_cluster_counts[zero_centers] = 1
            
            # Covnert sums to means
            new_centers /= new_cluster_counts[:, None]
            
            if all(new_cluster_ids == cluster_ids):
                return new_centers
            else:
                # Update for the next iteration
                centers = new_centers
                kdt = KDTree(centers)
                cluster_ids = new_cluster_ids
    
    @staticmethod
    def Orthope(n_cells, n_samples, a_min, a_max):
        """Creates a CVT cell encoder with cells uniformly distributed within an n-orthope.

        The resultant cell encoder takes features of the same dimensionality as `center`.

        :param n_cells: the number of cells in the encoder
        :param n_samples: the number of samples used to distribute the cells
        :param a_min: the minimal vertex of the box bounding the region
        :param a_max: the maximal vertex of the box bounding the region

        :return: a CVT encoder with cells distributed within the n-orthope
        """
        
        a_min = np.array(a_min, ndmin=1)
        a_max = np.array(a_max, ndmin=1)
        n_dim = len(a_min)
        
        def _sample_points(n_points):
            return np.random.uniform(a_min, a_max, (n_points, n_dim))
        
        centers = CVTEncoder._converge_centers(
            _sample_points(n_cells), _sample_points(n_samples)
        )

        return CVTEncoder(centers)
    
    @staticmethod
    def Ball(n_cells, n_samples, center, radius):
        """ Creates a CVT cell encoder with cells uniformly distributed within an n-ball.

        The resultant cell encoder takes features of the same dimensionality as `center`.

        :param n_cells: the number of cells in the encoder
        :param n_samples: the number of samples used to distribute the cells
        :param center: the center of the ball
        :param radius: the radius of the ball
            
        :return: a CVT encoder with cells distributed within the n-ball
        """
        
        center = np.array(center, ndmin=1)
        n_dim = len(center)
        
        def _sample_points(n_points):
            norm_pts = np.random.standard_normal((n_points, n_dim))
            # First we map the points onto the unit sphere
            unit_pts = norm_pts / np.linalg.norm(norm_pts, axis=1)[:, None]
            # Then we uniformly distribute them within the volume
            return unit_pts * np.random.random(n_points)[:, None] ** (1 / n_dim)
        
        centers = CVTEncoder._converge_centers(
            _sample_points(n_cells), _sample_points(n_samples)
        )
        
        return CVTEncoder(centers * radius + center)
    
    @staticmethod
    def Sample(n_cells, n_samples, sample_func):
        """Creates a CVT cell encoder with cells distributed over a user defined distribution.

        The resultant cell encoder takes features of the same dimensionality as the return value of `sample_func`

        Note that the final distribution of points will cover the same region as the user
        defined distribution, but may not have the same density distribution.

        :param n_cells: the number of cells in the encoder
        :param n_samples: the number of samples used to distribute the cells
        :param sample_func: a function that takes no arguments and returns a sampled point
        
        :return: a CVT encoder with cells distributed within the sampled region
        """
        
        centers = CVTEncoder._converge_centers(
            np.array([
                sample_func() for _ in range(n_cells)
            ]),
            np.array([
                sample_func() for _ in range(n_samples)
            ])            
        )
        
        return CVTEncoder(centers)