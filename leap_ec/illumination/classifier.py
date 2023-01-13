import numpy as np
import abc

class CellClassifier(abc.ABC):
    
    @abc.abstractmethod
    def classify(self, feature):
        pass

class GridClassifier(CellClassifier):
    
    def __init__(self, a_min, a_max, shape) -> None:
        self.a_min = np.array(a_min)
        self.extent = np.array(a_max) - self.a_min
        self.shape = shape
    
    def classify(self, feature):
        coord = (np.array(feature) - self.a_min) / self.extent
        coord *= self.shape
        
        return tuple(np.clip(coord, 0, self.shape).astype(int))
        
class CVTClassifier(CellClassifier):
    
    def __init__(self, centers):
        self.centers = np.array(centers)
    
    def classify(self, feature):
        feat_arr = np.array(feature)
        return np.argmin(
            np.linalg.norm(self.centers - feat_arr[None, :], axis=1),
        )
    
    @staticmethod
    def _converge_centers(centers, samples):
        cluster_ids = np.zeros(len(samples))
        new_cluster_ids = np.ones(len(samples))
        reverse_ids = [[] for _ in range(len(centers))]
        
        while True:
            for l in reverse_ids:
                l.clear()
            
            for i, sample in enumerate(samples):
                clus_id = np.argmin(
                    np.linalg.norm(centers - sample[None, :], axis=1)
                )
                new_cluster_ids[i] = clus_id
                reverse_ids[clus_id].append(sample)
            
            if (new_cluster_ids == cluster_ids).all():
                return centers
            else:
                centers = np.array([
                    np.mean(points, axis=0) if points else orig_center
                    for points, orig_center in zip(reverse_ids, centers)
                ])
                
                cluster_ids, new_cluster_ids = new_cluster_ids, cluster_ids
    
    @staticmethod
    def Cube(n_cells, n_samples, a_min, a_max):
        a_min = np.array(a_min)
        extent = np.array(a_max) - a_min
        n_dim = len(extent) if isinstance(extent, np.ndarray) else 1
        
        centers = np.random.random((n_cells, n_dim)) * extent
        samples = np.random.random((n_samples, n_dim)) * extent
        
        centers = CVTClassifier._converge_centers(centers, samples)

        return CVTClassifier(centers + a_min)
    
    @staticmethod
    def Ball(n_cells, n_samples, center, radius):
        def unit_points(n_points, d):
            pts = np.random.standard_normal((n_points, d))
            return pts / np.linalg.norm(pts, axis=1)[:, None]
        
        n_dim = len(center)
        
        centers = unit_points(n_cells, n_dim)\
            * np.random.random(n_cells)[:, None] ** (1 / n_dim)
        samples = unit_points(n_samples, n_dim)\
            * np.random.random(n_samples)[:, None] ** (1 / n_dim)
        
        centers = CVTClassifier._converge_centers(centers, samples)
        
        return CVTClassifier(centers * radius + center)
    
    @staticmethod
    def Normal(n_cells, n_samples, loc, scale):
        centers = np.random.normal(loc, scale, (n_cells, len(loc)))
        samples = np.random.normal(loc, scale, (n_samples, len(loc)))
        
        centers = CVTClassifier._converge_centers(centers, samples)
        
        return CVTClassifier(centers)