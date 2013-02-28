# -*- coding: utf-8 -*-
import numpy as np

class DStreamCharacteristicVector():
    
    def __init__(self, 
                 last_update_time, 
                 last_sporadic_removal_time, 
                 density, 
                 label, 
                 status):
        
        self.last_update_time = last_update_time
        self.last_sporadic_removal_time = last_sporadic_removal_time
        self.density = density
        self.label = label
        self.status = status
        
class DStreamCluster():

    def __init__(self, 
                 characteristic_vectors = None):
                     
        self.characteristic_vectors = np.array([], dtype=type(DStreamCharacteristicVector))
        
        if characteristic_vectors != None:
            self.characteristic_vectors = np.append(self.characteristic_vectors, characteristic_vectors)
              
class DStreamClusterer():  
    '''
    Initialize with defaults from reference algorithm
    '''
    def __init__(self, 
                 dense_threshold_parameter = 3.0, #C_m
                 sparse_threshold_parameter = 0.8,  #C_l
                 sporadic_threshold_parameter = 0.3, #beta
                 decay_factor = 0.998, #lambda
                 dimensions = 2, 
                 domains_per_dimension = ((0.0, 100.0), (0.0, 100.0)),
                 partitions_per_dimension = (10, 10)):
                     
        self.dense_threshold_parameter = dense_threshold_parameter
        self.sparse_thresold_parameter = sparse_threshold_parameter
        self.sporadic_threshold_parameter = sporadic_threshold_parameter
        self.decay_factor = decay_factor
        
        self.dimensions = dimensions
        self.domains_per_dimension = domains_per_dimension
        self.partitions_per_dimension = partitions_per_dimension        


        N = 1        
        for i in range(dimensions):
            N *= partitions_per_dimension[i]
        
        self.maximum_grid_count = N
        self.grids = {}
        
        self.gap_time = -1.0
        self.compute_gap_time()
        
    def add_datum(self):
        pass
        
    def initialize_grids(self):
        pass
        
    def initialize_clusters(self):
        pass
    
    def cluster(self):
        pass
    
    def detect_and_remove_sporadic_grids(self):
        pass
    
    def compute_gap_time(self):
        
        quotient1 = self.sparse_thresold_parameter/self.dense_threshold_parameter
        quotient2 = (self.maximum_grid_count - self.dense_threshold_parameter)/(self.maximum_grid_count - self.sparse_thresold_parameter)
        max_val = np.max([quotient1 , quotient2])
        max_log = np.log(max_val)/np.log(self.decay_factor)
        gap = np.floor([max_log])
        self.gap_time = gap[0]
        print 'computed gap time: ', self.gap_time
        
    def get_grid_indices(self, datum):
        pass
    
    def density_threshold_function(self, current_time, last_update_time):
        pass
    
       
        
    


if __name__ == "__main__":
    
    d_stream_clusterer = DStreamClusterer()
    print d_stream_clusterer
    
    d_stream_characteristic_vector = DStreamCharacteristicVector(0, 0, 0.0, 'NO_CLASS', 'NORMAL')
    print d_stream_characteristic_vector
    
    d_stream_cluster = DStreamCluster()
    print d_stream_cluster