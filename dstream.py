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
        self.has_clustered_once = False
        
    def add_datum(self):
        pass
        
    def initialize_clusters(self):

        #do stuff here        
        
        self.has_clustered_once = True
    
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
        '''
        dimensions = 2, 
        domains_per_dimension = ((0.0, 100.0), (0.0, 100.0)),
        partitions_per_dimension = (10, 10)
        '''
        indices = np.array([])
        for i in range(self.dimensions):
            
            domain_tuple = self.domains_per_dimension[i]
            partitions = self.partitions_per_dimension[i]
            
            index = np.floor([(datum[i]/domain_tuple[1])*(partitions)])[0]
            if index == partitions:
                print 'index equals partitions: ', index, partitions
                index -= 1
            indices = np.append(indices, index)
        return indices
    
    def density_threshold_function(self, current_time, last_update_time):
        
        top = self.sparse_thresold_parameter * (1.0 - self.decay_factor ** (current_time - last_update_time + 1))
        bottom = self.maximum_grid_count * (1.0 - self.decay_factor)
        return top/bottom
    
       
        
    


if __name__ == "__main__":
    
    d_stream_clusterer = DStreamClusterer()
    print d_stream_clusterer
    print 'indices for inserting 35.0, 100.0: ', d_stream_clusterer.get_grid_indices((35.0, 100.0))
    print 'indices for inserting 0.0, 60.0: ', d_stream_clusterer.get_grid_indices((0.0, 60.0))
    print 'getting dth for t=2, t=0', d_stream_clusterer.density_threshold_function(2, 0)
    print 'getting dth for t=4, t=0', d_stream_clusterer.density_threshold_function(4, 0)
    print 'getting dth for t=8, t=0', d_stream_clusterer.density_threshold_function(8, 0)    
    
    d_stream_characteristic_vector = DStreamCharacteristicVector(0, 0, 0.0, 'NO_CLASS', 'NORMAL')
    print d_stream_characteristic_vector
    
    d_stream_cluster = DStreamCluster()
    print d_stream_cluster