# -*- coding: utf-8 -*-
import numpy as np

class DStreamCharacteristicVector():
    
    def __init__(self, 
                 density=0.0, 
                 label='NO_CLASS', 
                 status='NORMAL',
                 last_update_time=-1, 
                 last_sporadic_removal_time=-1, 
                 last_marked_sporadic_time=-1):
        
        self.last_update_time = last_update_time
        self.last_sporadic_removal_time = last_sporadic_removal_time
        self.last_marked_sporadic_time = last_marked_sporadic_time
        self.density = density
        self.label = label
        self.status = status
        
class DStreamCluster():

    def __init__(self, 
                 grids = None):
                     
        self.grids = {}#np.array([], dtype=type(DStreamCharacteristicVector))
        if grids != None:
            self.addGrids(grids)
        
    def addGrids(self, grids):
        
        for indices, grid in grids.items():
            self.grids[indices] = grid
                
    def removeGrids(self, grids):
        
        for indices, grid in grids.items():
            if self.grids.has_key(indices):
                self.grids =  {key: value for key, value in self.grids.items() if value is not grid} #this can be re-written
                
              
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
        self.removed_grid_cache = {}
        self.clusters = np.array([], dtype=type(DStreamCluster))
        self.cluster_count = 4 #from ref        
        
        self.gap_time = -1.0
        self.compute_gap_time()
        self.current_time = 0
        
        
        
    def add_datum(self, datum):
        
        indices = self.get_grid_indices(datum)
        
        if self.grids.has_key(indices):
            grid = self.grids[indices]
        else:
            if self.removed_grid_cache.has_key(indices):
                grid = self.removed_grid_cache[indices]
            else:
                grid = DStreamCharacteristicVector()
                
        grid.density = 1.0 + grid.density*self.decay_factor**(self.current_time - grid.last_update_time)
        grid.last_update_time = self.current_time
        self.grids[indices] = grid
            
        if self.current_time == self.gap_time:
            self.initialize_clusters()
        if np.mod(self.current_time, self.gap_time) == 0:
            sporadic_grids = self.get_sporadic_grids()
            for indices, grid in sporadic_grids.items():
                if grid.last_marked_sporadic_time != -1 and grid.last_marked_sporadic_time + 1 <=self.current_time:
                    if grid.last_update_time != self.current_time:
                        self.grids = {key: value for key, value in self.grids.items() if value is not grid}
                        grid.last_sporadic_removal_time = self.current_time
                        self.removed_grid_cache[indices ] = grid
                    else:
                        if self.is_sporadic(grid, self.current_time) == False:
                            grid.status = 'NORMAL'
                            self.grids[indices] = grid
            self.detect_sporadic_grids(self.current_time)
            self.cluster()
            
        self.current_time += 1
        
        
    def initialize_clusters(self):

        self.update_density_category()
        
        cluster_counts = np.array([])
        dense_grids, non_dense_grids = self.get_dense_grids()
        
        cluster_size = np.round(len(dense_grids.keys)/self.cluster_count)
        print 'cluster size: ', cluster_size
        for i in range(self.cluster_count):
            if i == self.cluster_count - 1:
                current_total = np.sum(cluster_counts)
                last_count = len(dense_grids.keys) - current_total
                cluster_counts = np.append(cluster_counts, last_count)
                print 'last cluster size: ', last_count
            else:
                cluster_counts = np.append(cluster_counts, cluster_size)
        counter = 0
        for grid_count in cluster_counts: 
            cluster_grids = {}
            for i in range(grid_count):
                k = dense_grids.keys()[counter]
                v = dense_grids.values()[counter]
                cluster_grids[k] = v
                counter += 1
            cluster = DStreamCluster(cluster_grids)
            self.clusters = np.append(self.clusters, cluster)
            #cluster = DStreamCluster(a;osirferj)
        for indices, grid in non_dense_grids.items():
            grid.label = 'NO_CLASS'
            self.grids[indices] = grid
        
        #repeat section TODO
        

    def cluster(self):
        pass
    
    def update_density_category(self):
        for indices, grid in self.grids.items():
            if grid.density >= self.dense_threshold_parameter/(self.maximum_grid_count*(1.0-self.decay_factor)):
                grid.label = 'DENSE'
                print 'grid with indices: ', indices, ' is DENSE'
            if grid.density <= self.sparse_thresold_parameter/(self.maximum_grid_count*(1.0-self.decay_factor)):
                grid.label = 'SPARSE'
                print 'grid with indices: ', indices, ' is SPARSE'                
            if grid.density >= self.sparse_thresold_parameter/(self.maximum_grid_count*(1.0-self.decay_factor)) and grid.density <= self.dense_threshold_parameter/(self.maximum_grid_count*(1.0-self.decay_factor)):
                grid.label = 'TRANSITIONAL'
                print 'grid with indices: ', indices, ' is TRANSITIONAL' 
            self.grids[indices] = grid
    def get_dense_grids(self):
        dense_grids = {}
        non_dense_grids = {}
        for indices, grid in self.grids.items():
            if grid.label == 'DENSE':
                dense_grids[indices] = grid
            else:
                non_dense_grids[indices] = grid
                
        return dense_grids, non_dense_grids
    
    def get_sporadic_grids(self):
        sporadic_grids = {}#np.array([], type(DStreamCharacteristicVector))
        for indices, grid in self.grids.items():
            if grid.status == 'SPORADIC':
                sporadic_grids[indices] = grid#np.append(sporadic_grids, d_stream_characteristic_vector)
        return sporadic_grids
    def is_sporadic(self, grid, current_time):
        if grid.density < self.density_threshold_function(grid.last_update_time, current_time) and current_time >= (1.0 + self.sporadic_threshold_parameter)*grid.last_sporadic_removal_time:
            return True
        return False
    def detect_sporadic_grids(self, current_time):
        for indices, grid in self.grids.items():
            if self.is_sporadic(grid, current_time):
                #if d_stream_characteristic_vector.density < self.density_threshold_function(d_stream_characteristic_vector.last_update_time, current_time) and current_time >= (1.0 + self.sporadic_threshold_parameter)*d_stream_characteristic_vector.last_sporadic_removal_time:
                print 'detected sporadic grid at indices: ', indices                
                grid.status = 'SPORADIC'
                grid.last_marked_sporadic_time = current_time
                self.grids[indices] = grid
    
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
    
    def density_threshold_function(self, last_update_time, current_time):
        
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
    
    d_stream_characteristic_vector = DStreamCharacteristicVector ()
    print d_stream_characteristic_vector
    
    d_stream_cluster = DStreamCluster()
    print d_stream_cluster