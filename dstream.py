# -*- coding: utf-8 -*-
import sys
sys.path.append('libs/networkx-1.7-py2.7.egg')
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


'''
there are many other places where I can/should substitute my algs with the new NetworkX version using graphs

TODO:
    -iteration impl
    -in all the places where a cluster is implicitly created/destroyed (either in init cluster or cluster funcs), must update self.class_keys accordingly (5 locations?)
    -actually cartesian product my be incorrect to use here; neighbors can only have ONE dimensions inex off by one, not all of them off by one. so need to iteratate
        over dimensions and take ndim-1 points from the test grid and then check if we can fiddle with the other ONE, and repeat for each dim.

'''

class DStreamCharacteristicVector():
    
    def __init__(self, 
                 density=0.0, 
                 label='NO_CLASS', 
                 status='NORMAL',
                 density_category=None,
                 last_update_time=-1, 
                 last_sporadic_removal_time=-1, 
                 last_marked_sporadic_time=-1,
                 category_changed_last_time=False):
        
        self.last_update_time = last_update_time
        self.last_sporadic_removal_time = last_sporadic_removal_time
        self.last_marked_sporadic_time = last_marked_sporadic_time
        self.density_category = density_category
        self.density = density
        self.label = label
        self.status = status
        self.category_changed_last_time = category_changed_last_time
        

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
                 partitions_per_dimension = (50, 50)):
        print 'provdied: ', dense_threshold_parameter, sparse_threshold_parameter, sporadic_threshold_parameter, decay_factor, dimensions, domains_per_dimension, partitions_per_dimension
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
        #self.clusters = np.array([], dtype=type(DStreamCluster))
        self.cluster_count = 4 #from ref        
        
        self.gap_time = -1.0
        self.compute_gap_time()
        self.has_clustered_once = False
        
        self.current_time = 0
        #self.last_updated_grids = {}
        
        seed = 331
        self.seed = seed        
        self.class_keys = np.array([])
        random.seed(self.seed)
        
    def generate_unique_class_key(self):
        test_key = np.int(np.round(random.uniform(0, 1), 8)*10**8)
        while test_key in self.class_keys:
            print 'class key test collision...weird'
            test_key = np.int(np.round(random.uniform(0, 1), 8)*10**8)
        
        return test_key
        
    def add_datum(self, datum):
        #print 'current time: ', self.current_time, ' and gap time: ', self.gap_time
        #print 'adding: ', datum
        indices = tuple(self.get_grid_indices(datum))
        #print 'with indices: ', indices, tuple(indices)
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
        print "current time, gap time: ", self.current_time, self.gap_time
        if np.mod(self.current_time, self.gap_time) == 0 and self.has_clustered_once:
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
        #print 'grids after adding: ', self.grids

    def initialize_clusters(self):
        self.has_clustered_once = True
        self.update_density_category()
        
        cluster_counts = np.array([])
        dense_grids, non_dense_grids = self.get_dense_grids()
        cluster_size = np.round(len(dense_grids.keys())/self.cluster_count)
        print 'dense count: {} non-dense count: {} cluster count: {} cluster size: {}'.format(len(dense_grids.keys()), len(non_dense_grids.keys()), self.cluster_count, cluster_size)
        
        print 'cluster size: ', cluster_size
        for i in range(self.cluster_count):
            if i == self.cluster_count - 1:
                current_total = np.sum(cluster_counts)
                last_count = len(dense_grids.keys()) - current_total
                cluster_counts = np.append(cluster_counts, np.int(last_count))
                print 'last cluster size: ', last_count
            else:
                cluster_counts = np.append(cluster_counts, np.int(cluster_size))
        counter = 0
        print cluster_counts
        for grid_count in cluster_counts: 
            grid_count = np.int(grid_count)
            cluster_grids = {}
            unique_class_key = self.generate_unique_class_key()
            self.class_keys = np.append(self.class_keys, unique_class_key)
            print grid_count
            for i in range(grid_count):
                k = dense_grids.keys()[counter]
                v = dense_grids.values()[counter]
                v.label = unique_class_key
                cluster_grids[k] = v
                counter += 1
        for indices, grid in non_dense_grids.items():
            grid.label = 'NO_CLASS'
            self.grids[indices] = grid
        
        
        '''
        This eventually needs to be wrapped in a do until no change in clusters
        TODO
        '''
        for i in range(self.class_keys.size):
            class_key = self.class_keys[i]
            cluster_grids = self.get_grids_of_cluster_class(class_key)
            inside_grids, outside_grids = self.get_inside_grids(cluster_grids)
            for indices, grid in outside_grids.items():
                neighboring_grids = self.get_neighboring_grids(indices)
                for neighbor_indices, neighbor_grid in neighboring_grids.items():
                    for j in range(self.class_keys.size):
                        test_class_key = self.class_keys[j]
                        test_cluster_grids = self.get_grids_of_cluster_class(test_class_key)
                        
                        neighbor_belongs_to_test_cluster = self.validate_can_belong_to_cluster(test_cluster_grids, (neighbor_indices, neighbor_grid))
                        if neighbor_belongs_to_test_cluster:
                            if len(cluster_grids.keys) > len(test_cluster_grids.keys):
                                self.assign_to_cluster_class(test_cluster_grids, class_key)
                            else:
                                self.assign_to_cluster_class(cluster_grids, test_class_key)
                        elif neighbor_grid.density_category == 'TRANSITIONAL':
                            self.assign_to_cluster_class({neighbor_indices:neighbor_grid}, class_key)


        
    def cluster(self):
        self.update_density_category()
        for indices, grid in self.get_most_recently_categorically_changed_grids().items():
            
                                
            neighboring_grids = self.get_neighboring_grids(indices)
            neighboring_clusters = {}
            for neighbor_indices, neighbor_grid in neighboring_grids.items():
                neighbors_cluster_class = neighbor_grid.label
                neighbors_cluster_grids = self.get_grids_of_cluster_class(neighbors_cluster_class)
                neighboring_clusters[neighbor_indices, neighbors_cluster_class] =  neighbors_cluster_grids
                
            max_neighbor_cluster_size = 0
            max_size_indices = None
            #max_size_cluster = None
            print 'neighboring clusters: ', neighboring_clusters.keys()
            for k, ref_neighbor_cluster_grids in neighboring_clusters.items():
                test_size = len(ref_neighbor_cluster_grids.keys())
                print 'size comparison: ', test_size, max_neighbor_cluster_size
                if test_size > max_neighbor_cluster_size:
                    max_neighbor_cluster_size = test_size
                    #max_size_cluster = neighbor_cluster
                    max_size_cluster_key = k[1]
                    max_size_indices = k[0]
                    max_cluster_grids = ref_neighbor_cluster_grids
            if max_size_indices == None:
                print 'no neighbors, thus no biggest neighbors, skipping clustering this time***********'
                return
            max_size_grid = neighboring_grids[max_size_indices]
            grids_cluster = self.get_grids_of_cluster_class(grid.label)                    
                                
            
            if grid.density_category == 'SPARSE':
                changed_grid_cluster_class = grid.label
                cluster_grids_of_changed_grid = self.get_grids_of_cluster_class(changed_grid_cluster_class)
                would_still_be_connected = self.cluster_still_connected_upon_removal(cluster_grids_of_changed_grid, (indices, grid))
                grid.label = 'NO_CLASS'
                self.grids[indices] = grid
                
                if would_still_be_connected == False:
                    self.extract_two_clusters_from_grids_having_just_removed_given_grid(cluster_grids_of_changed_grid, (indices, grid))
            elif grid.density_category == 'DENSE':
                if max_size_grid.density_category == 'DENSE':
                    
                    if grid.label == 'NO_CLASS':
                        grid.label = max_size_cluster_key
                        self.grids[indices] = grid
                    elif len(grids_cluster.keys()) > max_neighbor_cluster_size:
                        for max_indices, max_grid in max_cluster_grids.items():
                            max_grid.label = grid.label
                            self.grids[max_indices] = max_grid
                    elif len(grids_cluster.keys()) <= max_neighbor_cluster_size:
                        for grids_cluster_indices, grids_cluster_grid in grids_cluster.items():
                            grids_cluster_grid.label = max_size_cluster_key
                            self.grids[grids_cluster_indices] = grids_cluster_grid
                elif max_size_grid.density_category == 'TRANSITIONAL':
                    if grid.label == 'NO_CLASS' and self.grid_becomes_outside_if_other_grid_added_to_cluster((max_size_indices, max_size_grid), max_cluster_grids, (indices, grid)):
                        grid.label = max_size_cluster_key
                        self.grids[indices] = grid
                    elif len(grids_cluster.keys()) >= max_neighbor_cluster_size:
                        max_size_grid.label = grid.label
                        self.grids[max_size_indices] = max_size_grid
            elif grid.density_category == 'TRANSITIONAL':
                max_outside_cluster_size = 0
                max_outside_cluster_class = None
                for k, ref_neighbor_cluster_grids in neighboring_clusters.items():
                    ref_cluster_key = k[1]
                    #ref_indices = k[0]
                    ref_grids = ref_neighbor_cluster_grids
                    if self.grid_is_outside_if_added_to_cluster((indices, grid), ref_grids) == True:
                        test_size = len(ref_grids.keys())
                        if test_size > max_outside_cluster_size:
                            max_outside_cluster_size = test_size
                            max_outside_cluster_class = ref_cluster_key
                grid.label = max_outside_cluster_class
                self.grids[indices] = grid
           
      




    def extract_two_clusters_from_grids_having_just_removed_given_grid(self, grids_without_removal, removed_grid):
        #first remove it, then split into two, then add the two to self.grids
        removed_grid_indices = removed_grid[0]
        
        grids_with_removal =  {key: value for key, value in grids_without_removal if key is not removed_grid_indices}
        graph_with_removal = self.get_graph_of_cluster(grids_with_removal)
        subgraphs = nx.connected_component_subgraphs(graph_with_removal)
        
        if len(subgraphs) != 2:
            print 'found more than two subgraphs'
        
        for i in range(len(subgraphs)):
            if i != 0:
                nodes = subgraphs[i].nodes()
                new_class_key = self.generate_unique_class_key()
                self.class_keys = np.append(self.class_keys, new_class_key)
                for node in nodes:
                    grid = self.grids[node]
                    grid.label = new_class_key
                    self.grids[node] = grid
            


    def cluster_still_connected_upon_removal(self, grids_without_removal, removal_grid):
        
        removal_grid_indices = removal_grid[0]
        
        grids_with_removal = {key: value for key, value in grids_without_removal if key is not removal_grid_indices}
        graph_with_removal = self.get_graph_of_cluster(grids_with_removal)
        return nx.is_connected(graph_with_removal)

    def get_graph_of_cluster(self, grids):
        indices_list = grids.keys()
        g = nx.empty_graph()
        for i in range(len(indices_list)):
            indices = indices_list[i]
            for j in range(len(indices_list)):
                other_indices = indices_list[j]
                if self.are_neighbors(indices, other_indices):
                    if g.has_edge(indices, other_indices) == False:
                        g.add_edge(indices, other_indices)
        return g
        
    def are_neighbors(self, grid1_indices, grid2_indices):
        target_identical_count = self.dimensions - 1
        identical_count = 0
        for i in range(self.dimensions):
            if grid1_indices[i] == grid2_indices[i]:
                identical_count += 1
            elif np.abs(grid1_indices[i] - grid2_indices[i]) != 1:
                return False
        return identical_count == target_identical_count
  
    '''below are completed'''
    
    def validate_can_belong_to_cluster(self, cluster, test_grid):
        #first validate cluster?
        print 'checking if grid can be valid in cluster. first doing pre-addition check'
        is_valid_before = self.is_valid_cluster(cluster)
        if is_valid_before != True:
            print 'provided cluster is invalid...returning False'
            return False
            
        cluster[test_grid[0]] = test_grid[1]
        is_valid_after = self.is_valid_cluster(cluster)
        return is_valid_after
        
    def is_valid_cluster(self, grids):
        inside_grids, outside_grids = self.get_inside_grids(grids)
        for indices, grid in inside_grids.items():
            if grid.density_category != 'DENSE':
                return False
        for indices, grid in outside_grids.items():
            if grid.density_category == 'DENSE':
                return False
        return True
        
    def grid_becomes_outside_if_other_grid_added_to_cluster(self, test_grid, cluster_grids, insert_grid):
        cluster_grids[insert_grid[0]] = insert_grid[1]
        inside_grids, outside_grids = self.get_inside_grids(cluster_grids)
        if outside_grids.has_key(test_grid[0]):
            return True
        return False
        
    def grid_is_outside_if_added_to_cluster(self, test_grid, grids):
        grids[test_grid[0]] = test_grid[1] 
        inside_grids, outside_grids = self.get_inside_grids(grids)
        if outside_grids.has_key(test_grid[0]):
            return True
        return False
        
    def get_inside_grids(self, grids):
        inside_grids = {}
        outside_grids = {}
        target_inside_neighbor_count = 2 * self.dimensions
        for indices, grid in grids.items():
            neighboring_grids = self.get_neighboring_grids(indices, grids)
            if len(neighboring_grids.keys()) == target_inside_neighbor_count:
                inside_grids[indices] = grid
            else:
                outside_grids[indices] = grid
                
        return inside_grids, outside_grids

    def get_neighboring_grids(self, ref_indices, cluster_grids = None):
        '''
        there is obvious room for optimization here using nicer data structures (BFS tree), right now will just test naive approach 
        '''    
        neighbors = {}
        
        per_dimension_possible_indices = np.array([])
        total_possible_neighbors = 1
        for i in range(self.dimensions):
            ref_index = ref_indices[i]
            possibles = np.array([])
            
            if ref_index == 0: 
                possibles = np.append(possibles, 1)
            elif ref_index == self.partitions_per_dimension[i] - 1:
                possibles = np.append(possibles, ref_index - 1)
            else:
                possibles = np.append(possibles, ref_index - 1)
                possibles = np.append(possibles, ref_index + 1)
            print 'possibles: ', possibles
            if per_dimension_possible_indices.size == 0:
                per_dimension_possible_indices = possibles
            else:    
                per_dimension_possible_indices = np.row_stack((per_dimension_possible_indices,possibles))#np.append(per_dimension_possible_indices, tuple(possibles))  
            print 'pers: ', per_dimension_possible_indices
            total_possible_neighbors *= possibles.size                    
            
        
        print 'possible indices: ', per_dimension_possible_indices        
        cartesian_product_of_possible_indices = cartesian(per_dimension_possible_indices)
        print 'cartesian product of possible indices tuple: ', cartesian_product_of_possible_indices
        
        for indices in cartesian_product_of_possible_indices:
            indices = tuple(indices)
            print 'testing if in cluster: ', indices
            if cluster_grids != None:
                print 'checking in cluster: ', cluster_grids.keys()
                if cluster_grids.has_key(indices):
                    grid = cluster_grids[indices]
                    neighbors[indices] = grid
            else:
                print 'checking in all grids: ', self.grids.keys()
                if self.grids.has_key(indices):
                    grid = self.grids[indices]
                    neighbors[indices] = grid
        
        return neighbors

    def assign_to_cluster_class(self, grids, class_key):
        for indices, grid in grids.items():
            grid.label = class_key
            self.grids[indices] = grid
            
    def get_grids_of_cluster_class(self, class_key):
        grids = {}
        for indices, grid in self.grids.items():
            if grid.label == class_key:
                grids[indices] = grid
                
        return grids                
    def get_most_recently_categorically_changed_grids(self):
        return_grids = {}
        for indices, grid in self.grids.items():
            if grid.category_changed_last_time == True:
                return_grids[indices] = grid
        return return_grids        
        
    
    def update_density_category(self):
        #self.last_updated_grids = {}
        for indices, grid in self.grids.items():
            if grid.density >= self.dense_threshold_parameter/(self.maximum_grid_count*(1.0-self.decay_factor)):
                if grid.density_category != 'DENSE':
                    grid.category_changed_last_time = True
                else:
                    grid.category_changed_last_time = False
                    
                grid.density_category = 'DENSE'
                print 'grid with indices: ', indices, ' is DENSE'
            if grid.density <= self.sparse_thresold_parameter/(self.maximum_grid_count*(1.0-self.decay_factor)):
                if grid.density_category != 'SPARSE':
                    grid.category_changed_last_time = True
                else:
                    grid.category_changed_last_time = False
                grid.density_category = 'SPARSE'
                print 'grid with indices: ', indices, ' is SPARSE'                
            if grid.density >= self.sparse_thresold_parameter/(self.maximum_grid_count*(1.0-self.decay_factor)) and grid.density <= self.dense_threshold_parameter/(self.maximum_grid_count*(1.0-self.decay_factor)):
                if grid.density_category != 'TRANSITIONAL':
                    grid.category_changed_last_time = True
                else:
                    grid.category_changed_last_time = False
                grid.density_category = 'TRANSITIONAL'
                print 'grid with indices: ', indices, ' is TRANSITIONAL' 
            self.grids[indices] = grid       
    def get_dense_grids(self):
        dense_grids = {}
        non_dense_grids = {}
        for indices, grid in self.grids.items():
            if grid.density_category == 'DENSE':
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
        print 'gap params: ', quotient1, quotient2, max_val, max_log, gap
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
        
        print 'getting dtf({}, {})'.format(last_update_time, current_time)
        top = self.sparse_thresold_parameter * (1.0 - self.decay_factor ** (current_time - last_update_time + 1))
        bottom = self.maximum_grid_count * (1.0 - self.decay_factor)
        return top/bottom
        
       
        
def cartesian(arrays, out=None):
    #from http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    #faster than itertools.combination (unverified)
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out    


if __name__ == "__main__":
    
    d_stream_clusterer = DStreamClusterer()
    for i in range(150):
        d_stream_clusterer.add_datum((25.4 + np.mod(i+1, 4), 13.1+np.mod(i, 20)))
    '''print d_stream_clusterer
    print 'indices for inserting 35.0, 100.0: ', d_stream_clusterer.get_grid_indices((35.0, 100.0))
    print 'indices for inserting 0.0, 60.0: ', d_stream_clusterer.get_grid_indices((0.0, 60.0))
    print 'getting dth for t=0, t=4', d_stream_clusterer.density_threshold_function(0, 4)
    print 'getting dth for t=0, t=8', d_stream_clusterer.density_threshold_function(0, 8)
    print 'getting dth for t=0, t=16', d_stream_clusterer.density_threshold_function(0, 16)    
    
    d_stream_characteristic_vector = DStreamCharacteristicVector ()
    print d_stream_characteristic_vector
    
    test_cart_array = np.array([[0], [2, 3], [4, 5, 6]])
    print 'test_cart_array: ', test_cart_array
    test_cart_tuple = tuple(tuple(x) for x in test_cart_array)
    print 'test_cart_tuple: ', test_cart_tuple
    test_cart_out = cartesian(test_cart_tuple)
    print 'test_cart_out: ', test_cart_out
    
    
    
    G=nx.empty_graph()
    G.add_edge(1,2)
    G.add_edge(2,3)
    G.add_edge(2,4)
    G.add_edge(4,5)
    G.add_edge(6,7)
    
    print 'path btw 1, 5: '
    print nx.bidirectional_dijkstra(G,1,5)
    print 'path btw 7, 5: '
    try:
        print nx.bidirectional_dijkstra(G,7,5)
    except nx.exception.NetworkXNoPath, e:
        print e
    
    fig = plt.figure()
    nx.draw(G)
    plt.show()'''