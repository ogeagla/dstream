# -*- coding: utf-8 -*-
'''import sys
sys.path.append('libs/networkx-1.7-py2.7.egg')'''

import numpy as np
import random

'''
Note: networkx is required. the easiest way to acquire the module is to either use my provided egg in ../lib 
or go to NetworkX's site and get it and do something like sudo easy_install *.egg
'''
import networkx as nx

'''
there are many other places where I can/should substitute my algs with the new NetworkX version using graphs

TODO: 

    -validate/check for 'None's in grid.label and grid.status.  they should NOT appear after a call to cluster.
    
    
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
                 category_changed_last_time=False,
                 label_changed_last_iteration=True):
        self.last_update_time = last_update_time
        self.last_sporadic_removal_time = last_sporadic_removal_time
        self.last_marked_sporadic_time = last_marked_sporadic_time
        self.density_category = density_category
        self.density = density
        self.label = label
        self.status = status
        self.category_changed_last_time = category_changed_last_time
        self.label_changed_last_iteration = label_changed_last_iteration


class DStreamClusterer():
    '''
    Initialize with defaults from reference algorithm
    '''

    def __init__(self,
                 dense_threshold_parameter=3.0,  # 3.0, #C_m
                 sparse_threshold_parameter=0.8,  # 0.8,  #C_l
                 sporadic_threshold_parameter=0.3,  # 0.3, #beta
                 decay_factor=0.998,  # 0.998, #lambda
                 dimensions=2,
                 domains_per_dimension=((0.0, 100.0), (0.0, 100.0)),
                 partitions_per_dimension=(5, 5),
                 initial_cluster_count=4,
                 seed=331):
        print 'provided: ', dense_threshold_parameter, sparse_threshold_parameter, sporadic_threshold_parameter, decay_factor, dimensions, domains_per_dimension, partitions_per_dimension
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
        # self.clusters = np.array([], dtype=type(DStreamCluster))
        self.cluster_count = initial_cluster_count

        self.gap_time = -1.0
        self.compute_gap_time()
        self.has_clustered_once = False

        self.current_time = 0
        # self.last_updated_grids = {}


        self.seed = seed
        self.class_keys = np.array([])
        random.seed(self.seed)

        self.data = np.array([])

    def get_density_nmatrix(self, grids):

        nmat = np.zeros(shape=self.partitions_per_dimension)
        for indices, grid in grids.items():
            # print indices
            nmat[indices] = grid.density
        return nmat

    def get_per_cluster_density_nmatrix_dict(self):

        ret_dict = {}
        for class_key in self.class_keys:
            grids = self.get_grids_of_cluster_class(class_key)
            grids_nmat = self.get_density_nmatrix(grids)
            ret_dict[class_key] = grids_nmat
        return ret_dict

    def add_datum(self, datum):

        if self.data.size == 0:
            self.data = np.array(datum)
        else:
            self.data = np.row_stack((self.data, np.array(datum)))
        # print 'current time: ', self.current_time, ' and gap time: ', self.gap_time
        # print 'adding: ', datum
        indices = tuple(self.get_grid_indices(datum))
        # print 'with indices: ', indices, tuple(indices)
        if self.grids.has_key(indices):
            grid = self.grids[indices]
        else:
            if self.removed_grid_cache.has_key(indices):
                grid = self.removed_grid_cache[indices]
            else:
                grid = DStreamCharacteristicVector()
        old_density = grid.density
        grid.density = 1.0 + grid.density * self.decay_factor ** (self.current_time - grid.last_update_time)
        # print 'grid at indices {} updated density from {} to {}'.format(indices, old_density, grid.density)

        grid.last_update_time = self.current_time
        self.grids[indices] = grid
        # print 'current time: ', self.current_time, ' ', indices, ' added to grids; all count: ', len(self.grids.keys())

        '''if self.current_time >= self.gap_time - 1:
            print 'clusters before: ', self.class_keys'''

        if self.current_time >= self.gap_time and not self.has_clustered_once:
            # print 'initializing clusters'
            self.initialize_clusters()
            # print 'clusters after: ', self.class_keys
        # print "current time, gap time: ", self.current_time, self.gap_time
        if np.mod(self.current_time, self.gap_time) == 0 and self.has_clustered_once:
            # print '\tCLUSTERING!'
            sporadic_grids = self.get_sporadic_grids()
            # print 'found ', len(sporadic_grids.keys()), ' sporadic grids'

            for indices, grid in sporadic_grids.items():
                if grid.last_marked_sporadic_time != -1 and grid.last_marked_sporadic_time + 1 <= self.current_time:
                    if grid.last_update_time != self.current_time:

                        self.grids = {key: value for key, value in self.grids.items() if value is not grid}
                        grid.last_sporadic_removal_time = self.current_time
                        self.removed_grid_cache[indices] = grid

                    else:
                        if self.is_sporadic(grid, self.current_time) == False:
                            grid.status = 'NORMAL'
                            self.grids[indices] = grid
            self.detect_sporadic_grids(self.current_time)
            self.cluster()
            # print 'clusters after: ', self.class_keys
        # print 'cluster count: ', self.class_keys.size
        self.current_time += 1
        # print 'grids after adding: ', self.grids

    def initialize_clusters(self):

        self.update_density_category()

        cluster_counts = np.array([])
        dense_grids, non_dense_grids = self.get_dense_grids()
        print 'dense count: {} non-dense count: {}'.format(len(dense_grids.keys()), len(non_dense_grids.keys()))

        if len(dense_grids.keys()) < self.cluster_count:
            print 'not enough dense clusters'
            # self.cluster_count = 0
            return

        cluster_size = np.round(len(dense_grids.keys()) / self.cluster_count)

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

        iter_count = 0
        last_label_changed_grids = self.get_last_label_changed()
        last_label_changed_grids_2 = self.get_last_label_changed()
        diff = np.setdiff1d(np.array(last_label_changed_grids.keys()), np.array(last_label_changed_grids_2.keys()))

        while iter_count == 0 or diff.size > 0:  # last_label_changed_grids.keys()#len(last_label_changed_grids.keys()) != 0:
            print 'iter_count: ', iter_count
            # raw_input('waiting on return')
            print last_label_changed_grids.keys(), (self.grids[last_label_changed_grids.keys()[0]]).label
            iter_count += 1
            for i in range(self.class_keys.size):

                class_key = self.class_keys[i]
                print 'class_key: ', class_key
                cluster_grids = self.get_grids_of_cluster_class(class_key)
                inside_grids, outside_grids = self.get_inside_grids(cluster_grids)
                print 'inside grid count: {} outside grid count: {} total grid_count: {}'.format(
                    inside_grids.keys().__len__(), outside_grids.keys().__len__(), self.grids.keys().__len__())
                for indices, grid in outside_grids.items():
                    neighboring_grids = self.get_neighboring_grids(indices)
                    for neighbor_indices, neighbor_grid in neighboring_grids.items():
                        print 'class key sizes: ', self.class_keys.size
                        for j in range(self.class_keys.size):
                            print j
                            test_class_key = self.class_keys[j]
                            test_cluster_grids = self.get_grids_of_cluster_class(test_class_key)

                            neighbor_belongs_to_test_cluster = self.validate_can_belong_to_cluster(test_cluster_grids, (
                            neighbor_indices, neighbor_grid))
                            self.reset_last_label_changed()
                            if neighbor_belongs_to_test_cluster:
                                if len(cluster_grids.keys()) > len(test_cluster_grids.keys()):
                                    self.assign_to_cluster_class(test_cluster_grids, class_key)
                                else:
                                    self.assign_to_cluster_class(cluster_grids, test_class_key)
                            elif neighbor_grid.density_category == 'TRANSITIONAL':
                                self.assign_to_cluster_class({neighbor_indices: neighbor_grid}, class_key)
                                # self.update_class_keys()
            last_label_changed_grids_2 = last_label_changed_grids
            last_label_changed_grids = self.get_last_label_changed()
            diff = np.setdiff1d(np.array(last_label_changed_grids.keys()), np.array(last_label_changed_grids_2.keys()))
        self.has_clustered_once = True

    def cluster(self):
        self.update_density_category()
        for indices, grid in self.get_most_recently_categorically_changed_grids().items():

            neighboring_grids = self.get_neighboring_grids(indices)
            neighboring_clusters = {}
            if neighboring_grids != None:

                for neighbor_indices, neighbor_grid in neighboring_grids.items():
                    neighbors_cluster_class = neighbor_grid.label
                    neighbors_cluster_grids = self.get_grids_of_cluster_class(neighbors_cluster_class)
                    neighboring_clusters[neighbor_indices, neighbors_cluster_class] = neighbors_cluster_grids
                if len(neighboring_grids.keys()) != 0:
                    max_neighbor_cluster_size = 0
                    max_size_indices = None
                    # max_size_cluster = None
                    # print 'neighboring clusters: ', neighboring_clusters.keys()
                    for k, ref_neighbor_cluster_grids in neighboring_clusters.items():
                        test_size = len(ref_neighbor_cluster_grids.keys())
                        # print 'size comparison: ', test_size, max_neighbor_cluster_size
                        if test_size > max_neighbor_cluster_size:
                            max_neighbor_cluster_size = test_size
                            # max_size_cluster = neighbor_cluster
                            max_size_cluster_key = k[1]
                            max_size_indices = k[0]
                            max_cluster_grids = ref_neighbor_cluster_grids
                    max_size_grid = neighboring_grids[max_size_indices]
                    grids_cluster = self.get_grids_of_cluster_class(grid.label)

            if grid.density_category == 'SPARSE':
                changed_grid_cluster_class = grid.label
                cluster_grids_of_changed_grid = self.get_grids_of_cluster_class(changed_grid_cluster_class)
                # print 'cluster grids of changed grid keys: ', cluster_grids_of_changed_grid.keys()
                would_still_be_connected = self.cluster_still_connected_upon_removal(cluster_grids_of_changed_grid,
                                                                                     (indices, grid))
                grid.label = 'NO_CLASS'
                self.grids[indices] = grid

                if would_still_be_connected == False:
                    self.extract_two_clusters_from_grids_having_just_removed_given_grid(cluster_grids_of_changed_grid,
                                                                                        (indices, grid))

            elif grid.density_category == 'DENSE':
                if len(neighboring_clusters.keys()) == 0:
                    print 'no neighbors, returning'
                    return
                if max_size_grid.density_category == 'DENSE':

                    if grid.label == 'NO_CLASS':
                        grid.label = max_size_cluster_key
                        self.grids[indices] = grid
                    elif len(grids_cluster.keys()) > max_neighbor_cluster_size:
                        if grid.label != 'NO_CLASS':
                            for max_indices, max_grid in max_cluster_grids.items():
                                max_grid.label = grid.label
                                self.grids[max_indices] = max_grid
                    elif len(grids_cluster.keys()) <= max_neighbor_cluster_size:
                        if max_size_cluster_key != 'NO_CLASS':
                            for grids_cluster_indices, grids_cluster_grid in grids_cluster.items():
                                grids_cluster_grid.label = max_size_cluster_key
                                self.grids[grids_cluster_indices] = grids_cluster_grid
                elif max_size_grid.density_category == 'TRANSITIONAL':
                    if grid.label == 'NO_CLASS' and self.grid_becomes_outside_if_other_grid_added_to_cluster(
                            (max_size_indices, max_size_grid), max_cluster_grids, (indices, grid)):
                        grid.label = max_size_cluster_key
                        self.grids[indices] = grid
                    elif len(grids_cluster.keys()) >= max_neighbor_cluster_size:
                        max_size_grid.label = grid.label
                        self.grids[max_size_indices] = max_size_grid
            elif grid.density_category == 'TRANSITIONAL':
                if len(neighboring_clusters.keys()) == 0:
                    print 'no neighbors, returning'
                    return
                max_outside_cluster_size = 0
                max_outside_cluster_class = None
                for k, ref_neighbor_cluster_grids in neighboring_clusters.items():
                    ref_cluster_key = k[1]
                    # ref_indices = k[0]
                    ref_grids = ref_neighbor_cluster_grids
                    if self.grid_is_outside_if_added_to_cluster((indices, grid), ref_grids) == True:
                        test_size = len(ref_grids.keys())
                        if test_size > max_outside_cluster_size:
                            max_outside_cluster_size = test_size
                            max_outside_cluster_class = ref_cluster_key
                grid.label = max_outside_cluster_class
                self.grids[indices] = grid

            self.update_class_keys()
        self.split_and_merge()

    def create_new_cluster(self, grids):
        unique_class_key = self.generate_unique_class_key()
        # print 'creating new cluster with key: ', unique_class_key, ', uniqueness : ', not unique_class_key in self.class_keys
        for indices, grid in grids.items():
            # print 'grid goes from {} to {}'.format(grid.label, unique_class_key)
            grid.label = unique_class_key
            self.grids[indices] = grid

    def split(self):
        for class_key in self.class_keys:
            if class_key == None or class_key == '' or class_key == []:
                # print class_key, ' is no good class key'
                continue
            # print 'attemping split of class {}'.format(class_key)
            cluster = self.get_grids_of_cluster_class(class_key)
            cluster_graph = self.get_graph_of_cluster(cluster)
            # print 'cluster graph size {}'.format(len(cluster_graph.nodes()))

            if len(cluster_graph.nodes()) == 0:
                for indices, grid in cluster.items():
                    print indices, grid.label
                # print 'null graph, splitting each grid into cluster'
                # raw_input()
                for indices, grid in cluster.items():
                    self.create_new_cluster({indices: grid})
                continue
            subgraphs = nx.connected_component_subgraphs(cluster_graph)
            if len(subgraphs) != 1:

                print 'SPLIT', cluster.keys(), 'into {} clusters'.format(len(subgraphs))
                '''print 'subgraph count: ', len(subgraphs)
                print 'class keys before: ', self.class_keys.size, self.class_keys'''
                '''fig = plt.figure()
                nx.draw(cluster_graph)
                plt.show()'''
                for subgraph in subgraphs:
                    nodes = subgraph.nodes()
                    # print 'nodes: ', nodes
                    new_grids = {}
                    for node in nodes:
                        new_grids[node] = self.grids[node]
                    self.create_new_cluster(new_grids)
                # print 'class keys after: ', self.class_keys.size, self.class_keys
                self.update_class_keys()
                self.split()

                # else:
                # print 'subgraphs size one'

    def merge(self):

        for class_key in self.class_keys:
            # print 'attempting merge of cluster class ', class_key
            cluster = self.get_grids_of_cluster_class(class_key)
            if len(cluster.keys()) != 0:
                cluster_graph = self.get_graph_of_cluster(cluster)
                # print 'cluster graph size ', len(cluster_graph.nodes())
                for test_class_key in self.class_keys:
                    if test_class_key != class_key:
                        test_cluster = self.get_grids_of_cluster_class(test_class_key)
                        if len(test_cluster.keys()) != 0:
                            if self.is_valid_cluster(dict(cluster.items() + test_cluster.items())) == False:
                                continue
                            test_cluster_graph = self.get_graph_of_cluster(test_cluster)

                            cg_copy = cluster_graph.copy()
                            # print 'adding {} of size {} to {} of size {}'.format(test_cluster_graph.nodes(), len(test_cluster_graph.nodes()), cg_copy.nodes(), len(cg_copy.nodes()))
                            cg_copy.add_edges_from(test_cluster_graph.edges())
                            if len(test_cluster_graph.nodes()) == 1:
                                cg_copy.add_node(test_cluster_graph.nodes()[0])
                            for node in cg_copy.nodes():

                                for test_node in cg_copy.nodes():
                                    if test_node != node:
                                        if self.are_neighbors(node, test_node):
                                            if cg_copy.has_edge(node, test_node) == False:
                                                # print 'adding edge ', (node, test_node)
                                                cg_copy.add_edge(node, test_node)

                            subgraphs = nx.connected_component_subgraphs(cg_copy)
                            # print 'results has {} subgraphs'.format(len(subgraphs))


                            if (len(subgraphs) == 1):

                                print 'MERGE', cluster.keys(), test_cluster.keys()

                                '''for subgraph in subgraphs:
                                        
                                        fig = plt.figure()
                                        nx.draw(subgraph)
                                    plt.show()'''
                                # print type(subgraphs), len(subgraphs), subgraphs
                                '''fig = plt.figure()
                                    nx.draw(subgraphs[0])                                        
                                    #nx.draw(self.get_graph_of_cluster(dict(cluster.items() + test_cluster.items())))
                                    plt.show()'''
                                # print 'before ', len(self.class_keys)
                                if len(cluster.keys()) > len(test_cluster.keys()):
                                    self.assign_to_cluster_class(test_cluster, class_key)
                                else:
                                    self.assign_to_cluster_class(cluster, test_class_key)
                                self.update_class_keys()
                                # print 'after ', len(self.class_keys)
                                self.merge()

    def split_and_merge(self):

        # print 'merging'
        '''print 'all graph'
        all_graph = self.get_graph_of_cluster(self.grids)
        fig = plt.figure()
        nx.draw(all_graph)
        plt.show()'''
        # ClusterDisplay2D.display_all(self.grids, self.class_keys, self.data, self.partitions_per_dimension, self.domains_per_dimension, 'in merge', False)
        '''
        graph-based split
        '''
        self.split()

        '''
        graph-based merge
        '''

        self.merge()

        self.update_class_keys()

    def extract_two_clusters_from_grids_having_just_removed_given_grid(self, grids_without_removal, removed_grid):
        # first remove it, then split into two, then add the two to self.grids
        removed_grid_indices = removed_grid[0]

        grids_with_removal = {key: value for key, value in grids_without_removal.items() if
                              key is not removed_grid_indices}
        # print 'ex2 grids with removal: ', grids_with_removal
        graph_with_removal = self.get_graph_of_cluster(grids_with_removal)
        subgraphs = nx.connected_component_subgraphs(graph_with_removal)

        if len(subgraphs) != 2:
            pass  # print 'found != 2 subgraphs; count: ', len(subgraphs)

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
        # print 'removal grid indices: ', removal_grid_indices
        grids_with_removal = {key: value for key, value in grids_without_removal.items() if
                              key is not removal_grid_indices}

        # print 'connect grids with removal keys: ', grids_with_removal.keys()

        graph_with_removal = self.get_graph_of_cluster(grids_with_removal)
        if graph_with_removal.size() == 0:
            return False
        return nx.is_connected(graph_with_removal)

    def get_graph_of_cluster(self, grids):
        # print '%%%%%%%%%%%%%%%%%%%%%is valid: ', self.is_valid_cluster(grids), ' %%%'
        # print 'graph of cluster grids keys: ', grids.keys()
        indices_list = grids.keys()
        g = nx.empty_graph()
        for i in range(len(indices_list)):
            indices = indices_list[i]
            # print 'indices: ', indices
            for j in range(len(indices_list)):
                other_indices = indices_list[j]
                # print 'other_indices: ', other_indices
                # print 'i, oi: ', indices, other_indices
                if self.are_neighbors(indices, other_indices):
                    # print '***** ', indices, other_indices, ' ARE neighbors'
                    if g.has_edge(indices, other_indices) == False:
                        g.add_edge(indices, other_indices)
                        continue
                g.add_node(other_indices)
            if g.has_node(indices) == False:
                g.add_node(indices)

        # print 'g size {}'.format(g.size())


        return g

    def are_neighbors(self, grid1_indices, grid2_indices):
        target_identical_count = self.dimensions - 1
        identical_count = 0
        for i in range(self.dimensions):
            # print grid1_indices[i], grid2_indices[i]
            if grid1_indices[i] == grid2_indices[i]:
                identical_count += 1
            elif np.abs(grid1_indices[i] - grid2_indices[i]) != 1:
                return False
        return identical_count == target_identical_count

    def validate_can_belong_to_cluster(self, cluster, test_grid):
        # first validate cluster?
        # print 'checking if grid can be valid in cluster. first doing pre-addition check'
        is_valid_before = self.is_valid_cluster(cluster)
        if is_valid_before != True:
            print 'provided cluster is invalid...returning False'
            return False

        cluster[test_grid[0]] = test_grid[1]
        is_valid_after = self.is_valid_cluster(cluster)
        return is_valid_after

    def is_valid_cluster(self, grids):

        for indices, grid in grids.items():
            for indices2, grid2 in grids.items():
                if indices != indices2:
                    if not self.are_neighbors(indices, indices2):
                        # print 'grids not neighbors! ' , indices, indices2
                        return False

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
        grids = grids.copy()
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
            if neighboring_grids != None:
                if len(neighboring_grids.keys()) == target_inside_neighbor_count:
                    inside_grids[indices] = grid
                else:
                    outside_grids[indices] = grid

        return inside_grids, outside_grids

    def get_neighboring_grids(self, ref_indices, cluster_grids=None):
        '''
        there is obvious room for optimization here using nicer data structures (BFS tree), right now will just test naive approach 
            -update: should implemnt this using a graph of all the edges. then can take a node and get all edges to get neighboring nodes and viola
        '''

        to_get_neighbors_amongst = {}

        if cluster_grids != None:
            to_get_neighbors_amongst = cluster_grids
        else:
            to_get_neighbors_amongst = self.grids
        to_get_neighbors_amongst[ref_indices] = self.grids[ref_indices]
        to_get_neighbors_amongst_graph = self.get_graph_of_cluster(to_get_neighbors_amongst)
        neighbors = to_get_neighbors_amongst_graph.neighbors(ref_indices)
        neighbors_dict = {}
        for i in range(len(neighbors)):
            indices = neighbors[i]
            neighbors_dict[indices] = self.grids[indices]
        if len(neighbors_dict) == 0:
            return None
        return neighbors_dict

        '''
        neighbors = {}
        #print 'ref indices: ', ref_indices
        
        per_dimension_possible_indices = {}
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
            #print 'possibles: ', possibles
            
            per_dimension_possible_indices[i] = possibles
            
            #print 'pers: ', per_dimension_possible_indices
            total_possible_neighbors *= possibles.size       

        possible_indices = np.array([])          
        #print 'per dim poss: ', per_dimension_possible_indices
        for i in range(self.dimensions):
            
            possibles = per_dimension_possible_indices[i]
            #print 'dim, possible indices, ref', i, possibles, ref_indices[i]
            for j in range(possibles.size):
                ref_new = np.array(copy.deepcopy(ref_indices))
                ref_new[i] = possibles[j]
                #print 'new, _:', ref_new, ref_indices
                if possible_indices.size == 0:
                    possible_indices = ref_new
                else:
                    possible_indices = np.row_stack((possible_indices, ref_new))#np.append(possible_indices, tuple(ref_new))
                #print possible_indices
                
        for indices in possible_indices:
            indices = tuple(indices)
            #print 'testing if in cluster: ', indices
            if cluster_grids != None:
                #print 'checking in cluster: ', cluster_grids.keys()
                if cluster_grids.has_key(indices):
                    grid = cluster_grids[indices]
                    neighbors[indices] = grid
            else:
                #print 'checking in all grids: ', self.grids.keys()
                if self.grids.has_key(indices):
                    grid = self.grids[indices]
                    neighbors[indices] = grid
        #print 'neighbors keys: ', neighbors.keys()
        return neighbors'''

    def get_last_label_changed(self):
        grids = {}
        for indices, grid in self.grids.items():
            if grid.label_changed_last_iteration == True:
                grids[indices] = grid
        return grids

    def reset_last_label_changed(self):
        for indices, grid in self.grids.items():
            grid.label_changed_last_iteration = False
            self.grids[indices] = grid

    def assign_to_cluster_class(self, grids, class_key):
        for indices, grid in grids.items():
            if grid.label != class_key:
                grid.label_changed_last_iteration = True
                # print grid.label, class_key
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
        # self.last_updated_grids = {}
        for indices, grid in self.grids.items():
            dense_param = self.dense_threshold_parameter / (self.maximum_grid_count * (1.0 - self.decay_factor))
            sparse_param = self.sparse_thresold_parameter / (self.maximum_grid_count * (1.0 - self.decay_factor))
            test_density = grid.density
            # print 'test density {} dense thresh {} sparse thresh {}'.format(test_density, dense_param, sparse_param)
            if test_density >= dense_param:
                if grid.density_category != 'DENSE':
                    grid.category_changed_last_time = True
                else:
                    grid.category_changed_last_time = False

                grid.density_category = 'DENSE'
                # print 'grid with indices: ', indices, ' is DENSE'
            if test_density <= sparse_param:
                if grid.density_category != 'SPARSE':
                    grid.category_changed_last_time = True
                else:
                    grid.category_changed_last_time = False
                grid.density_category = 'SPARSE'
                # print 'grid with indices: ', indices, ' is SPARSE'
            if test_density >= sparse_param and grid.density <= dense_param:
                if grid.density_category != 'TRANSITIONAL':
                    grid.category_changed_last_time = True
                else:
                    grid.category_changed_last_time = False
                grid.density_category = 'TRANSITIONAL'
                # print 'grid with indices: ', indices, ' is TRANSITIONAL'
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
        sporadic_grids = {}  # np.array([], type(DStreamCharacteristicVector))
        for indices, grid in self.grids.items():
            if grid.status == 'SPORADIC':
                sporadic_grids[indices] = grid  # np.append(sporadic_grids, d_stream_characteristic_vector)
        return sporadic_grids

    def is_sporadic(self, grid, current_time):
        if grid.density < self.density_threshold_function(grid.last_update_time, current_time) and current_time >= (
            1.0 + self.sporadic_threshold_parameter) * grid.last_sporadic_removal_time:
            return True
        return False

    def detect_sporadic_grids(self, current_time):
        for indices, grid in self.grids.items():
            if self.is_sporadic(grid, current_time):
                # if d_stream_characteristic_vector.density < self.density_threshold_function(d_stream_characteristic_vector.last_update_time, current_time) and current_time >= (1.0 + self.sporadic_threshold_parameter)*d_stream_characteristic_vector.last_sporadic_removal_time:
                # print 'detected sporadic grid at indices: ', indices
                grid.status = 'SPORADIC'
                grid.last_marked_sporadic_time = current_time
            else:
                grid.status = 'NORMAL'
            self.grids[indices] = grid

    def compute_gap_time(self):

        quotient1 = self.sparse_thresold_parameter / self.dense_threshold_parameter
        quotient2 = (self.maximum_grid_count - self.dense_threshold_parameter) / (
        self.maximum_grid_count - self.sparse_thresold_parameter)
        max_val = np.max([quotient1, quotient2])
        max_log = np.log(max_val) / np.log(self.decay_factor)
        gap = np.floor([max_log])
        self.gap_time = gap[0]
        print 'gap params: ', quotient1, quotient2, max_val, max_log, gap
        print 'computed gap time: ', self.gap_time

    def update_class_keys(self):
        new_keys = np.array([])
        # print 'updating class keys: ', self.class_keys
        for indices, grid in self.grids.items():
            if grid.label not in new_keys and grid.label != 'NO_CLASS' and grid.label != None:
                # print 'new class key ', grid.label
                new_keys = np.append(new_keys, grid.label)
        self.class_keys = new_keys

    def generate_unique_class_key(self):
        test_key = np.int(np.round(random.uniform(0, 1), 8) * 10 ** 8)
        while test_key in self.class_keys:
            # print 'class key test collision...weird'
            test_key = np.int(np.round(random.uniform(0, 1), 8) * 10 ** 8)

        return test_key

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
            domain_size = domain_tuple[1] - domain_tuple[0]
            test_datum = datum[i] - domain_tuple[0]
            index = np.floor([(test_datum / domain_size) * (partitions)])[0]
            if index >= partitions:
                print 'index equals partitions: ', index, partitions
                index = partitions - 1
            indices = np.append(indices, index)
        return indices

    def density_threshold_function(self, last_update_time, current_time):

        # print 'getting dtf({}, {})'.format(last_update_time, current_time)
        top = self.sparse_thresold_parameter * (1.0 - self.decay_factor ** (current_time - last_update_time + 1))
        bottom = self.maximum_grid_count * (1.0 - self.decay_factor)
        return top / bottom
