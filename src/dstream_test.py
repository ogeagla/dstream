import unittest
import networkx as nx
import dstream

class DStreamTests(unittest.TestCase):
    
    def setUp(self):
        print 'setting up'
        self.dstream = dstream.DStreamClusterer(dense_threshold_parameter=3.0, 
                                                sparse_threshold_parameter=0.8, 
                                                sporadic_threshold_parameter=0.3, 
                                                decay_factor=0.998, 
                                                dimensions=4, 
                                                domains_per_dimension=((0.0, 10.0), (0.0, 100.0), (0.0, 25.0), (50.0, 150.0)), 
                                                partitions_per_dimension=(2, 10, 5, 20), 
                                                initial_cluster_count=4, 
                                                seed=331)
        
        
        
    def tearDown(self):
        print 'tearing down'
        self.dstream = None
        
    def test_nothing(self):
        print 'testing nothing'
    
    def test_get_grid_indices(self):
        
        test_datum_1 = (0.0, 0.0, 0.0, 50.0)
        test_indices_1 = self.dstream.get_grid_indices(test_datum_1)
        correct_indices_1 = (0, 0, 0, 0)
        
        
        test_datum_2 = (10.0, 100.0, 25.0, 150.0)
        test_indices_2 = self.dstream.get_grid_indices(test_datum_2)
        correct_indices_2 = (1, 9, 4, 19)
        
        
#         print 'found indices {} for datum {}; correct indices are: {}'.format(test_indices_1, test_datum_1, correct_indices_1)
        for i in range(len(test_datum_1)):
            
            self.assertTrue(test_indices_1[i] == correct_indices_1[i])
                        
            self.assertTrue(test_indices_2[i] == correct_indices_2[i])
          
            
            