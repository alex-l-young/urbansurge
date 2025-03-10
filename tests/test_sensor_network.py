# Test class for urbansurge/sensor_network.py
# ==============================================

from urbansurge import sensor_network

class TestSensorNetwork:
    def setup(self):
        self.inp_filepath = r"/Users/alexyoung/Desktop/Cornell/Research/urbansurge/analysis/Bellinge/7_SWMM/BellingeSWMM_v021_nopervious.inp"

    def test_link_nodes(self):
        conduit_nodes_dict = sensor_network.link_nodes(self.inp_filepath, int_convert=False)
        
        assert conduit_nodes_dict

    def test_adjacency_matrix(self):
        conduit_nodes_dict = sensor_network.link_nodes(self.inp_filepath, int_convert=False)
        A, node_names = sensor_network.adjacency_matrix(conduit_nodes_dict, self.inp_filepath)

        assert len(node_names) > 0
        assert A.shape == (len(node_names), len(node_names))

