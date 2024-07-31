# Tests for urbansurge/sensing/sensor_utils.py

# Imports.
from urbansurge.sensing import sensor_utils

class TestSensorUtils():
    def setup(self):
        self.in_filepath = r"test_files\lab_system.inp"

    def test_conduit_nodes(self):
        conduit_nodes_dict = sensor_utils.link_nodes(self.in_filepath)
        print(conduit_nodes_dict)

        assert isinstance(conduit_nodes_dict, dict)
        assert len(conduit_nodes_dict.keys()) > 0

    def test_adjacency_matrix(self):
        conduit_nodes_dict = sensor_utils.link_nodes(self.in_filepath)

        A, node_names = sensor_utils.adjacency_matrix(conduit_nodes_dict, self.in_filepath)

        # Ensure A is not an empty array.
        assert A.any()

        # Check if node_names is a list with items.
        assert len(node_names) > 0

    def test_upstream_assign(self):
        # Conduit nodes dictionary.
        link_node_dict = sensor_utils.link_nodes(self.in_filepath, int_convert=False)

        # Sensor nodes.
        sensor_nodes = sensor_utils.upstream_assign(link_node_dict, self.in_filepath, Nups=2)


    def test_upstream_assign_links(self):
        # Link nodes dictionary.
        link_node_dict = sensor_utils.link_nodes(self.in_filepath, int_convert=False)

        # Sensor links.
        sensor_links = sensor_utils.upstream_assign_links(link_node_dict, self.in_filepath, Nups=1, exclude_weirs=True)

        print(sensor_links)