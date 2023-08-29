# Tests for urbansurge/sensing/sensor_utils.py

# Imports.
from urbansurge.sensing import sensor_utils

class TestSensorUtils():
    def setup(self):
        self.in_filepath = r"test_files\Canandaigua.inp"

    def test_conduit_nodes(self):
        conduit_nodes_dict = sensor_utils.conduit_nodes(self.in_filepath)
        print(conduit_nodes_dict)

        assert isinstance(conduit_nodes_dict, dict)
        assert len(conduit_nodes_dict.keys()) > 0

    def test_adjacency_matrix(self):
        conduit_nodes_dict = sensor_utils.conduit_nodes(self.in_filepath)

        A, node_names = sensor_utils.adjacency_matrix(conduit_nodes_dict, self.in_filepath)

        # Ensure A is not an empty array.
        assert A.any()

        # Check if node_names is a list with items.
        assert len(node_names) > 0
