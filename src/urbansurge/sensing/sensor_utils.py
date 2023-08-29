###########################################################################
# Utility functions for sensor operations.
###########################################################################

# Library imports.
import numpy as np

# Local imports.
from urbansurge import file_utils


def conduit_nodes(in_filepath):
    # Get the names of each conduit.
    conduit_names = file_utils.get_component_names(in_filepath, 'CONDUITS')

    # Get the end nodes of each conduit.
    conduit_nodes_dict = {}
    for name in conduit_names:
        from_node = file_utils.get_inp_section(in_filepath, 'CONDUITS', 'From Node', name)
        to_node = file_utils.get_inp_section(in_filepath, 'CONDUITS', 'To Node', name)

        # Populate dictionary as {conduit_name: (from_node, to_node), ...}
        conduit_nodes_dict[name] = (from_node, to_node)

    return conduit_nodes_dict


def adjacency_matrix(conduit_nodes_dict, in_filepath):

    # All node names.
    node_names = file_utils.get_component_names(in_filepath, 'JUNCTIONS')

    # Outfall names.
    outfall_names = file_utils.get_component_names(in_filepath, 'OUTFALLS')

    # Append outfall names to node names.
    node_names.extend(outfall_names)

    # Create an empty node adjacency matrix.
    A = np.zeros((len(node_names), len(node_names)))

    # Loop through conduit-node dictionary and assign adjacencies.
    for cname, nodes in conduit_nodes_dict.items():
        # Get node indices.
        from_node_idx = node_names.index(nodes[0])
        to_node_idx = node_names.index(nodes[1])

        # Update adjacency matrix.
        A[from_node_idx, to_node_idx] = 1

    return A, node_names