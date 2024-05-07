###########################################################################
# Utility functions for sensor operations.
###########################################################################

# Library imports.
import networkx as nx
import numpy as np
import pandas as pd

# Local imports.
from urbansurge import file_utils


def link_nodes(inp_filepath, int_convert=False):
    """
    Create a dictionary of edge: (start_node, end_node) from a swmm network.
    :param inp_filepath: Swmm input filepath.
    :param int_convert: Convert edge and node IDs/names to int.
    :return:
    """
    # Get the names of each conduit and weir.
    conduit_names = file_utils.get_component_names(inp_filepath, 'CONDUITS')
    weir_names = file_utils.get_component_names(inp_filepath, 'WEIRS')

    # Dictionary to store link nodes.
    link_nodes_dict = {}

    # Get the end nodes of each conduit.
    for name in conduit_names:
        from_node = file_utils.get_inp_section(inp_filepath, 'CONDUITS', 'From Node', name)
        to_node = file_utils.get_inp_section(inp_filepath, 'CONDUITS', 'To Node', name)

        # Populate dictionary as {conduit_name: (from_node, to_node), ...}
        if int_convert is True:
            name = int(name)
            from_node = int(from_node)
            to_node = int(to_node)
        link_nodes_dict[name] = (from_node, to_node)

    # Get the end nodes from each weir.
    if weir_names:
        for name in weir_names:
            from_node = file_utils.get_inp_section(inp_filepath, 'WEIRS', 'From Node', name)
            to_node = file_utils.get_inp_section(inp_filepath, 'WEIRS', 'To Node', name)

            # Populate dictionary as {conduit_name: (from_node, to_node), ...}
            if int_convert is True:
                name = int(name)
                from_node = int(from_node)
                to_node = int(to_node)
            link_nodes_dict[name] = (from_node, to_node)

    return link_nodes_dict


def adjacency_matrix(conduit_nodes_dict, inp_filepath, include_cnames=True):
    """
    Create adjacency matrix of network.
    :param conduit_nodes_dict: Conduit node dictionary. Output from conduit_nodes().
    :param inp_filepath: Initialization filepath (.inp) for EPASWMM model.
    :param include_cnames: Instead of 1 for adjacent nodes, add the conduit ID as an integer.
    :return: Tuple (adjacency matrix, node names)
    """

    # All node names.
    node_names = file_utils.get_component_names(inp_filepath, 'JUNCTIONS')

    # Outfall names.
    outfall_names = file_utils.get_component_names(inp_filepath, 'OUTFALLS')
    node_names.extend(outfall_names)

    # Add storage nodes.
    storage_ids = file_utils.get_component_names(inp_filepath, 'STORAGE')
    node_names.extend(storage_ids)

    # Create an empty node adjacency matrix.
    A = np.zeros((len(node_names), len(node_names)))

    # Loop through conduit-node dictionary and assign adjacencies.
    for cname, nodes in conduit_nodes_dict.items():
        # Get node indices.
        from_node_idx = node_names.index(nodes[0])
        to_node_idx = node_names.index(nodes[1])

        # Update adjacency matrix.
        A[from_node_idx, to_node_idx] = int(cname)

    return A, node_names


def upstream_assign(A_matrix, node_names, Nups=2):
    """
    Assign sensor locations by the number of upstream components.
    :param A_matrix: Adjacency matrix [numpy array]
    :param node_names: Node names [list]
    :param Nups: Number of upstream components to prune by.
    :return sensor_nodes: Node ids where sensors should go.
    """
    # G is the directed graph of the network.
    G_node_names = node_names
    # node_name_idx = {i:node_names[i] for i in range(len(node_names))}

    # Directed graph of the network.
    G = nx.from_numpy_array(A_matrix, create_using=nx.DiGraph)
    nx.relabel_nodes(G, dict(zip(G.nodes, G_node_names)), copy=False)

    # Reversed digraph.
    H = G.reverse()

    # Assign sensors based on number of upstream links.
    sensor_nodes = []
    max_upstream = Nups + 1  # Maximum number of components upstream of any node.
    H_node_names = np.array(G_node_names.copy())

    while max_upstream >= Nups:
        # # Number of upstream nodes for each sensor.
        # upstream_edges = np.zeros((len(H_node_names), 2), dtype=int)
        # upstream_edges[:, 0] = H_node_names
        # for i, node_name in enumerate(H_node_names):
        #     n_upstream = list(nx.bfs_tree(H, source=node_name).nodes())
        #     upstream_edges[i, 1] = len(n_upstream) - 1  # -1 to not include node.

        # Number of upstream edges for each node.
        upstream_edges = np.zeros(len(H_node_names))
        for i, node_name in enumerate(H_node_names):
            n_upstream = list(nx.bfs_tree(H, source=node_name).nodes())
            upstream_edges[i] = len(n_upstream) - 1  # -1 to not include node.

        # Maximum upstream nodes.
        max_upstream = np.max(upstream_edges)

        # If the maximum number of upstream nodes is less than Nupstream, add
        # the outfall as the final node and break the loop.
        if max_upstream < Nups:
            most_upstream_idx = np.argmax(upstream_edges)
            sensor_nodes.append(H_node_names[most_upstream_idx])
            break

        # Choose a node with the number of upstream edges closest to Nupstream.
        # Choose randomly if there is more than 1. Increment the search number if
        # there are no nodes found with N = Nupstream.
        sensor_node = None
        search_num = Nups
        while sensor_node is None:
            # Array of potential sensors where the number of upstream edges is
            # equal to search_num.
            # potential_sensors = upstream_edges[upstream_edges[:, 1] == search_num, 0]
            potential_sensors = H_node_names[upstream_edges == search_num]

            if len(potential_sensors) == 0:
                # If there are no potential sensors, increment search_num.
                search_num += 1
            elif len(potential_sensors) > 1:
                # Randomly choose sensor if there is more than 1 option.
                sensor_node = np.random.choice(potential_sensors)
            else:
                sensor_node = potential_sensors[0]

        # Add sensor to list of sensor nodes.
        sensor_nodes.append(sensor_node)

        # Prune graph at the potential sensor.
        sensor_upstream = list(nx.bfs_tree(H, source=sensor_node).nodes())
        for upstream_node in sensor_upstream:
            # Remove all nodes upstream of sensor node.
            if upstream_node != sensor_node:
                H.remove_node(upstream_node)

                # Remove node name from node name list.
                H_node_names = H_node_names[H_node_names != upstream_node]

    return sensor_nodes

