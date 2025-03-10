# Stormwater system sensor network.
# Alex Young
# ========================================================

# Library imports.
import numpy as np
from typing import List, Dict

# UrbanSurge imports.
from urbansurge import file_utils


def dfs_upstream(adj_matrix: List[List[int]], start_node: int) -> List[int]:
    """
    Perform a depth-first search upstream in a stormwater sewer network graph.
    
    :param adj_matrix: Adjacency matrix representing the network. Each entry A_ij is:
                       - 1 if flow is from i to j,
                       - -1 if flow is from j to i,
                       - 0 otherwise.
    :type adj_matrix: List[List[int]]
    :param start_node: The starting node for the traversal.
    :type start_node: int
    :return: The order of visited nodes in the upstream traversal.
    :rtype: List[int]
    
    **Example usage:**
    
    .. code-block:: python
    
        adj_matrix = [
            [ 0,  1,  0,  0],
            [-1,  0,  1,  0],
            [ 0, -1,  0,  1],
            [ 0,  0, -1,  0]
        ]
        start_node = 3
        print(dfs_upstream(adj_matrix, start_node))
        # Output: [3, 2, 1, 0]
    """
    num_nodes = len(adj_matrix)
    visited = set()
    traversal_order = []
    
    def dfs(node: int) -> None:
        if node in visited:
            return
        visited.add(node)
        traversal_order.append(node)
        
        for neighbor in range(num_nodes):
            if adj_matrix[neighbor][node] == 1:  # Flow from neighbor to node (upstream direction)
                dfs(neighbor)
    
    dfs(start_node)
    return traversal_order


def dfs_surcharge(adj_matrix: List[List[int]], start_node: int, invert_elevations: Dict, surcharge_depths: Dict) -> Dict:
    # Dictionary of exceedance lists for each node.
    exceed_dict = {start_node: {}}

    num_nodes = len(adj_matrix)
    visited = set()
    traversal_order = []
    
    def dfs(node: int, L: Dict) -> None:
        if node in visited:
            return
        
        visited.add(node)
        traversal_order.append(node)

        # Get current node invert elevation.
        inv_elev = invert_elevations[node]

        # Keep nodes in L where surcharge depth > invert elevation.
        L = {k: v for k, v in L.items() if v > inv_elev}
        
        # Add current node's surcharge depth.
        L[node] = surcharge_depths[node]

        # Add L to exceedance dictionary.
        exceed_dict[node] = L
        
        for neighbor in range(num_nodes):
            if adj_matrix[neighbor][node] == 1:  # Flow from neighbor to node (upstream direction)
                dfs(neighbor, exceed_dict[node])
    
    dfs(start_node, exceed_dict[start_node])

    # Remove the entry that refers to the same node in exceed_dict.
    for node, L in exceed_dict.items():
        L.pop(node)

    return exceed_dict


def link_nodes(inp_filepath, int_convert=False):
    """
    Create a dictionary of edge: (start_node, end_node) from a swmm network.
    :param inp_filepath: Swmm input filepath.
    :param int_convert: Convert edge and node IDs/names to int.
    :return link_nodes_dict: Dictionary of {edge:(from_node, to_node), ...}
    """
    # Get the names of each conduit and weir.
    conduit_names = file_utils.get_component_names(inp_filepath, 'CONDUITS')
    weir_names = file_utils.get_component_names(inp_filepath, 'WEIRS')

    # Filter names if they start with ';' which is a line comment.
    conduit_names = [c for c in conduit_names if c[0] != ';']
    weir_names = [w for w in weir_names if w[0] != ';']

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


def adjacency_matrix(conduit_nodes_dict, inp_filepath, include_cnames=False):
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
        try:
            from_node_idx = node_names.index(nodes[0])
            to_node_idx = node_names.index(nodes[1])
        except Exception as e:
            print(e)

        # Update adjacency matrix.
        if include_cnames is True:
            A[from_node_idx, to_node_idx] = int(cname)
        else:
            A[from_node_idx, to_node_idx] = 1
            A[to_node_idx, from_node_idx] = -1

    return A, node_names


