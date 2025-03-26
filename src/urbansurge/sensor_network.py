# Stormwater system sensor network.
# Alex Young
# ========================================================

# Library imports.
from collections import OrderedDict
import numpy as np
import numpy.typing as npt
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


def dfs_surcharge_old(A: List[List[int]], node_names: List, start_node: int, invert_elevations: Dict, total_elevations: Dict) -> Dict:
    # Dictionary of exceedance lists for each node.
    exceed_dict = {start_node: OrderedDict()}

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
        
        # Add current node's maximum depth.
        L[node] = total_elevations[node]

        # Add L to exceedance dictionary.
        exceed_dict[node] = L

        # Upstream nodes.
        node_idx = node_names.index(node)
        upstream_indices = np.argwhere(A[:,node_idx] == 1).flatten()
        upstream_nodes = [node_names[i] for i in upstream_indices]

        for upstream_node in upstream_nodes:
            dfs(upstream_node, exceed_dict[node])
    
    dfs(start_node, exceed_dict[start_node])

    # Remove the entry that refers to the same node in exceed_dict.
    for node, L in exceed_dict.items():
        L.pop(node)

    return exceed_dict


def dfs_surcharge_old1(A, node_names, start_node, invert_elevations, total_elevations) -> dict:
    # Dictionary of exceedance lists for each node; using OrderedDict to keep downstream order.
    exceed_dict = {start_node: OrderedDict()}
    visited = set()
    
    def dfs(node: int, L: OrderedDict) -> None:
        if node in visited:
            return
        
        visited.add(node)

        # Get current node’s invert elevation.
        inv_elev = invert_elevations[node]

        # Build a new L by iterating over L in its downstream (insertion) order.
        # Once a node is removed (fails the requirement), all nodes later in the order are dropped.
        new_L = OrderedDict()
        violation_found = False
        for key, surcharge in L.items():
            if violation_found:
                continue
            if surcharge > inv_elev:
                new_L[key] = surcharge
            else:
                violation_found = True
        
        # Add current node’s maximum depth.
        new_L[node] = total_elevations[node]
        
        # Save the current new_L for the current node.
        exceed_dict[node] = new_L

        # Find upstream nodes.
        node_idx = node_names.index(node)
        upstream_indices = np.argwhere(A[:, node_idx] == 1).flatten()
        upstream_nodes = [node_names[i] for i in upstream_indices]

        # Recurse upstream using the updated new_L.
        for upstream_node in upstream_nodes:
            dfs(upstream_node, new_L)
    
    dfs(start_node, exceed_dict[start_node])
    
    # Remove self-references from the exceedance dictionaries.
    for node, L in exceed_dict.items():
        if node in L:
            del L[node]
    
    return exceed_dict


def dfs_surcharge(A, node_names, start_node, invert_elevations, total_elevations) -> dict:
    # Dictionary of exceedance lists for each node; using OrderedDict to maintain insertion order.
    exceed_dict = {start_node: OrderedDict()}
    visited = set()
    
    def dfs(node: int, L: OrderedDict) -> None:
        if node in visited:
            return
        visited.add(node)
        
        # Get current node's invert elevation.
        inv_elev = invert_elevations[node]
        
        # Process L: Remove all entries that were added before any violation.
        # We'll iterate through L in order, and if we encounter a violation (surcharge <= inv_elev),
        # we update our valid_block_start to be the index just after the violation.
        keys = list(L.keys())
        valid_block_start = 0
        for i, key in enumerate(keys):
            if L[key] > inv_elev:
                # Valid entry, do nothing.
                continue
            else:
                # Violation encountered, update valid_block_start to drop this entry and everything before it.
                valid_block_start = i + 1
        
        # Build new_L using only the entries from valid_block_start onward.
        new_L = OrderedDict()
        for key in keys[valid_block_start:]:
            # It is not necessary to re-check validity here because if an item in this block was invalid,
            # the pointer would have been updated further.
            new_L[key] = L[key]
        
        # Add current node's maximum depth.
        new_L[node] = total_elevations[node]
        exceed_dict[node] = new_L
        
        # Identify upstream nodes.
        node_idx = node_names.index(node)
        upstream_indices = np.argwhere(A[:, node_idx] == 1).flatten()
        upstream_nodes = [node_names[i] for i in upstream_indices]
        
        # Recurse upstream with the updated new_L.
        for upstream_node in upstream_nodes:
            dfs(upstream_node, new_L)
    
    dfs(start_node, exceed_dict[start_node])
    
    # Remove self-references from the exceedance dictionaries.
    for node, L in exceed_dict.items():
        if node in L:
            del L[node]
    
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

    if weir_names:
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


def assign_elevation_sensor_nodes(exceed_all, n_min=0):
    sensor_nodes = []
    exceed_counts = compute_exceed_counts(exceed_all)

    while exceed_counts:
        max_node = max(exceed_counts, key=exceed_counts.get)

        # Test whether max_node has more than n_min in its FOV.
        if exceed_counts[max_node] < n_min:
            break

        sensor_nodes.append(max_node)

        # Nodes in sensor field of view.
        fov_nodes = list(exceed_all[max_node].keys())
        fov_nodes.append(max_node)

        # Remove top-level keys that are in fov_nodes
        exceed_all = {k: v for k, v in exceed_all.items() if k not in fov_nodes}

        # Remove keys in fov_nodes from the sub-dictionaries
        for key, sub_dict in exceed_all.items():
            exceed_all[key] = {sub_k: sub_v for sub_k, sub_v in sub_dict.items() if sub_k not in fov_nodes}

        # Recompute exceed_counts.
        exceed_counts = compute_exceed_counts(exceed_all)

    return sensor_nodes


def compute_exceed_counts(exceed_all):
    exceed_counts = {}
    for node, exceed_node_dict in exceed_all.items():
        exceed_counts[node] = len(exceed_node_dict.keys())

    return exceed_counts
