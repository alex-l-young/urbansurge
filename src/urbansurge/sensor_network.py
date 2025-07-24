# Stormwater system sensor network.
# Alex Young
# ========================================================

# Library imports.
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import pandas as pd
import re
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


def dfs_surcharge(A, node_names, start_node, invert_elevations, total_elevations, visited=None) -> dict:
    if visited is None:
        visited = set()

    # Dictionary of exceedance lists for each node; using OrderedDict to maintain insertion order.
    exceed_dict = {start_node: OrderedDict()}
    
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
    
    # # Remove self-references from the exceedance dictionaries.
    # for node, L in exceed_dict.items():
    #     if node in L.keys():
    #         del L[node]
    
    return exceed_dict


def link_nodes(swmm, int_convert=False, link_sections=None):
    """
    Create a dictionary of edge: (start_node, end_node) from a swmm network.
    :param swmm: Instance of urbansurge.swmm_model.SWMM.
    :param int_convert: Convert edge and node IDs/names to int.
    :return link_nodes_dict: Dictionary of {edge:(from_node, to_node), ...}
    """
    if link_sections is None:
        link_sections=['CONDUITS', 'WEIRS', 'PUMPS', 'ORIFICES', 'OUTLETS']

    # Get the names within all link sections.
    link_nodes_dict = {} # Dictionary to store link nodes.
    for link_section in link_sections:
        # link_names = file_utils.get_component_names(inp_filepath, link_section)
        link_names = swmm.get_component_names(link_section)

        # If section doesn't exist or has no names, skip.
        if link_names is None:
            continue

        # Filter names if they start with ';' which is a line comment.
        link_names = [c for c in link_names if c[0] != ';']

        for name in link_names:
            # from_node = file_utils.get_inp_section(inp_filepath, link_section, 'From Node', name)
            # to_node = file_utils.get_inp_section(inp_filepath, link_section, 'To Node', name)
            from_node, to_node = swmm.get_link_nodes(name)

            # Populate dictionary as {conduit_name: (from_node, to_node), ...}
            if int_convert is True:
                name = int(name)
                from_node = int(from_node)
                to_node = int(to_node)
            link_nodes_dict[name] = (from_node, to_node)

    return link_nodes_dict


def adjacency_matrix(conduit_nodes_dict, swmm, include_cnames=False):
    """
    Create adjacency matrix of network.
    :param conduit_nodes_dict: Conduit node dictionary. Output from conduit_nodes().
    :param swmm: Instance of urbansurge.swmm_model.SWMM.
    :param include_cnames: Instead of 1 for adjacent nodes, add the conduit ID as an integer.
    :return: Tuple (adjacency matrix, node names)
    """

    # All node names.
    # node_names = file_utils.get_component_names(inp_filepath, 'JUNCTIONS')
    node_names = swmm.get_component_names('JUNCTIONS')

    # Outfall names.
    # outfall_names = file_utils.get_component_names(inp_filepath, 'OUTFALLS')
    outfall_names = swmm.get_component_names('OUTFALLS')
    node_names.extend(outfall_names)

    # Add storage nodes.
    # storage_ids = file_utils.get_component_names(inp_filepath, 'STORAGE')
    storage_ids = swmm.get_component_names('STORAGE')
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


def assign_elevation_sensor_nodes(exceed_all: dict, n_min=0) -> List[str]:
    """
    Assign sensor nodes based on a greedy algorithm.
    1. First sensor is node with largest fov.
    2. That node and all nodes in fov are removed from contention.
    3. Repeat.

    :param exceed_all: Dictionary with node names as keys and fov dictionarys as values.
    :param n_min: Minimum number of nodes allowed in an fov.

    :return: List of sensor nodes.
    """
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
    """
    Counts the number of nodes in each node's FoV.

    :param exceed_all: Dictionary containing all nodes and the downstream FoVs.
    :return: Dictionary with keys as node names and values as FoV counts. 
    """
    exceed_counts = {}
    for node, exceed_node_dict in exceed_all.items():
        exceed_counts[node] = len(exceed_node_dict.keys())

    return exceed_counts


def exceed_counts_df(exceed_all):
    """
    Returns a DataFrame with all nodes and corresponding FoV counts from exceed_all.

    :param exceed_all: Dictionary containing all nodes and the downstream FoVs.
    :return: DataFrame containing list of all nodes and the FoV counts for each node. 
    """
    exceed_counts = {'node': [], 'fov_count': []}
    for node, exceed_node_dict in exceed_all.items():
        exceed_counts['node'].append(node)
        exceed_counts['fov_count'].append(len(exceed_node_dict.keys()))

    return pd.DataFrame(exceed_counts)


def get_starting_nodes(swmm):
    """
    Starting nodes are: 
    - Outfalls.
    - Nodes that are immediately upstream of a weir, orifice, pump, or outlet and downstream of a regular conduit.
    - Nodes that are immediately upstream of a storage unit.

    :param swmm: A swmm_model.SWMM object.

    :return: List of start nodes.
    """
    # Outfall nodes.
    outfall_nodes = swmm.get_component_names('OUTFALLS')

    # Function to get list of from nodes from a link section.
    def from_nodes(links, link_section): 
        from_nodes = []
        for link in links:
            from_node, _ = swmm.get_link_nodes(link, link_section=link_section)
            from_nodes.append(from_node)
        return from_nodes

    def upstream_of_storage(storage_id, link_nodes_dict, conduit_ids, junction_ids):
        # Link(s) upstream of storage.
        upstream_links = swmm.get_upstream_links(storage_id, link_nodes_dict)

        upstream_start_nodes = []
        if upstream_links:
            for link in upstream_links:
                if link in conduit_ids:
                    upstream_node = from_nodes([link], 'CONDUITS')
                    # Only add junction nodes.
                    if upstream_node[0] in junction_ids:
                        upstream_start_nodes.extend(upstream_node)
        
        return upstream_start_nodes
    
    def upstream_of_outfall(outfall_id, link_nodes_dict, conduit_ids, junction_ids):
        # Link(s) upstream of outfall.
        upstream_links = swmm.get_upstream_links(outfall_id, link_nodes_dict)        

        upstream_start_nodes = []
        if upstream_links:
            for link in upstream_links:
                if link in conduit_ids:
                    upstream_node = from_nodes([link], 'CONDUITS')
                    # Only use as a starting node if the upstream node is a junction.
                    if upstream_node[0] in junction_ids:
                        upstream_start_nodes.extend(upstream_node)
        
        return upstream_start_nodes

    # From nodes.
    weir_from_nodes = from_nodes(swmm.get_component_names('WEIRS'), 'WEIRS')
    orifice_from_nodes = from_nodes(swmm.get_component_names('ORIFICES'), 'ORIFICES')
    pump_from_nodes = from_nodes(swmm.get_component_names('PUMPS'), 'PUMPS')
    outlet_from_nodes = from_nodes(swmm.get_component_names('OUTLETS'), 'OUTLETS')

    # Nodes upstream of storage units.
    storage_ids = swmm.get_component_names('STORAGE')
    conduit_ids = swmm.get_component_names('CONDUITS')
    junction_ids = swmm.get_component_names('JUNCTIONS')
    link_nodes_dict = swmm.get_link_node_map(link_sections=['CONDUITS'])
    storage_upstream_nodes = []
    for storage_id in storage_ids:
        storage_upstream_nodes.extend(upstream_of_storage(storage_id, link_nodes_dict, conduit_ids, junction_ids))

    # Nodes upstream of outfalls.
    
    outfall_upstream_nodes = []
    for outfall_node in outfall_nodes:
        outfall_upstream_nodes.extend(upstream_of_outfall(outfall_node, link_nodes_dict, conduit_ids, junction_ids))

    # Start nodes.
    start_nodes = weir_from_nodes + orifice_from_nodes + pump_from_nodes + outlet_from_nodes + storage_upstream_nodes + outfall_upstream_nodes

    # Remove node names that don't include a number or letter.
    start_nodes = [s for s in start_nodes if re.search(r'[A-Za-z0-9]', s)] 

    # Remove any storage nodes.
    start_nodes = [s for s in start_nodes if s not in storage_ids]

    return start_nodes

  
def generate_sensor_network(swmm, A, node_names, invert_elevations, S=None, surcharge_threshold=1.0, min_fov=1):
    """
    Generates a sensor network that detects downstream backup using existing sensor network.

    :param swmm: A swmm_model.SWMM object.
    :param A: Adjacency matrix for nodes.
    :param node_names: Array of node names that correspond to the rows and columns of the adjacency matrix.
    :param invert_elevations: Invert elevations of all nodes in node_names.
    :param S: Set of nodes IDs to exclude from the sensor network.
    :param surcharge_threshold: Threshold above which a manhole is considered surcharged.
        This is a fraction of manhole depth, i.e., 1.0 = surcharge is full depth and surface is breached.
    :param min_fov: Minimum number of manholes in a field of view inclusive of sensor manhole. 1 is minimum.
    
    :return sensor_nodes: List of nodes assigned sensors.
    :return sensor_locs: Dictionary of sensor locations {sensor_node: (x, y)}
    :return exceed_all: FoV dictionary {sensor_node: OrderedDict{fov_node_1: invert_elevation, ...}}
    :return exceed_counts: Number of manholes in each sensor node's FoV.
    """
    if S is None:
        S = set()

    print('>>> Getting node elevations...')
    # Get node names.
    outfall_nodes = swmm.get_component_names('OUTFALLS')
    storage_nodes = swmm.get_component_names('STORAGE')

    # Maximum elevations is the maximum of the node depth and the maximum downstream link diameter.
    max_elevations = {node: swmm.get_node_max_depth(node) * surcharge_threshold for node in node_names if node not in outfall_nodes + storage_nodes}

    # Total elevations.
    total_elevations = {}
    for node in node_names:
        if node in outfall_nodes + storage_nodes:
            # total_elevations[node] = invert_elevations[node]
            continue
        else:
            total_elevations[node] = max_elevations[node] + invert_elevations[node]

    print('>>> Running sensor placement algorithm...')
    # Starting nodes are outfalls.
    start_nodes = get_starting_nodes(swmm)

    exceed_dict_list = []
    for start_node in start_nodes:
        exceed_dict = dfs_surcharge(A, node_names, start_node, invert_elevations, total_elevations, visited=S)
        exceed_dict_list.append(exceed_dict)

    exceed_all = {}
    for exceed_dict in exceed_dict_list:
        for node, exceed_node_dict in exceed_dict.items():
            if node in exceed_all.keys():
                exceed_all[node].update(exceed_node_dict)
            else:
                exceed_all[node] = exceed_node_dict

    exceed_counts = compute_exceed_counts(exceed_all)

    # # Remove outfalls and storage units from exceed_all so that they won't end up as sensors.
    # for node in outfall_nodes + storage_nodes:
    #     exceed_all.pop(node)

    # Assign sensor locations.
    sensor_nodes = assign_elevation_sensor_nodes(exceed_all, n_min=min_fov-1)

    # Sensor locations.
    sensor_locs = swmm.get_node_coordinates(sensor_nodes)

    print('>>> DONE.')

    return sensor_nodes, sensor_locs, exceed_all, exceed_counts


def generate_threshold_sensor_network(swmm, surcharge_thresholds):
    """
    Generate optimal sensor networks for multiple surcharge threshold by starting at full manhole surcharge and
    progressively lowering the surcharge threshold.

    :param swmm: Instance of urbansurge.swmm_model.SWMM.
    :param surcharge_thresholds: Array of surcharge thresholds starting at 1.0 (full surcharge) and ending at a minimum value.

    :return: Data frame of sensor networks for each depth threshold. The d_thresh columns of the data frame correspond to the 
        values in the surcharge_thresholds array.
    """
    # Link-nodes dictionary.
    link_nodes_dict = link_nodes(swmm, link_sections=['CONDUITS'])

    # Adjacency matrix.
    A, node_names = adjacency_matrix(link_nodes_dict, swmm, include_cnames=False)

    # Invert elevations
    invert_elevations = {node: swmm.get_node_invert_elevation(node) for node in node_names}


    optimal_sensor_networks = []
    # S = set()
    S_network = []
    fov_dicts = {}
    for i, d_thresh in enumerate(surcharge_thresholds):
        # Include storage units in initial visited set S so that they aren't included in the sensor network.
        S = set(swmm.get_component_names('STORAGE'))
        
        print('Threshold fraction', d_thresh)
        sensor_nodes, sensor_locs, exceed_all, exceed_counts = generate_sensor_network(swmm, A, node_names, invert_elevations, S=S, surcharge_threshold=d_thresh)

        # Get combined FoV from all sensors.
        fov_list = []
        for sensor_node, fov in exceed_all.items():
            fov_list.append(sensor_node)
            for fov_node, _ in fov.items():
                fov_list.append(fov_node)
        fov_list = list(set(fov_list))
        fov_dicts[i] = exceed_all
        
        # Update sensor network nodes.
        S_network = list(set(S_network + sensor_nodes))
        S = set(deepcopy(S_network + fov_list + list(S)))

        # Save optimal sensor network.
        optimal_sensor_networks.append(S_network)

    d_thresh_cols = [f'd_thresh_{i}' for i in range(len(surcharge_thresholds))]

    data_dict = {}
    data_dict['nodes'] = node_names
    for i, d_thresh_col in enumerate(d_thresh_cols):
        data_dict[d_thresh_col] = np.array([1 if node in optimal_sensor_networks[i] else 0 for node in node_names])

    sensor_network_df = pd.DataFrame(data_dict)

    return sensor_network_df, fov_dicts


# def get_fovs(swmm, sensor_nodes, surcharge_threshold=1):
#     # Link-nodes dictionary.
#     link_nodes_dict = link_nodes(swmm, link_sections=['CONDUITS'])

#     # Adjacency matrix.
#     A, node_names = adjacency_matrix(link_nodes_dict, swmm, include_cnames=False)

#     # Invert elevations
#     invert_elevations = {node: swmm.get_node_invert_elevation(node) for node in node_names}

#     # Get start nodes.
#     start_nodes = get_starting_nodes(swmm)

#     # Get node names.
#     outfall_nodes = swmm.get_component_names('OUTFALLS')
#     storage_nodes = swmm.get_component_names('STORAGE')

#     # Maximum elevations is the maximum of the node depth and the maximum downstream link diameter.
#     max_elevations = {node: swmm.get_node_max_depth(node) * surcharge_threshold for node in node_names if node not in outfall_nodes + storage_nodes}

#     # Total elevations.
#     total_elevations = {}
#     for node in node_names:
#         if node in outfall_nodes + storage_nodes:
#             # total_elevations[node] = invert_elevations[node]
#             continue
#         else:
#             total_elevations[node] = max_elevations[node] + invert_elevations[node]

#     # Include storage units in initial visited set S so that they aren't included in the sensor network.
#     S = set(swmm.get_component_names('STORAGE'))

#     exceed_dict_list = []
#     for start_node in start_nodes:
#         exceed_dict = dfs_surcharge(A, node_names, start_node, invert_elevations, total_elevations, S=S)
#         exceed_dict_list.append(exceed_dict)

#     exceed_all = {}
#     for exceed_dict in exceed_dict_list:
#         for node, exceed_node_dict in exceed_dict.items():
#             if node in exceed_all.keys():
#                 exceed_all[node].update(exceed_node_dict)
#             else:
#                 exceed_all[node] = exceed_node_dict

#     # Select sensor nodes.
#     sensor_fovs = {node: fov for node, fov in exceed_all.items() if node in sensor_nodes}

#     return sensor_fovs