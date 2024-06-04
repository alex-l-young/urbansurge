###########################################################################
# Utility functions for sensor operations.
###########################################################################

# Library imports.
import networkx as nx
import numpy as np
import pandas as pd

# Local imports.
from urbansurge import file_utils, swmm_model


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


def upstream_assign(link_node_dict, inp_filepath, Nups=2, link_sensors=False):
    """
    Assign sensor locations by the number of upstream components.
    :param link_node_dict: Dictionary output from link_nodes().
    :param inp_filepath: SWMM input filepath.
    :param Nups: Number of upstream components to prune by.
    :param link_sensors: Use links as sensor locations.
            The function use link downstream from each sensor node as a sensor location. If there is no link downstream
            of the node, the upstream link will be used. Duplicate links are removed.
    :return sensor_nodes: Node ids where sensors should go.
    """
    # Create adjacency matrix and get node names.
    A_matrix, node_names = adjacency_matrix(link_node_dict, inp_filepath)

    # G is the directed graph of the network.
    G_node_names = node_names

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

    # Use links as sensors if flag is set to True.
    if link_sensors is True:
        # Collect upstream and downstream nodes, links.
        upstream_nodes = []
        downstream_nodes = []
        links = []
        for link, nodes in link_node_dict.items():
            # Upstream and dowstream nodes for a link.
            upstream_nodes.append(nodes[0])
            downstream_nodes.append(nodes[1])
            links.append(link)

        # Check if sensor node is in upstream nodes.
        sensor_locations = []
        for sensor_node in sensor_nodes:
            if sensor_node in upstream_nodes:
                # Index of sensor node in upstream nodes.
                sensor_idx = upstream_nodes.index(sensor_node)
                sensor_locations.append(links[sensor_idx])
                continue

            # If sensor node was not found in upstream nodes, add the upstream link.
            sensor_idx = downstream_nodes.index(sensor_node)
            sensor_locations.append(links[sensor_idx])

        # Remove duplicates.
        sensor_locations = list(set(sensor_locations))

    else:
        sensor_locations = sensor_nodes


    return sensor_locations


def upstream_assign_links(link_node_dict, inp_filepath, Nups=2, exclude_weirs=False):
    """
    Assign sensor locations to conduits by the number of upstream conduits, excluding weirs.
    :param link_node_dict: Dictionary output from link_nodes().
    :param inp_filepath: SWMM input filepath.
    :param Nups: Number of upstream components to prune by.
    :return sensor_nodes: Node ids where sensors should go.
    """
    # Create adjacency matrix and get node names.
    A_matrix, node_names = adjacency_matrix(link_node_dict, inp_filepath)

    # G is the directed graph of the network.
    G_node_names = node_names

    # Flip link-node dict so edge names are accessible. Also reverse node order.
    node_link_dict = {(v[1], v[0]): k for k, v in link_node_dict.items()}

    # Directed graph of the network.
    G = nx.from_numpy_array(A_matrix, create_using=nx.DiGraph)
    nx.relabel_nodes(G, dict(zip(G.nodes, G_node_names)), copy=False)

    # Reversed digraph.
    H = G.reverse()

    if exclude_weirs is True:
        # Get weir names.
        weir_names = file_utils.get_component_names(inp_filepath, 'WEIRS')

    # Assign sensors based on number of upstream links.
    sensor_links = []
    max_upstream = Nups + 1  # Maximum number of components upstream of any node.
    H_node_names = np.array(G_node_names.copy())
    H_link_names = np.array([node_link_dict[nodes] for nodes in H.edges])

    if exclude_weirs is True:
        # Remove weir ids from link names.
        H_link_names = np.array([edge for edge in H_link_names if edge not in weir_names])

    while max_upstream >= Nups:
        # Number of upstream edges for each node.
        upstream_edges = np.zeros(len(H_link_names))
        for i, link_name in enumerate(H_link_names):
            # Get the upstream node for the link.
            up_node = link_node_dict[link_name][0]

            # List of links upstream of that node.
            node_edges = list(nx.bfs_tree(H, source=up_node).edges)
            link_names = [node_link_dict[nodes] for nodes in node_edges]

            if exclude_weirs is True:
                # Remove weir ids.
                n_upstream = [edge for edge in link_names if edge not in weir_names]

            upstream_edges[i] = len(link_names)

        # If there are no upstream edges, add outfall link(s) and break loop.
        if len(upstream_edges) == 0:
            sensor_links.extend(H_link_names)
            break

        # Maximum upstream nodes.
        max_upstream = np.max(upstream_edges)

        # If the maximum number of upstream nodes is less than Nupstream, add
        # the outfall as the final node and break the loop.
        if max_upstream < Nups:
            most_upstream_idx = np.argmax(upstream_edges)
            sensor_links.append(H_link_names[most_upstream_idx])
            break

        # Choose a node with the number of upstream edges closest to Nupstream.
        # Choose randomly if there is more than 1. Increment the search number if
        # there are no nodes found with N = Nupstream.
        sensor_link = None
        search_num = Nups
        while sensor_link is None:
            # Array of potential sensors where the number of upstream edges is
            # equal to search_num.
            # potential_sensors = upstream_edges[upstream_edges[:, 1] == search_num, 0]
            potential_sensors = H_link_names[upstream_edges == search_num]

            if len(potential_sensors) == 0:
                # If there are no potential sensors, increment search_num.
                search_num += 1
            elif len(potential_sensors) > 1:
                # Randomly choose sensor if there is more than 1 option.
                sensor_link = np.random.choice(potential_sensors)
            else:
                sensor_link = potential_sensors[0]

        # Add sensor to list of sensor nodes.
        sensor_links.append(sensor_link)

        # Get the upstream node of the sensor link and use it to prune the network.
        prune_node = link_node_dict[sensor_link][0]

        # Nodes and links to remove in pruning operation.
        nodes_to_remove = list(nx.bfs_tree(H, source=prune_node).nodes())
        links_to_remove = [node_link_dict[edge] for edge in list(nx.bfs_tree(H, source=prune_node).edges())]

        # Prune nodes.
        for node_to_remove in nodes_to_remove:
            # Remove all nodes upstream of the prune node including the prune node.
            # if upstream_node != down_node:
            H.remove_node(node_to_remove)

            # Remove node name from node name list.
            H_node_names = H_node_names[H_node_names != node_to_remove]

        # Prune links. I.e., update link names based on pruned network.
        H_link_names = np.array([node_link_dict[edge] for edge in list(H.edges())])

        if exclude_weirs is True:
            # Remove weir ids from link names.
            H_link_names = np.array([edge for edge in H_link_names if edge not in weir_names])

    # Remove duplicates.
    sensor_locations = list(set(sensor_links))

    return sensor_locations


class SensorNetwork():
    def __init__(self):
        self.depth_sensors = {}
        self.velocity_sensors = {}
        self.flow_sensors = {}
        self.rain_gauges = {}

    def add_depth_sensor(self, sensor):
        self.depth_sensors[sensor.sensor_id] = sensor

    def remove_depth_sensor(self, sensor_id):
        self.depth_sensors.pop(sensor_id)

    def add_velocity_sensor(self, sensor):
        self.velocity_sensors[sensor.sensor_id] = sensor

    def remove_velocity_sensor(self, sensor_id):
        self.velocity_sensors.pop(sensor_id)

    def add_flow_sensor(self, sensor):
        self.flow_sensors[sensor.sensor_id] = sensor

    def remove_flow_sensor(self, sensor_id):
        self.flow_sensors.pop(sensor_id)

    def add_rain_gauge(self, sensor):
        self.rain_gauges[sensor.sensor_id] = sensor

    def remove_rain_gauge(self, sensor_id):
        self.rain_gauges.pop(sensor_id)


class Sensor():
    def __init__(self, sensor_id, units, fs=None, dt=None, time=None, measure_data=None, model_data=None, **kwargs):
        # Sensor ID and units are required.
        self.sensor_id = sensor_id
        self.units = units

        # Sampling frequency and time step.
        if fs:
            self.fs = fs
            self.dt = 1 / fs
        elif dt:
            self.dt = dt
            self.fs = 1 / dt
        else:
            raise ValueError('Sampling frequency (fs) or time step duration (dt) must be specified.')

        # Set data if it is supplied.
        if time is not None:
            self.time = time
        if measure_data is not None:
            self.measure_data = measure_data
        if model_data is not None:
            self.model_data = model_data

        super().__init__(**kwargs)


    def compute_residual(self):
        """
        Compute sensor residual between modeled and measured data.
        Residual = model_data - measure_data
        :return: Residual.
        """
        self.residual = self.model_data - self.measure_data

        return self.residual


class DepthSensor(Sensor):
    def __init__(self, cfg_filepath, component_id, component_type, **kwargs):


        self.cfg_filepath = cfg_filepath
        self.component_id = component_id
        self.component_type = component_type

        super().__init__(**kwargs)


    def depth_range(self, link_shape):
        """
        Compute the depth range observed over the link-mounted sensor's data record as a fraction of the
        maximum depth.
        :return: depth_range (min, max)
        """
        # Swmm model instance.
        swmm = swmm_mode.SWMM(self.cfg_filepath)

        # Link geometry.
        link_geometry = swmm.get_link_geometry(self.component_id)

        # Link max depth.
        link_max_depth = link_geometry[0]

        # Minimum and maximum depth from data record as fractions of max depth.
        h_min = np.min(self.data) / link_max_depth
        h_max = np.max(self.data) / link_max_depth

        return (h_min, h_max)


class VelocitySensor(Sensor):
    def __init__(self, cfg_filepath, component_id, component_type, **kwargs):

        self.cfg_filepath = cfg_filepath
        self.component_id = component_id
        self.component_type = component_type

        super().__init__(**kwargs)


class FlowSensor(Sensor):
    def __init__(self, cfg_filepath, component_id, component_type, **kwargs):

        self.cfg_filepath = cfg_filepath
        self.component_id = component_id
        self.component_type = component_type

        super().__init__(**kwargs)

    def compute_cumu_storm_flow(self, P_start_idxs, P_end_idxs):
        # Loop through peaks and compute cumulative storm flow.
        cumu_storm_flow = np.zeros(len(P_start_idxs))
        for i in range(len(P_start_idxs)):
            cumu_flow = np.sum(self.measure_data[P_start_idxs[i]:P_end_idxs[i]])
            cumu_storm_flow[i] = cumu_flow

        self.cumu_storm_flow = cumu_storm_flow
        return self.cumu_storm_flow


class RainGauge(Sensor):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def find_storms(self, thresh):
        """
        Find start and end indices of storms in a time series of precipitation.
        Storms are defined as periods where precipitation exceeds a certain threshold.
        :param thresh: Precipitation threshold to define a storm.
        :return: Start and end indices for storms as numpy arrays.
        """
        # Precipitation is the measured data.
        P = self.measure_data

        # Integer boolean array where P > thresh.
        exceed = P > thresh
        exceed = exceed.astype(int)

        # Difference between boolean exceedances.
        exceed_diff = np.diff(exceed)

        # Storms start where exceed is equal to 1 (i.e., crosses threshold).
        self.storm_start_idx = np.where(exceed_diff == 1)[0]

        # Storm end index is the storm start index plus the last index.
        self.storm_end_idx = np.append(self.storm_start_idx[1:], len(P) - 1)

        # Number of storms.
        self.n_storms = len(self.storm_start_idx)

        return self.storm_start_idx, self.storm_end_idx

    def compute_cumu_storm_prcp(self):
        P = self.measure_data
        # Loop through peaks and compute cumulative storm precipitation.
        cumu_storm_prcp = np.zeros(len(self.storm_start_idx))
        for i in range(len(self.storm_start_idx)):
            cumu_prcp = np.sum(P[self.storm_start_idx[i]:self.storm_end_idx[i]])
            cumu_storm_prcp[i] = cumu_prcp

        self.cumu_storm_prcp = cumu_storm_prcp
        return self.cumu_storm_prcp

    def compute_max_storm_prcp(self):
        P = self.measure_data
        # Loop through peaks and compute maximum storm precipitation.
        max_storm_prcp = np.zeros(len(self.storm_start_idx))
        for i in range(len(self.storm_start_idx)):
            max_prcp = np.max(P[self.storm_start_idx[i]:self.storm_end_idx[i]])
            max_storm_prcp[i] = max_prcp

        self.max_storm_prcp = max_storm_prcp
        return self.max_storm_prcp