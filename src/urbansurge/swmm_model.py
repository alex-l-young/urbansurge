# ================================================================================
# SWMM Model Class.
# ================================================================================

# Package imports.
from urbansurge import file_utils

# Library imports.
from pathlib import Path
from pyswmm import Simulation, Nodes, Links, Output
from swmm.toolkit.shared_enum import LinkAttribute, NodeAttribute, SubcatchAttribute, SystemAttribute
import yaml
import shutil
import os
import numpy as np
import pandas as pd
from typing import Union, List


class SWMM:
    def __init__(self, config_path):
        """
        SWMM class for manipulation of an EPA SWMM model.

        :param config_path: Path to configuration file.
        """
        # Parse the configuration file into a dictionary.
        self.cfg = self._parse_config(config_path)

        # Run from temporary file if needed.
        if self.cfg['temp_inp'] is True:
            self._create_temp_inp()

        # Extract required configurations.
        self.inp_path = self.cfg['inp_path']
        self.out_path = os.path.splitext(self.inp_path)[0] + '.out'
        self.verbose = self.cfg['verbose']

        with Simulation(self.inp_path) as sim:
            # Simulation information
            print("Simulation info")
            flow_units = sim.flow_units
            print("Flow Units: {}".format(flow_units))
            system_units = sim.system_units
            print("System Units: {}".format(system_units))
            print("Start Time: {}".format(sim.start_time))
            print("Start Time: {}".format(sim.end_time))

        # Parse input file into a database.
        self.inp_db = file_utils.inp_to_database(self.inp_path)


    def _parse_config(self, config_path: str) -> dict:
        """
        Parses the configuration file.

        :param config_path: Path to configuration file.
        :return: Configuration dictionary.

        """
        with open(config_path, "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        return cfg


    def configure_model(self):
        """
        Configure the EPA SWMM model from the configuration file. 

        :return: No return.
        
        """

        # Conduits.
        # =====================================================
        # Diameter.
        conduit_diams = self.cfg['CONDUITS']['geometry']
        if conduit_diams:
            for link_id, geom in conduit_diams.items():
                self.set_link_geometry(link_id, geom)

        # Roughness.
        conduit_roughnesses = self.cfg['CONDUITS']['roughness']
        if conduit_roughnesses:
            for link_id, roughness in conduit_roughnesses.items():
                self.set_link_roughness(link_id, roughness)

        # Length.
        conduit_lengths = self.cfg['CONDUITS']['length']
        if conduit_lengths:
            for link_id, length in conduit_lengths.items():
                self.set_link_length(link_id, length)

        # Junctions.
        # =====================================================
        # Max depth.
        junction_max_depths = self.cfg['JUNCTIONS']['max_depth']

        # Invert elevation.
        junction_invert_elevations = self.cfg['JUNCTIONS']['invert_elevation']

        # Subcatchments.
        # =====================================================
        # Percent impervious.
        subcatchment_perc_impervious = self.cfg['SUBCATCHMENTS']['perc_impervious']

        # Precipitation.
        # =====================================================
        # Precipitation timeseries.
        timeseries_ids = self.cfg['TIMESERIES']
        if timeseries_ids:
            overwrite = self.cfg['TIMESERIES_OPTIONS']['overwrite']
            for timeseries_id, timeseries in timeseries_ids.items():
                ts_name = timeseries['name']
                ts_description = timeseries['description']
                times = timeseries['hours']
                values = timeseries['values']
                dates = timeseries['dates']
                self.add_prcp_timeseries(ts_name, ts_description, times, values, dates=dates, overwrite=overwrite)

                # Prevent double overwrite for subsequent time series in the config file.
                if overwrite is True:
                    overwrite = False

        # Rain gauges.
        # =====================================================
        rain_gages = self.cfg['RAINGAGE']
        if rain_gages:
            for raingage_id, ts_dict in rain_gages.items():
                self.set_raingage_timeseries(raingage_id, ts_dict['timeseries'])


    def _create_temp_inp(self):
        """
        Create a temporary inp file to run the model from.

        :return: Sets the name of self.cfg['inp_path'] to temporary path.
        """
        # Create new temporary file path.
        split_inp_path = os.path.splitext(self.cfg['inp_path'])
        inp_temp_path = split_inp_path[0] + '_tmp' + split_inp_path[1]

        # Copy inp file to temporary new file.
        shutil.copy(self.cfg['inp_path'], inp_temp_path)

        # Update name in self.cfg.
        self.cfg['inp_path'] = inp_temp_path


    def run_simulation(self):
        # Instantiate model.
        print(f'INP PATH: {self.inp_path}')
        with Simulation(self.inp_path) as sim:
            for ind, step in enumerate(sim):
                if ind % 100 == 0:
                    print(sim.current_time, ",", round(sim.percent_complete * 100))
                    

    def get_component_names(self, section):
        """
        Returns the names of all components for a given section.

        :param section: The name of the section.
        :type section: str
        :return: A list of component names.
        :rtype: list

        """
        name_col_idx = 0
        component_names = list(self.inp_db[section].iloc[:, name_col_idx])

        return component_names


    def get_components_by_tag(self, tag):
        """
        Returns the names of all components for a given tag.

        :param tag: tag name.
        :return: List of component names.
        """
        component_names = file_utils.get_components_by_tag(self.inp_path, tag)

        return component_names


    def get_node_section(self, node_id):
        """
        Find which section the node ID falls under.

        :param node_id:
        :return: Section name.
        """
        node_sections = ['JUNCTIONS', 'OUTFALLS', 'STORAGE']

        for node_section in node_sections:
            component_names = self.get_component_names(node_section)
            if component_names is None:
                continue

            if node_id in component_names:
                return node_section

        return 'NODE ID NOT FOUND IN ANY NODE SECTION'
    

    def get_node_coordinates(self, node_ids) -> np.array:
        """
        Get the X, Y coordinates of a node.

        :param node_id: ID of node.
        :return: (X, Y) as a numpy array.
        """
        section = 'COORDINATES'
        x_column_name = 'X-Coord'
        y_column_name = 'Y-Coord'
        X = np.zeros(len(node_ids))
        Y = np.zeros(len(node_ids))
        with open(self.inp_path, 'r') as file:
            # Read the file into a list of lines
            lines = file.readlines()
            for i, component_name in enumerate(node_ids):
                x = file_utils.get_inp_section_from_lines(lines, section, x_column_name, component_name)
                y = file_utils.get_inp_section_from_lines(lines, section, y_column_name, component_name)
                X[i] = float(x)
                Y[i] = float(y)

        return np.column_stack((X, Y))
    

    def get_node_invert_elevation(self, node_id) -> float:
        """
        Get the invert elevation of a node.

        :param node_id: ID of the node.
        :return: Node invert elevation.
        """
        section = self.get_node_section(str(node_id))
        if section == 'STORAGE':
            column_name = 'Elev.'
        else:
            column_name = 'Elevation'
        component_name = str(node_id)
        # invert_elevation = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)

        node_df = self.inp_db[section]
        invert_elevation = node_df.loc[node_df['Name']==component_name, column_name].iloc[0]

        return float(invert_elevation)
    

    def get_node_surcharge_depth(self, node_id) -> float:
        """
        Get the surcharge depth of a node.

        :param node_id: ID of the node.
        :return: Node node surcharge depth.
        """
        section = self.get_node_section(str(node_id))
        column_name = 'SurDepth'
        component_name = str(node_id)
        try:
            surcharge_depth = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)
        except Exception as e:
            print(e)
            print(section)

        return float(surcharge_depth)
    

    def get_node_max_depth(self, node_id) -> float:
        """
        Get the maximum depth of a node.

        :param node_id: ID of the node.
        :return: Node maximum depth.
        """
        section = self.get_node_section(str(node_id))
        column_name = 'MaxDepth'
        component_name = str(node_id)
        try:
            max_depth = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)
        except Exception as e:
            print(e)
            print(section)

        return float(max_depth)


    def get_downstream_components(self, start_component_id, start_component_type, downstream_component_type):

        # Make component ID a string.
        start_component_id = str(start_component_id)

        # Handle section names for different component types.
        section_dict = {'Link': 'CONDUITS', 'Junction': 'JUNCTIONS', 'Outfall': 'OUTFALLS', 'Storage': 'STORAGE', 'Weir': 'WEIRS'}
        non_link_component_types = list(section_dict.keys())
        non_link_component_types.remove('Link')
        non_link_component_types.remove('Weir')
        downstream_section = section_dict[downstream_component_type]
        downstream_section_ids = self.get_component_names(downstream_section)

        # Column names for from and to nodes.
        from_column_name = 'From Node'
        to_column_name = 'To Node'

        # Conduit names.
        conduit_ids = self.get_component_names('CONDUITS')
        junction_ids = self.get_component_names('JUNCTIONS')
        storage_ids = self.get_component_names('STORAGE')
        outfall_ids = self.get_component_names('OUTFALLS')
        weir_ids = self.get_component_names('WEIRS')

        # Weirs are treated as conduits.
        if weir_ids:
            conduit_ids = conduit_ids + weir_ids

        # {Node: component type} dictionary.
        node_type_dict = {}
        if junction_ids:
            for id in junction_ids:
                node_type_dict[id] = 'Junction'
        if storage_ids:
            for id in storage_ids:
                node_type_dict[id] = 'Storage'
        if outfall_ids:
            for id in outfall_ids:
                node_type_dict[id] = 'Outfall'

        # Create dataframe with columns | conduit_id | from_node_id | to_node_id | from CONDUITS section.
        nodes = {'conduit_id': [], 'from_node_id': [], 'to_node_id': []}
        for i, id in enumerate(conduit_ids):
            if weir_ids and id in weir_ids:
                from_node_id = file_utils.get_inp_section(self.inp_path, 'WEIRS', from_column_name, id)
                to_node_id = file_utils.get_inp_section(self.inp_path, 'WEIRS', to_column_name, id)
            else:
                from_node_id = file_utils.get_inp_section(self.inp_path, 'CONDUITS', from_column_name, id)
                to_node_id = file_utils.get_inp_section(self.inp_path, 'CONDUITS', to_column_name, id)
            nodes['conduit_id'].append(id)
            nodes['from_node_id'].append(from_node_id)
            nodes['to_node_id'].append(to_node_id)

        # Make nodes into a data frame.
        nodes = pd.DataFrame(nodes)

        # Traverse downstream components to outfall while collecting components of required type.
        downstream_components = []
        component_id = start_component_id
        component_type = start_component_type

        # If starting on a link and output is links, add that link.
        if downstream_component_type == 'Link' and start_component_type == 'Link':
            downstream_components.append(component_id)

        break_counter = 1
        while component_id not in outfall_ids or component_type != 'Outfall':
            
            # Immediate downstream component.
            if component_type == 'Link':
                # Get the outlet node (to node) of the link.
                next_component_id = nodes.loc[nodes['conduit_id'] == component_id, 'to_node_id'].iloc[0]
                next_component_type = node_type_dict[next_component_id]
                print(next_component_id, next_component_type)
            elif component_type in non_link_component_types:
                # Get the conduit corresponding to the node.
                next_component_id = nodes.loc[nodes['from_node_id'] == component_id, 'conduit_id'].iloc[0]
                next_component_type = 'Link'
            else:
                raise Exception(f'Component type of "{component_type}" not one of {list(section_dict.keys())}')

            # Add component to downstream components if the component should be collected.
            if next_component_type == downstream_component_type and str(next_component_id) in downstream_section_ids:
                downstream_components.append(next_component_id)

            # Update component ID.
            component_id = next_component_id
            component_type = next_component_type

            # Break loop if it repeats more than 10,000 times.
            if break_counter >= 1e4:
                raise Exception('Downstream search budget of {} exceeded'.format(break_counter))
            break_counter += 1

        # If the desired downstream component is an outfall, add it once the previous loop has finished.
        if downstream_component_type == 'Outfall':
            downstream_components = [component_id]

        return downstream_components

    def upstream_distance(self, component_1, component_1_type, component_2, component_2_type):
        """
        Calculates the distance upstream from component 1 to component 2.

        :param component_1: ID of component 1.
        :param component_2: ID of component 2.
        :return: Distance upstream from component 1 to component 2. Returns 0 if component_1 == component_2 and np.nan
            if component_1 is upstream of component 2.
        """
        # Make component names strings.
        component_1 = str(component_1)
        component_2 = str(component_2)

        # All links downstream of component 2.
        downstream_links = self.get_downstream_components(component_2, component_2_type, 'Link')

        # Handle edge cases.
        if component_1 == component_2 and component_1_type == component_2_type:
            return 0.0
        elif  not downstream_links:
            # Check if component_1 is in downstream links.
            return np.nan
        elif downstream_links[0] == component_1 and component_1_type == 'Link':
            # If there are no between links, component_1 == component_2, return length of current link.
            return self.get_link_length(component_1)
        
        # List of links between component 2 and component 1.
        # TODO: if component_1 isn't a link, it will not be found.
        between_links = []
        for link in downstream_links:
            if link == component_1 and component_1_type == 'Link':
                break
            else:
                between_links.append(link)

        # Length of links in between_links.
        between_lengths = [self.get_link_length(link) for link in between_links]

        # Upstream distance.
        dist = np.sum(between_lengths)

        return dist
    

    def get_upstream_links_OLD(self, node_id: str) -> List:
        """
        Gets the link id(s) upstream of a specified node.

        :param node_id: ID of node.
        :return List of upstream links. If there is only one upstream link, the list will have a length of 1.
        """
        to_column_name = 'To Node'
        link_sections = ['CONDUITS', 'WEIRS', 'PUMPS', 'ORIFICES', 'OUTLETS']

        upstream_links = []
        for link_section in link_sections:
            # Check if section exists. If not, skip.
            if file_utils.check_for_section(self.inp_path, link_section) is False:
                continue

            # DataFrame corresponding to link section.
            section_df = file_utils.inp_section_to_dataframe(self.inp_path, link_section)

            # Extend list of upstream links.
            upstream_links.extend(list(section_df.loc[section_df[to_column_name]==node_id, :].iloc[:,0]))

        return upstream_links
    



    def get_link_geometry(self, link_id):
        """
        Gets a link's geometry.

        :param link_id: Link ID.
        :return: Geometry of link. 4 item list for "Geom1" through "Geom4"
        """
        # Setting variables.
        section = 'XSECTIONS'
        column_names = ['Geom1', 'Geom2', 'Geom3', 'Geom4']
        component_name = link_id

        # Get the link diameter.
        link_geometry = []
        for column_name in column_names:
            geom = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)
            # geom = float(geom)
            link_geometry.append(geom)

        return link_geometry


    def set_link_geometry(self, link_id, geom):
        """
        Set a link's geometry.

        :param link_id: Link ID to edit.
        :param geom: Four item list with values for "Geom1" through "Geom4".
        :return: Prints success if verbose.
        """
        # Setting variables.
        section = 'XSECTIONS'
        column_names = ['Geom1', 'Geom2', 'Geom3', 'Geom4']
        component_name = link_id

        # Set the new geometries.
        for i, column_name in enumerate(column_names):
            new_value = geom[i]
            file_utils.set_inp_section(self.inp_path, section, column_name, component_name, new_value)

        if self.verbose == 1:
            print(f'Set Link {link_id} geometry to {geom}')

        return geom
    

    def get_link_nodes(self, link_id):
        """
        Get the from and to node IDs at the end of a link.

        :param link_id: Link ID.
        :return: from_node_id, to_node_id
        """
        # Get the endpoint node IDs.
        conduit_section = 'CONDUITS'
        from_column_name = 'From Node'
        to_column_name = 'To Node'
        component_name = link_id

        # from_node_id = file_utils.get_inp_section(self.inp_path, conduit_section, from_column_name, component_name)
        # to_node_id = file_utils.get_inp_section(self.inp_path, conduit_section, to_column_name, component_name)

        conduit_df = self.inp_db[conduit_section]
        from_node_id = conduit_df.loc[conduit_df['Name']==component_name, from_column_name].iloc[0]
        to_node_id = conduit_df.loc[conduit_df['Name']==component_name, to_column_name].iloc[0]

        return from_node_id, to_node_id


    def get_link_roughness(self, link_id):
        """
        Gets a link's roughness.

        :param link_id: Link ID.
        :return: Roughness of link.
        """
        # Setting variables.
        section = 'CONDUITS'
        column_name = 'Roughness'
        component_name = link_id

        # Get the link roughness.
        link_roughness = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)
        link_roughness = float(link_roughness)

        return link_roughness


    def set_link_roughness(self, link_id, roughness):
        """
        Set a link's roughness.

        :param link_id: Link ID to edit.
        :param roughness: New link roughness.
        :return: Prints success if verbose.
        """
        # Setting variables.
        section = 'CONDUITS'
        column_name = 'Roughness'
        component_name = link_id
        new_value = roughness

        # Set the new diameter.
        file_utils.set_inp_section(self.inp_path, section, column_name, component_name, new_value)

        if self.verbose == 1:
            print(f'Set Link {link_id} roughness to {new_value}')

        return new_value


    def get_link_length(self, link_id):
        """
        Gets a link's length.

        :param link_id: Link ID.
        :return: Length of link.
        """
        # Setting variables.
        section = 'CONDUITS'
        column_name = 'Length'
        component_name = link_id

        # Get the link length.
        link_length = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)
        link_length = float(link_length)

        return link_length


    def set_link_length(self, link_id, length):
        """
        Set a link's length.

        :param link_id: Link ID to edit.
        :param length: New link length.
        :return: Prints success if verbose.
        """
        # Setting variables.
        section = 'CONDUITS'
        column_name = 'Length'
        component_name = link_id
        new_value = length

        # Set the new diameter.
        file_utils.set_inp_section(self.inp_path, section, column_name, component_name, new_value)

        if self.verbose == 1:
            print(f'Set Link {link_id} length to {new_value}')

        return new_value


    def get_link_offsets(self, link_id):
        """
        Gets link upstream and downstream offsets.

        :param link_id: Link ID.
        :return: Link offsets as a tuple. (Inlet Offset, Outlet Offset)
        """
        # Link variables.
        section = 'CONDUITS'
        in_column_name = 'InOffset'
        out_column_name = 'OutOffset'
        component_name = link_id

        # Get the link offsets.
        in_offset = file_utils.get_inp_section(self.inp_path, section, in_column_name, component_name)
        in_offset = float(in_offset)
        out_offset = file_utils.get_inp_section(self.inp_path, section, out_column_name, component_name)
        out_offset = float(out_offset)

        return (in_offset, out_offset)


    def set_link_offsets(self, link_id, offsets):
        """
        Sets the inlet and outlet offsets for a link.

        :param link_id: ID of the link.
        :param offsets: Link offsets as a tuple (Inlet Offset, Outlet Offset)
        :return: Offsets.
        """
        # Link variables.
        section = 'CONDUITS'
        in_column_name = 'InOffset'
        out_column_name = 'OutOffset'
        component_name = link_id
        inlet_value = offsets[0]
        outlet_value = offsets[1]

        # Set the new offsets.
        file_utils.set_inp_section(self.inp_path, section, in_column_name, component_name, inlet_value)
        file_utils.set_inp_section(self.inp_path, section, out_column_name, component_name, outlet_value)

        if self.verbose == 1:
            print(f'Set Link {link_id} offsets to {offsets}')

        return offsets


    def get_link_seepage(self, link_id):
        """
        Gets link seepage rate.

        :param link_id: Link ID.
        :return: Link seepage rate.
        """
        # Link variables.
        section = 'LOSSES'
        column_name = 'Seepage'
        component_name = link_id

        # Get the link seepage rate.
        seepage_rate = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)

        return float(seepage_rate)


    def set_link_seepage(self, link_id, seepage_rate):
        """
        Set the link seepage rate. Roughly equivalent to hydraulic conductivity.

        :param link_id: ID of link.
        :param seepage_rate: Seepage rate in project units.
        :return: Prints success if verbose.
        """
        # Setting variables.
        section = 'LOSSES'
        column_name = 'Seepage'
        component_name = link_id
        new_value = seepage_rate

        # Set the new diameter.
        file_utils.set_inp_section(self.inp_path, section, column_name, component_name, new_value)

        if self.verbose == 1:
            print(f'Set Link {link_id} seepage rate to {new_value}')

        return new_value


    def get_link_slope(self, link_id):

        # Length of link.
        link_length = self.get_link_length(link_id)

        # Get the endpoint node IDs.
        conduit_section = 'CONDUITS'
        from_column_name = 'From Node'
        to_column_name = 'To Node'
        component_name = link_id

        from_node_id = file_utils.get_inp_section(self.inp_path, conduit_section, from_column_name, component_name)
        to_node_id = file_utils.get_inp_section(self.inp_path, conduit_section, to_column_name, component_name)

        # Find what type of node the from and to nodes are.
        from_node_section = self.get_node_section(from_node_id)
        to_node_section = self.get_node_section(to_node_id)

        # Get from and to node elevations.
        from_elevation_column = 'Elevation'
        to_elevation_column = 'Elevation'
        if from_node_section == 'STORAGE':
            from_elevation_column = 'Elev.'
        if to_node_section == 'STORAGE':
            to_elevation_column = 'Elev.'

        from_node_elev = file_utils.get_inp_section(self.inp_path, from_node_section, from_elevation_column, from_node_id)
        to_node_elev = file_utils.get_inp_section(self.inp_path, to_node_section, to_elevation_column, to_node_id)

        # Compute slope. Downward is negative.
        from_node_elev = float(from_node_elev)
        to_node_elev = float(to_node_elev)
        S = (to_node_elev - from_node_elev) / link_length

        return S
    

    @staticmethod
    def node_positions(s1, s2, n_segment, n_round=3):
        """
        Generate equally spaced coordinates between two points.

        This function computes `n_segment - 1` intermediate coordinates between 
        two given points, `s1` and `s2`, such that they divide the segment into 
        `n_segment` equal parts.

        Parameters
        ----------
        s1 : ndarray
            A 1D NumPy array representing the coordinates of the starting point.
        s2 : ndarray
            A 1D NumPy array representing the coordinates of the ending point.
        n_segment : int
            The total number of segments to divide the line into.
        n_round : int, optional
            The number of decimal places to round the coordinates to (default is 3).

        Returns
        -------
        ndarray
            A (n_segment-1, 2) NumPy array containing the intermediate coordinates.

        Notes
        -----
        - The function assumes `s1` and `s2` are 2D points (i.e., have two elements).
        - The function does not include `s1` and `s2` in the returned array.
        - The segments are evenly spaced along the straight line connecting `s1` and `s2`.

        Examples
        --------
        >>> import numpy as np
        >>> s1 = np.array([6145.251, 8446.927])
        >>> s2 = np.array([3709.497, 4905.028])
        >>> node_positions(s1, s2, n_segment=5)
        array([[5430.75 , 7591.962],
            [4716.248, 6737.005],
            [4001.747, 5882.038],
            [3287.246, 5027.071]])
        """
        u = s2 - s1
        L = np.linalg.norm(u)
        n = u / L  # Unit vector

        c = L / n_segment  # Segment length

        # Generate points between s1 and s2
        I = np.arange(1, n_segment)  # Exclude start and end points
        uc = s1 + np.outer(I, n) * c

        return np.round(uc, n_round)

    
    @staticmethod
    def discretize_invert_elevations(h1, h2, n_segment):
        """
        Invert elevations for discretized nodes.

        :param h1: Invert elevation of point 1.
        :param h2: Invert elevation of point 2.
        :param n_segment: Number of segments that the original link was discretized into.

        :return: Invert elevations of discretized nodes.
        """

        return np.linspace(h1, h2, n_segment + 1)[1:-1]
    

    def discretize_link(self, link_id, n_segment):
        """
        Discretize link into N segments that preserve the original properties of the original link.

        :param swmm: SWMM model object.
        :param link_id: Link to discretize.
        :n_segment: Number of segments to discretize link into.
        """
        # Get upstream and downstream node IDs.
        from_node_id, to_node_id = self.get_link_nodes(link_id)
        
        # Get coordinates of the nodes.
        s1 = self.get_node_coordinates(from_node_id)
        s2 = self.get_node_coordinates(to_node_id)

        # Get discretized node coordinates.
        Dnode_coords = SWMM.node_positions(s1, s2, n_segment)

        # Get invert elevations of from and to nodes.
        h1 = self.get_node_invert_elevation(from_node_id)
        h2 = self.get_node_invert_elevation(to_node_id)

        # Invert elevations of discretized nodes.
        Dnode_elev = SWMM.discretize_invert_elevations(h1, h2, n_segment)

        # Format new nodes.
        #------------------------------------------------
        # Get IDs of existing coordinate nodes.
        exist_node_ids = self.get_component_names('COORDINATES')
        exist_node_ids = [int(i) for i in exist_node_ids]
        
        # Create new discretized node ids.
        Dnode_ids = [max(exist_node_ids) + i + 1 for i in range(len(Dnode_coords))]

        # Set surcharge depth to 100 + maximum invert elevation.
        max_elev = max(self.get_node_invert_elevation(int(nid)) for nid in exist_node_ids)
        sur_depth = max_elev + 100

        # Add new junctions.
        for i in range(Dnode_coords.shape[0]):
            coord_dict = {'Node': Dnode_ids[i], 'X-Coord': Dnode_coords[i,0], 'Y-Coord': Dnode_coords[i,1]}
            file_utils.add_inp_row(self.inp_path, 'COORDINATES', coord_dict)

            junction_dict = {'Name': Dnode_ids[i], 'Elevation': Dnode_elev[i], 'MaxDepth': 0, 'InitDepth': 0, 'SurDepth': sur_depth, 'Aponded': 0}
            file_utils.add_inp_row(self.inp_path, 'JUNCTIONS', junction_dict)

        # Format new conduits.
        #------------------------------------------------
        # Existing conduit parameters.
        geom = self.get_link_geometry(link_id)
        L = self.get_link_length(link_id)
        roughness = self.get_link_roughness(link_id)

        # New conduit segment length.
        Lc = L / n_segment

        # Remove the conduit from the inp file.
        file_utils.remove_inp_row(self.inp_path, 'CONDUITS', link_id)
        file_utils.remove_inp_row(self.inp_path, 'XSECTIONS', link_id)

        # Existing conduit IDs.
        # Get IDs of existing coordinate nodes.
        exist_link_ids = self.get_component_names('CONDUITS')
        exist_link_ids = [int(i) for i in exist_link_ids]
        
        # Create new discretized node ids.
        c_ids = [str(max(exist_link_ids) + i + 1) for i in range(n_segment)]

        # Set new CONDUITS AND XSECTIONS sections.
        for i in range(n_segment):
            # From and to nodes.
            if i == 0:
                from_node = from_node_id
                to_node = Dnode_ids[i]
            elif i == n_segment - 1:
                from_node = Dnode_ids[i-1]
                to_node = to_node_id
            else:
                from_node = Dnode_ids[i-1]
                to_node = Dnode_ids[i]

            xsection_dict = {'Link': c_ids[i], 'Shape': 'CIRCULAR', 'Geom1': geom[0], 'Geom2': geom[1], 'Geom3': geom[2], 'Geom4': geom[3], 'Barrels': 1, 'Culvert': ''}
            conduits_dict = {'Name': c_ids[i], 'From Node': from_node, 'To Node': to_node, 'Length': Lc, 'Roughness': roughness, 'InOffset': 0, 'OutOffset': 0, 'InitFlow': 0, 'MaxFlow': 0}

            file_utils.add_inp_row(self.inp_path, 'XSECTIONS', xsection_dict)
            file_utils.add_inp_row(self.inp_path, 'CONDUITS', conduits_dict)


    def get_weir_property(self, weir_id, weir_property_name):

        # Configurations.
        weir_section = 'WEIRS'
        component_name = weir_id
        component_property_name = weir_property_name

        # Get the weir property.
        weir_property = file_utils.get_inp_section(self.inp_path, weir_section, component_property_name, component_name)

        return weir_property


    def set_weir_property(self, weir_id, weir_property_name, weir_property):

        # Configurations.
        section = 'WEIRS'
        column_name = weir_property_name
        component_name = weir_id
        new_value = weir_property

        # Get the weir property.
        file_utils.set_inp_section(self.inp_path, section, column_name, component_name, new_value)

        if self.verbose == 1:
            print(f'Set weir {weir_id} {column_name} to {new_value}')


    def get_weir_geometry(self, weir_id):

        # Setting variables.
        section = 'XSECTIONS'
        column_names = ['Shape', 'Geom1', 'Geom2', 'Geom3', 'Geom4']
        component_name = weir_id

        # Get the weir geometry.
        weir_geometry = []
        for column_name in column_names:
            geom = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)
            if column_name != 'Shape':
                geom = float(geom)
            weir_geometry.append(geom)

        return weir_geometry


    def set_weir_geometry(self, weir_id, geom):

        # Setting variables.
        section = 'XSECTIONS'
        column_names = ['Shape', 'Geom1', 'Geom2', 'Geom3', 'Geom4']
        component_name = weir_id

        # Set the new geometries.
        for i, column_name in enumerate(column_names):
            new_value = geom[i]
            file_utils.set_inp_section(self.inp_path, section, column_name, component_name, new_value)

        if self.verbose == 1:
            print(f'Set weir {weir_id} geometry to {geom}')


    def get_storage_property(self, storage_id, property):
        """
        Get the value of a storage property.

        :param storage_id: ID of storage unit.
        :param property: Name of the property.
        :return: Value of the storage property. Evaluates type automatically.
        """
        section = 'STORAGE'
        column_name = property
        component_name = storage_id

        # Get the storage property.
        property_value = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)

        return eval(property_value)


    def set_storage_property(self, storage_id, property, property_value):
        """
        Get the value of a storage property.

        :param storage_id: ID of storage unit.
        :param property: Name of the property.
        :param property_value: New property value.
        :return: Prints success message.
        """
        section = 'STORAGE'
        column_name = property
        new_value = property_value
        component_name = storage_id

        # Set the storage property.
        file_utils.set_inp_section(self.inp_path, section, column_name, component_name, new_value)

        if self.verbose == 1:
            print(f'Set Storage {storage_id} {property} to {new_value}')


    def get_storage_outlet(self, storage_id):
        """
        Gets the outlet link for a storage component.

        :param storage_id: ID of storage unit.
        :return: ID of outlet link.
        """
        # Get conduit names.
        conduit_names = file_utils.get_component_names(self.inp_path, 'CONDUITS')

        # Loop through conduit names until the "from_node" is the storage node.
        conduit_section = 'CONDUITS'
        from_column_name = 'From Node'
        outlet_link_id = None
        for component_name in conduit_names:
            from_node_id = file_utils.get_inp_section(self.inp_path, conduit_section, from_column_name, component_name)

            if from_node_id == str(storage_id):
                outlet_link_id = component_name
                break

        return outlet_link_id

    def get_storage_inlet(self, storage_id):
        """
        Gets the inlet link for a storage component.

        :param storage_id: ID of storage unit.
        :return: ID of inlet link.
        """
        # Get conduit names.
        conduit_names = file_utils.get_component_names(self.inp_path, 'CONDUITS')

        # Loop through conduit names until the "to_node" is the storage node.
        conduit_section = 'CONDUITS'
        from_column_name = 'To Node'
        inlet_link_id = None
        for component_name in conduit_names:
            from_node_id = file_utils.get_inp_section(self.inp_path, conduit_section, from_column_name, component_name)

            if from_node_id == str(storage_id):
                inlet_link_id = component_name
                break

        return inlet_link_id


    def set_raingage_timeseries(self, raingage_id, timeseries_name):
        """
        Assign a timeseries to a rain gage.

        :param raingage_id: Rain gage ID.
        :param timeseries_name: Timeseries name.
        :return: None
        """

        section = 'RAINGAGES'
        rg_ts_name = f'TIMESERIES {timeseries_name}'
        component_name = raingage_id
        column_name = 'Source'
        file_utils.set_raingage(self.inp_path, column_name, component_name, rg_ts_name)


    def add_prcp_timeseries(self, ts_name, ts_description, times, values, dates=None, overwrite=False):
        # TODO: Add functionality to assign a file as a timeseries.
        file_utils.add_prcp_timeseries(self.inp_path, ts_name, ts_description, times, values, dates=dates, overwrite=overwrite)

    
    def add_timeseries_file(self, ts_name: str, ts_description: str, file_path: Path) -> None:
        """
        Add timeseries file to SWMM model.

        :param ts_name: Name of time series. No spaces.
        :param ts_description: Description of time series.
        :file_path: Path to .dat time series file.

        :return: None.
        """
        file_utils.add_timeseries_file(self.inp_path, ts_name, ts_description, file_path, overwrite=False)
        

    def set_node_inflow(self, node_id: Union[str, int], ts_name: str, Sfactor=1.0) -> None:
        """
        # TODO: NOT COMPLETE.
        Add an inflow time series specified by a file to a node. Type is FLOW.

        :param node_id: ID of node to add inflow to.
        :param ts_name: Name of time series. Must match the <timeseries name>.dat
        :param Sfactor: Scale factor. Default = 1.0.
        """
        section = 'INFLOWS'

        data_dict = {
            'Constituent': 'FLOW',
            'Time Series': ts_name,
            'Type': 'FLOW',
            'Mfactor': 1.0,
            'Sfactor': Sfactor
        }

        # Set the inflow.
        for column_name, new_value in data_dict.items():
            file_utils.set_inp_section(self.inp_path, section, column_name, node_id, new_value)


    # OUTPUT METHODS
    # ----------------------------------------------------------------------------------------------------------
    def unpack_series(self, series):
        "Unpacks SWMM output series into datetime and values."
        dts = [key for key in series.keys()]
        values = [val for val in series.values()]

        return dts, values


    def get_node_depth(self):
        """
        Get node depths.

        :return: Pandas data frame of node depths.
        """
        node_attribute = 'Depth'
        depth_df = self._get_node_series(node_attribute)

        return depth_df


    def get_node_flooding(self):
        """
        Get node flooding.

        :return: Pandas data frame of node flooding.
        """
        node_attribute = 'Flood'
        flood_df = self._get_node_series(node_attribute)

        return flood_df


    def get_node_head(self):
        """
        Get node head.

        :return: Pandas data frame of node head.
        """
        node_attribute = 'Head'
        head_df = self._get_node_series(node_attribute)

        return head_df


    def get_node_total_inflow(self):
        """
        Get node total inflow.

        :return: Pandas data frame of node total inflow.
        """
        node_attribute = 'Total_Inflow'
        total_inflow_df = self._get_node_series(node_attribute)

        return total_inflow_df


    def get_node_lateral_inflow(self):
        """
        Get node lateral inflow.

        :return: Pandas data frame of node lateral inflow.
        """
        node_attribute = 'Lateral_Inflow'
        lateral_inflow_df = self._get_node_series(node_attribute)

        return lateral_inflow_df


    def get_node_ponded_volume(self):
        """
        Get node ponded volume.

        :return: Pandas data frame of node ponded volume.
        """
        node_attribute = 'Ponded_Volume'
        ponded_volume_df = self._get_node_series(node_attribute)

        return ponded_volume_df


    def _get_node_series(self, node_attribute):
        """
        Get node series for a node attribute.

        :param node_attribute: Node attribute to get series for.
        :return: Pandas data frame of series of node attribute.
        """
        # Get list of node IDs.
        node_ids = file_utils.get_component_names(self.inp_path, 'JUNCTIONS')

        # Add outfall ids.
        try:
            outfall_ids = file_utils.get_component_names(self.inp_path, 'OUTFALLS')
            node_ids.extend(outfall_ids)
        except Exception as e:
            print('Model has no outfalls.')

        # Add storage nodes.
        try:
            storage_ids = file_utils.get_component_names(self.inp_path, 'STORAGE')
            node_ids.extend(storage_ids)
        except Exception as e:
            print('Model has no storage components.')

        # Remove any nodes with names starting with semicolon, these are commented out.
        node_ids = [node_id for node_id in node_ids if node_id[0] != ';']

        # Dictionary of node series.
        series_dict = {}

        # Node attributes.
        node_attr = {
            'Flood': NodeAttribute.FLOODING_LOSSES,
            'Depth': NodeAttribute.INVERT_DEPTH,
            'Head': NodeAttribute.HYDRAULIC_HEAD,
            'Total_Inflow': NodeAttribute.TOTAL_INFLOW,
            'Lateral_Inflow': NodeAttribute.LATERAL_INFLOW,
            'Ponded_Volume': NodeAttribute.PONDED_VOLUME
        }

        # Series attribute.
        series_attribute = node_attr[node_attribute]

        with Output(self.out_path) as out:
            for node_id in node_ids:
                node_dt, node_series = self.unpack_series(out.node_series(node_id, series_attribute))
                series_dict[f'{node_attribute}_node_{node_id}'] = node_series

            # Only take datetime from final node. It will be the same as the rest.
            series_dict['datetime'] = node_dt

        # Convert series_dict to Pandas DataFrame.
        series_df = pd.DataFrame(series_dict)

        return series_df


    def get_link_flow(self):
        """
        Get link flow rate.

        :return: Pandas data frame of link flow rate.
        """
        link_attribute = 'Flow'
        flow_df = self._get_link_series(link_attribute)

        return flow_df


    def get_link_depth(self):
        """
        Get link depth.

        :return: Pandas data frame of link depth.
        """
        link_attribute = 'Depth'
        depth_df = self._get_link_series(link_attribute)

        return depth_df


    def get_link_velocity(self):
        """
        Get link velocity.

        :return: Pandas data frame of link velocity.
        """
        link_attribute = 'Velocity'
        velocity_df = self._get_link_series(link_attribute)

        return velocity_df


    def get_link_volume(self):
        """
        Get link volume.

        :return: Pandas data frame of link volume.
        """
        link_attribute = 'Volume'
        volume_df = self._get_link_series(link_attribute)

        return volume_df


    def get_link_circular_Rh(self, depth, diameter):
        """
        Computes hydraulic radius (Rh) for a circular pipe.

        :return: Hydraulic radius.
        """
        theta = 2 * np.arccos(1 - 2 * depth / diameter)
        A = (diameter**2 / 8) * (theta - np.sin(theta))
        P = 0.5 * diameter * theta
        Rh = A / P

        return Rh
    
    def compute_area_from_depth(self, depth, link_id):
        """
        Computes the cross-sectional wetted area of the pipe from depth and diameter.

        :param depth: Depth of flow.
        :param link_id: ID of link.
        :return area: Cross-sectional area of water.
        """
        # Link geometry.
        geom = self.get_link_geometry(link_id)

        # Link diameter.
        D = float(geom[0])

        # Depth cannot be greater than diameter or less than 0.
        if np.any(depth > D):
            print('Found depth(s) greater than diameter -- assuming pipe is full.')
            depth[depth > D] = D
        elif np.any(depth < 0):
            depth[depth < 0] = 0
        
        # Calculate the angle theta in radians
        theta = 2 * np.arccos(1 - 2 * depth / D)
        
        # Calculate the area of the circular segment
        area = (D**2 / 8) * (theta - np.sin(theta))
        
        return area
    

    @staticmethod
    def depth_to_AP(d, D):
        theta = 2 * np.arccos(1 - 2 * d / D)
        A = (D**2 / 8) * (theta - np.sin(theta))
        P = 0.5 * D * theta

        return A, P
    

    def compute_manning_velocity(self, depth, link_id, alpha=1.49):
        """
        Compute the velocity from Manning's equation for a specific link given a time series of depth.

        :param depth: Array of depths.
        :param link_id: Link ID.
        :param alpha: Unit correction for imperial or metric. 1.49 for imperial, 1.0 for metric.

        :return: Array of velocities computed from Manning's equation.
        """

        # Slope.
        S = self.get_link_slope(link_id) * -1

        # Diameter.
        D = float(self.get_link_geometry(link_id)[0])

        # Hydraulic radius.
        Rh = np.zeros_like(depth)
        for i, d in enumerate(depth):
            Rh[i] = self.get_link_circular_Rh(d, D)

        # Roughness.
        n = self.get_link_roughness(link_id)

        # Velocity.
        v = alpha / n * S**(1 / 2) * Rh**(2 / 3)

        return v


    def compute_normal_depth_from_flow_OLD(self, Q, D, S, n):

        # Solver tolerance.
        tol = 1e-6

        # Initial guess of d.
        d = 0.5 * D
        
        # Inital area.
        A, P = self.depth_to_AP(d, D)
        A_new = (Q * n) / (S**(1 / 2) * (A / P)**(2 / 3))

        # # Initial flow.
        # Q_new = (1 / n) * A * S**(1 / 2) * (A / P)**(2 / 3)

        # Iterate for solution.
        cA = 0
        while abs(A - A_new) > tol:
            # Solve for diameter that produces A_new.
            d_new = d + 2 * tol
            cd = 0
            while abs(d - d_new) > tol:
                d = d_new
                d_new = (D / 2) * (1 - np.cos((4 * A_new) / D + np.sin(2 * np.arccos(1 - 2 * d / D))))
                print(d_new)
                if cd == 1000:
                    print(f'Loop stuck, d={d_new}, A={A_new}')
                    raise ValueError('Loop Error')
                cd += 1
            print(d_new)

            # Solve for new area.
            A, P = self.depth_to_AP(d_new, D)
            A_new = (Q * n) / (S**(1 / 2) * (A / P)**(2 / 3))

            if cA == 5:
                Q_new = (1 / n) * A_new * S**(1 / 2) * (A_new / P)**(2 / 3)
                print(f'Loop stuck, d={d_new}, A={A_new}')
                raise ValueError('Loop Error')
            cA += 1

        return d_new
    

    def compute_normal_depth_area_from_flow(self, Q, D, S, n):

        # Solver tolerance.
        tol = 1e-4

        # Initial guess of d.
        d = 0.5 * D
        theta = 0
        theta_new = 2 * np.cos(1 - 2 * d / D)
        
        c = 0
        while abs(theta - theta_new) > tol:
            theta = theta_new
            alpha = S**(1 / 2) * (D * (theta - np.sin(theta)) / (4 * theta))**(2/3)
            theta_new = (8 * Q * n) / (D**2 * (1 - (1 / theta) * np.sin(theta)) * alpha)
            print(theta_new)
            if c == 100:
                print('Loop limit reached')
                break
            c += 1

        d = 0.5 * D * (1 - np.cos(theta / 2))
        A = (D**2 / 8) * (theta - np.sin(theta))

        return d, A

    def compute_normal_depth_from_flow(self, Q, D, C):
        # Solver tolerance.
        tol = 1e-6

        # Initial guess of d.
        d = 0.5 * D
        theta = 0
        theta_new = 2 * np.cos(1 - 2 * d / D)
        
        c = 0
        while abs(theta - theta_new) > tol:
            theta = theta_new
            alpha = (D * (theta - np.sin(theta)) / (4 * theta))**(2/3)
            theta_new = (8 * Q) / (D**2 * (1 - (1 / theta) * np.sin(theta)) * alpha * C)
            print(theta_new)
            if c == 100:
                print('Loop limit reached')
                break
            c += 1

        d = 0.5 * D * (1 - np.cos(theta / 2))

        return d

    def _get_link_series(self, link_attribute):
        """
        Get link series for a link attribute.

        :param link_attribute: Link attribute to get series for.
        :return: Pandas data frame of link attribute.
        """
        # Get list of link IDs.
        link_ids = file_utils.get_component_names(self.inp_path, 'CONDUITS')

        # Dictionary of link series.
        series_dict = {}

        # Link attributes.
        attr_dict = {
            'Flow': LinkAttribute.FLOW_RATE,
            'Depth': LinkAttribute.FLOW_DEPTH,
            'Velocity': LinkAttribute.FLOW_VELOCITY,
            'Volume': LinkAttribute.FLOW_VOLUME
        }

        # Series attribute.
        series_attribute = attr_dict[link_attribute]

        with Output(self.out_path) as out:
            for link_id in link_ids:
                link_dt, link_series = self.unpack_series(out.link_series(link_id, series_attribute))
                series_dict[f'{link_attribute}_link_{link_id}'] = link_series

            # Only take datetime from final link. It will be the same as the rest.
            series_dict['datetime'] = link_dt

        # Convert series_dict to Pandas DataFrame.
        series_df = pd.DataFrame(series_dict)

        return series_df


    def get_rainfall_timeseries(self):

        rainfall_dict = {}

        with Output(self.out_path) as out:
            ts = out.system_series(SystemAttribute.RAINFALL)
            rainfall_dt, rainfall_series = self.unpack_series(ts)

        rainfall_dict['datetime'] = rainfall_dt
        rainfall_dict['prcp'] = rainfall_series

        # Convert rainfall_dict to data frame.
        rainfall_dt = pd.DataFrame(rainfall_dict)

        return rainfall_dt

    # ----------------------------------------------------------------------------------------------------------

    # def close(self):
    #     self.sim.close()
    #     print('Closed Model')

    # def __del__(self):
    #     # Close the simulation upon exiting.
    #     if self.sim:
    #         self.sim.close()
    #
    #     # Delete temporary run file if it was created.
    #     if self.cfg['temp_inp'] is True:
    #         os.remove(self.cfg['inp_path'])

def diameter_fault(swmm, fault_component, fault_value, value_type):
    # Get initial diameter.
    initial_geometry = swmm.get_link_geometry(fault_component)
    initial_diameter = initial_geometry[0]

    # Handle percentage fault or absolute fault.
    if value_type == 'fraction':
        offset = initial_diameter * fault_value
    elif value_type == 'absolute':
        offset = fault_value
    else:
        raise ValueError('Value type incorrectly specified as', value_type)

    # Update the link diameter.
    update_geometry = initial_geometry.copy()
    update_geometry[0] = initial_diameter - offset
    swmm.set_link_geometry(fault_component, update_geometry)

    # Update the link offsets.
    swmm.set_link_offsets(fault_component, (offset, offset))
    
    return swmm


def weir_fault(swmm, fault_component, fault_value):
    # Set the crest height as the fault value.
    swmm.set_weir_property(fault_component, 'CrestHt', fault_value)
    
    return swmm


def roughness_fault(swmm, fault_component, fault_value, value_type):
    # Assign fault variables.
    link_id = fault_component
    n = fault_value

    # Get link roughness.
    link_n = swmm.get_link_roughness(link_id)

    # Handle percentage fault or absolute fault.
    if value_type == 'fraction':
        n = link_n * fault_value
    elif value_type == 'absolute':
        n = fault_value
    else:
        raise ValueError('Value type incorrectly specified as', value_type)

    # Update the link roughness.
    swmm.set_link_roughness(link_id, n)
    
    return swmm
    
def silting_fault(swmm, fault_component, fault_value):
    # Assign fault variables.
    storage_id = fault_component
    silt_depth = fault_value

    # Get storage depth.
    Ds_init = swmm.get_storage_property(storage_id, 'MaxDepth')
    
    # Get inlet conduit offsets.
    inlet_link_id = swmm.get_storage_inlet(storage_id)
    (inlet_in_offset, inlet_out_offset_init) = swmm.get_link_offsets(inlet_link_id)

    # Get outlet conduit offsets.
    outlet_link_id = swmm.get_storage_outlet(storage_id)
    (outlet_in_offset_init, outlet_out_offset) = swmm.get_link_offsets(outlet_link_id)

    # Get the storage invert elevation.
    zi_init = swmm.get_storage_property(storage_id, 'Elev.')

    # Update the storage_depth.
    Ds = Ds_init - silt_depth
    swmm.set_storage_property(storage_id, 'MaxDepth', Ds)
    
    # Update inlet conduit offset.
    inlet_out_offset = inlet_out_offset_init - silt_depth
    swmm.set_link_offsets(inlet_link_id, (inlet_in_offset, inlet_out_offset))

    # Update outlet conduit offset.
    outlet_in_offset = outlet_in_offset_init - silt_depth
    swmm.set_link_offsets(outlet_link_id, (outlet_in_offset, outlet_out_offset))

    # Update storage invert elevation.
    zi = zi_init + silt_depth
    swmm.set_storage_property(storage_id, 'Elev.', zi)
    
    return swmm

if __name__ == '__main__':
    # SWMM model configuration file path.
    config_path = r"C:\Users\ay434\Documents\urbansurge\analysis\lab_system\lab_system_config.yml"

    swmm_model = SWMM(config_path)
    print(swmm_model.compute_area_from_depth())