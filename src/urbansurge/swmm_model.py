# ================================================================================
# SWMM Model Class.
# ================================================================================

# Package imports.
from urbansurge import file_utils

# Library imports.
from pyswmm import Simulation, Nodes, Links, Output
from swmm.toolkit.shared_enum import LinkAttribute, NodeAttribute, SubcatchAttribute, SystemAttribute
import yaml
import shutil
import os
import pandas as pd


class SWMM:
    def __init__(self, config_path):
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
            for link_id, roughness in conduit_lengths.items():
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
        '''
        Create a temporary inp file to run the model from.
        :return: Sets the name of self.cfg['inp_path'] to temporary path.
        '''
        # Create new temporary file path.
        split_inp_path = os.path.splitext(self.cfg['inp_path'])
        inp_temp_path = split_inp_path[0] + '_tmp' + split_inp_path[1]

        # Copy inp file to temporary new file.
        shutil.copy(self.cfg['inp_path'], inp_temp_path)

        # Update name in self.cfg.
        self.cfg['inp_path'] = inp_temp_path


    def run_simulation(self):
        # Instantiate model.
        with Simulation(self.inp_path) as sim:
            for ind, step in enumerate(sim):
                if ind % 100 == 0:
                    print(sim.current_time, ",", round(sim.percent_complete * 100))


    def get_link_geometry(self, link_id):
        '''
        Gets a link's geometry.
        :param link_id: Link ID.
        :return: Geometry of link. 4 item list for "Geom1" through "Geom4"
        '''
        # Setting variables.
        section = 'XSECTIONS'
        column_names = ['Geom1', 'Geom2', 'Geom3', 'Geom4']
        component_name = link_id

        # Get the link diameter.
        link_geometry = []
        for column_name in column_names:
            geom = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)
            geom = float(geom)
            link_geometry.append(geom)

        return link_geometry


    def set_link_geometry(self, link_id, geom):
        '''
        Set a link's geometry.
        :param link_id: Link ID to edit.
        :param geom: Four item list with values for "Geom1" through "Geom4".
        :return: Prints success if verbose.
        '''
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


    def get_link_roughness(self, link_id):
        '''
        Gets a link's roughness.
        :param link_id: Link ID.
        :return: Roughness of link.
        '''
        # Setting variables.
        section = 'CONDUITS'
        column_name = 'Roughness'
        component_name = link_id

        # Get the link roughness.
        link_roughness = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)
        link_roughness = float(link_roughness)

        return link_roughness


    def set_link_roughness(self, link_id, roughness):
        '''
        Set a link's roughness.
        :param link_id: Link ID to edit.
        :param roughness: New link roughness.
        :return: Prints success if verbose.
        '''
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
        '''
        Gets a link's length.
        :param link_id: Link ID.
        :return: Roughness of link.
        '''
        # Setting variables.
        section = 'CONDUITS'
        column_name = 'Length'
        component_name = link_id

        # Get the link roughness.
        link_length = file_utils.get_inp_section(self.inp_path, section, column_name, component_name)
        link_length = float(link_length)

        return link_length


    def set_link_length(self, link_id, length):
        '''
        Set a link's length.
        :param link_id: Link ID to edit.
        :param length: New link length.
        :return: Prints success if verbose.
        '''
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


    def set_raingage_timeseries(self, raingage_id, timeseries_name):
        '''
        Assign a timeseries to a rain gage.
        :param raingage_id: Rain gage ID.
        :param timeseries_name: Timeseries name.
        :return: None
        '''

        section = 'RAINGAGES'
        rg_ts_name = f'TIMESERIES {timeseries_name}'
        component_name = raingage_id
        column_name = 'Source'
        file_utils.set_raingage(self.inp_path, column_name, component_name, rg_ts_name)


    def add_prcp_timeseries(self, ts_name, ts_description, times, values, dates=None, overwrite=False):
        # TODO: Add functionality to assign a file as a timeseries.
        file_utils.add_prcp_timeseries(self.inp_path, ts_name, ts_description, times, values, dates=dates, overwrite=overwrite)


    def unpack_series(self, series):
        "Unpacks SWMM output series into datetime and values."
        dts = [key for key in series.keys()]
        values = [val for val in series.values()]

        return dts, values


    def get_node_depths(self):
        '''
        Get node depths.
        :return: Pandas data frame of node depths.
        '''
        # Get list of node IDs.
        node_ids = file_utils.get_component_names(self.inp_path, 'JUNCTIONS')

        # Add outfall ids.
        outfall_ids = file_utils.get_component_names(self.inp_path, 'OUTFALLS')
        node_ids.extend(outfall_ids)

        # Dictionary of node depths.
        depth_dict = {}

        with Output(self.out_path) as out:
            for node_id in node_ids:
                node_dt, node_series = self.unpack_series(out.node_series(node_id, NodeAttribute.INVERT_DEPTH))
                depth_dict[f'{node_id}_depth'] = node_series

            # Only take datetime from final node. It will be the same as the rest.
            depth_dict['datetime'] = node_dt

        # Convert depth_dict to Pandas DataFrame.
        depth_df = pd.DataFrame(depth_dict)

        return depth_df


    def get_node_flooding(self):
        '''
        Get node flooding.
        :return: Pandas data frame of node flooding.
        '''
        # Get list of node IDs.
        node_ids = file_utils.get_component_names(self.inp_path, 'JUNCTIONS')

        # Add outfall ids.
        outfall_ids = file_utils.get_component_names(self.inp_path, 'OUTFALLS')
        node_ids.extend(outfall_ids)

        # Dictionary of node depths.
        flood_dict = {}

        with Output(self.out_path) as out:
            for node_id in node_ids:
                node_dt, node_series = self.unpack_series(out.node_series(node_id, NodeAttribute.FLOODING_LOSSES))
                flood_dict[f'{node_id}_flood'] = node_series

            # Only take datetime from final node. It will be the same as the rest.
            flood_dict['datetime'] = node_dt

        # Convert depth_dict to Pandas DataFrame.
        flood_df = pd.DataFrame(flood_dict)

        return flood_df


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
