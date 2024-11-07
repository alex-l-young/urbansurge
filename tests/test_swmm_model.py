##########################################################################
# Tests for urbansurge/swmm_model.py
# Alex Young
##########################################################################

# Imports.
from urbansurge import swmm_model

# Library imports.
import numpy as np
import os
import pandas as pd

class TestSWMM():
    def setup(self):
        self.in_filepath = r"test_files\lab_system.inp"
        self.config_path = r"test_files\lab_system_config.yml"

        self.swmm = swmm_model.SWMM(self.config_path)
        self.swmm.configure_model()

    def test_get_link_geometry(self):
        link_id = 2

        link_geometry = self.swmm.get_link_geometry(link_id)
        print('Link Geometry: {}'.format(link_geometry))

        assert len(link_geometry) > 0


    def test_set_link_geometry(self):
        link_id = 2
        new_geom = [4.5, 0, 0, 0]

        return_diameter = self.swmm.set_link_geometry(link_id, new_geom)

        assert return_diameter == new_geom


    def test_create_temp_inp(self):
        self.swmm._create_temp_inp()

        assert '_tmp.inp' in self.swmm.cfg['inp_path']
        assert os.path.exists(self.swmm.cfg['inp_path'])


    def test_configure_model(self):
        self.swmm.configure_model()


    def test_get_node_depths(self):
        self.swmm.configure_model()
        self.swmm.run_simulation()
        node_depths = self.swmm.get_node_depth()

        assert isinstance(node_depths, pd.DataFrame)
        assert not node_depths.empty

    def test_get_rainfall_timeseries(self):
        self.swmm.configure_model()
        self.swmm.run_simulation()
        prcp_df = self.swmm.get_rainfall_timeseries()

        assert isinstance(prcp_df, pd.DataFrame)
        assert set(prcp_df.columns) == {'datetime', 'prcp'}
        assert not prcp_df.empty

    def test_get_storage_outfall_link(self):
        self.swmm.configure_model()
        self.swmm.run_simulation()

        storage_id = 20
        outlet_link_id = self.swmm.get_storage_outlet(storage_id)

        assert outlet_link_id == str(43)

    def test_get_downstream_components(self):
        start_component_id = 39
        start_component_type = 'Link'
        downstream_component_types = ['Link', 'Junction', 'Storage', 'Outfall']

        self.swmm.configure_model()

        # True downstream links.
        true_downstream_links = [39, 77, 40, 41, 78, 42, 43, 79, 44, 45, 80, 23, 21, 76, 20]
        true_downstream_nodes = [56, 55, 17, 51, 52, 53, 54, 15, 48, 47, 2, 21, 22]
        true_downstream_storages = [20]
        true_downstream_outfalls = [1]

        # Convert to strings.
        true_downstream_links = [str(i) for i in true_downstream_links]
        true_downstream_nodes = [str(i) for i in true_downstream_nodes]
        true_downstream_storages = [str(i) for i in true_downstream_storages]
        true_downstream_outfalls = [str(i) for i in true_downstream_outfalls]

        for downstream_component_type in downstream_component_types:
            downstream_components = self.swmm.get_downstream_components(start_component_id, start_component_type,
                                                                        downstream_component_type)

            if downstream_component_type == 'Link':
                assert set(downstream_components) == set(true_downstream_links)
            elif downstream_component_type == 'Junction':
                assert set(downstream_components) == set(true_downstream_nodes)
            elif downstream_component_type == 'Storage':
                assert set(downstream_components) == set(true_downstream_storages)
            elif downstream_component_type == 'Outfall':
                assert set(downstream_components) == set(true_downstream_outfalls)

    def test_upstream_distance(self):
        component_1 = 20
        component_1_type = 'Link'
        component_2 = 76
        component_2_type = 'Link'

        true_distance = 2025
        true_distance = 5

        distance = self.swmm.upstream_distance(component_1, component_1_type, component_2, component_2_type)

        assert distance == true_distance

