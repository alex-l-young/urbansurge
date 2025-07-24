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
from pathlib import Path

class TestSWMM():
    def setup_method(self):
        self.in_filepath = r"C:\Users\ay434\Documents\urbansurge\analysis\Bellinge\7_SWMM\BellingeSWMM_v021_nopervious_tmp.inp"
        self.config_path = r"C:\Users\ay434\Documents\urbansurge\analysis\Bellinge\7_SWMM\Bellinge_config.yml"

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


    def test_add_timeseries_file(self):
        ts_name = 'TESTTIMESERIES'
        ts_description = 'TEST TIMESERIES'
        file_path = Path(r'C:\Users\ay434\Documents\urbansurge\tests\test_files\Canandaigua_5min_period2_noise.dat')
        self.swmm.add_timeseries_file(ts_name, ts_description, file_path)


    def test_set_node_inflow(self):
        node_id = 19
        ts_name = 'DEFAULT'

        self.swmm.set_node_inflow(self, node_id, ts_name)


    def test_get_upstream_links(self):
        node_id = '28'
        upstream_links = self.swmm.get_upstream_links(node_id)

        assert len(upstream_links) > 0

    def test_get_component_names(self):
        section = 'CONDUITS'
        component_names = self.swmm.get_component_names(section)

        assert len(component_names) > 0

    def test_get_link_nodes(self):
        link_id = 'F74F370_F74F360_l1'
        from_node_id, to_node_id = self.swmm.get_link_nodes(link_id)

        assert from_node_id == 'F74F370'
        assert to_node_id == 'F74F360'

    def test_get_node_invert_elevation(self):
        # Junction.
        node_id = 'F74F370'
        invert_elevation = self.swmm.get_node_invert_elevation(node_id)
        assert invert_elevation == 9.419

        # Outfall.
        node_id = 'F74F360'
        invert_elevation = self.swmm.get_node_invert_elevation(node_id)
        assert invert_elevation == 9.409

        # Storage.
        node_id = 'G71F04R'
        invert_elevation = self.swmm.get_node_invert_elevation(node_id)
        assert invert_elevation == 12.74

        
