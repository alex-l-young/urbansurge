##########################################################################
# Tests for urbansurge/swmm_model.py
# Alex Young
##########################################################################

# Imports.
from urbansurge import swmm_model

# Library imports.
import os
import pandas as pd

class TestSWMM():
    def setup(self):
        self.in_filepath = r"test_files\Canandaigua_physical_system.inp"
        self.config_path = r"test_files\canandaigua_config_physical.yml"

        self.swmm = swmm_model.SWMM(self.config_path)


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

        storage_id = 21
        outfall_link_id = self.swmm.get_storage_outfall_link(storage_id)

        assert outfall_link_id == str(11)
