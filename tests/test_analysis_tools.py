##########################################################################
# Tests for urbansurge/analysis_tools.py
# Alex Young
##########################################################################

# Imports.
from urbansurge import analysis_tools, swmm_model


class TestAnalysisTools():
    def setup(self):
        self.in_filepath = r"test_files\Canandaigua.inp"
        self.config_path = r"C:\Users\ay434\Documents\urbansurge\tests\test_files\test_config.yml"

        self.swmm = swmm_model.SWMM(self.config_path)

    def test_flatten_df(self):
        self.swmm.configure_model()
        self.swmm.run_simulation()

        # Node depth df.
        node_depths = self.swmm.get_node_depths()

        flat_df = analysis_tools.flatten_df(node_depths)

        assert len(flat_df.columns) == node_depths.shape[0] * node_depths.shape[1]
        assert flat_df.shape[0] == 1


    def test_join_df(self):
        self.swmm.configure_model()
        self.swmm.run_simulation()

        # Node depth df.
        node_depths = self.swmm.get_node_depths()

        # Precipitation df.
        prcp_df = self.swmm.get_rainfall_timeseries()

        # Merge data frames.
        node_prcp_merge_df = node_depths.merge(prcp_df, on='datetime')

        assert not node_prcp_merge_df.empty
        assert len(node_prcp_merge_df.columns) == len(node_depths.columns) + len(prcp_df.columns) - 1





