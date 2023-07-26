# Tests for urbansurge/file_utils.py

# Imports.
from urbansurge import file_utils

class TestFileUtils:
    def setup(self):
        self.in_filepath = r"test_files\Canandaigua.inp"
        self.out_filepath = r"test_files\Canandaigua.out"

    def test_get_component_names(self):
        sections = ['CONDUITS', 'JUNCTIONS', 'SUBCATCHMENTS']

        for section in sections:
            name_list = file_utils.get_component_names(self.in_filepath, section)

            assert isinstance(name_list, list)
            assert len(name_list) > 0