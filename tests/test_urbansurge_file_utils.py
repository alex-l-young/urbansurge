# Tests for urbansurge/file_utils.py

# Imports.
from urbansurge import file_utils

class TestFileUtils:
    def setup(self):
        self.in_filepath = r"C:\Users\ay434\Documents\urbansurge\tests\test_files\Canandaigua.inp"
        self.out_filepath = r"C:\Users\ay434\Documents\urbansurge\tests\test_files\Canandaigua.out"

    def test_get_component_names(self):
        sections = ['CONDUITS', 'JUNCTIONS', 'SUBCATCHMENTS']

        for section in sections:
            name_list = file_utils.get_component_names(self.in_filepath, section)

            assert isinstance(name_list, list)
            assert len(name_list) > 0

    def test_get_inp_section(self):
        section = 'STORAGE'
        component_name = 21
        column_name = 'Ksat'

        value = file_utils.get_inp_section(self.in_filepath, section, column_name, component_name)

        assert float(value) == 1.0

    def test_set_inp_section(self):
        section = 'STORAGE'
        component_name = 21
        column_name = 'Ksat'
        new_value = 1.5

        file_utils.set_inp_section(self.in_filepath, section, column_name, component_name, new_value,
                                           out_filepath=None)

        # Get the new value and assert they are equal.
        value = file_utils.get_inp_section(self.in_filepath, section, column_name, component_name)

        assert float(value) == new_value

    def test_inp_section_to_dataframe(self):
        sections = ['CONDUITS', 'WEIRS']
        section = 'CONDUITS'

        # Test conduits.
        inp_filepath = self.in_filepath

        section_df = file_utils.inp_section_to_dataframe(inp_filepath, section)

        # Assert that the section dataframe has been filled.
        assert section_df.shape[0] > 1