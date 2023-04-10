# ================================================================================
# SWMM Model Class.
# ================================================================================

# Library imports.
from pyswmm import Simulation, Nodes, Links
import yaml


class SWMM:
    def __init__(self, config_path):
        # Parse the configuration file into a dictionary.
        self.cfg = self._parse_config(config_path)

        # Instantiate model.
        self.sim = Simulation(self.cfg['swmm_path'])

        # Simulation information
        print("Simulation info")
        flow_units = self.sim.flow_units
        print("Flow Units: {}".format(flow_units))
        system_units = self.sim.system_units
        print("System Units: {}".format(system_units))
        print("Start Time: {}".format(self.sim.start_time))
        print("Start Time: {}".format(self.sim.end_time))


    def _parse_config(self, config_path: str) -> dict:
        """
        Parses the configuration file.
        :param config_path: Path to configuration file.
        :return: Configuration dictionary.
        """
        with open(config_path, "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)

        return cfg


    def run_simulation(self):
        for ind, step in enumerate(self.sim):
            if ind % 100 == 0:
                print(self.sim.current_time, ",", round(self.sim.percent_complete * 100))


    def update_nodes(self):
        return


    def update_links(self):
        "Update link parameters."





    def __del__(self):
        # Close the simulation upon exiting.
        if self.sim:
            self.sim.close()