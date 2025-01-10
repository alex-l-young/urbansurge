##########################################################################
# Tests for urbansurge/data_acquisition/reflection_filter.py
# Alex Young
##########################################################################

# Library imports.
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Urbansurge imports.
from urbansurge.data_acquisition import reflection_filter


class TestAnalysisTools():
    def setup(self):
        self.data_dir = Path(r'tests/test_files/s18uuaq_data')
        self.fnames = [
            '21-Nov-2024_sensor_data0.csv'
        ]
        self.data = [pd.read_csv(self.data_dir / f) for f in self.fnames]

    def test_filter_reflections(self):
        # Get data.
        df = self.data[0]

        # Time stamps.
        t = pd.to_datetime(df['time'])

        # Voltage readings from AI0 port.
        v0 = df['V_ai0']

        # Voltage readings from AI1 port.
        v1 = df['V_ai1']

        plt.plot(t, v0)
        plt.show()
