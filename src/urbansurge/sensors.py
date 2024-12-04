# Stormwater system data acqusition.
# ========================================================

# Library imports.
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import shutil

# UrbanSurge imports.

class Sensor:
    # Metadata entires.
    metadata_entries = [
        'sensor_name',
        'description',
        'sensor_make',
        'sensor_model'
    ]


    def __init__(self, sensor_path, sensor_name, description, sensor_make, sensor_model):
        """
        Sensor class.

        :param sensor_path: Path to top-level directory containing all sensor subdirectories.
        :param sensor_name: Name of sensor. 20 or fewer characters and can only contain letters, numbers, underscores, whitespace, and hyphens.
        :param description: Description of the sensor.
        :param sensor_make: Company that produced the sensor.
        :param sensor_model: Model of the sensor.
        :param overwrite: When set to True, deletes existing directory with the name provided and creates a fresh instantiation.
        """
        # Make sensor subdirectory.
        sensor_subdir_name = Sensor.sensor_directory_name(sensor_name)
        self.sensor_subdir_path = sensor_path / sensor_subdir_name        

        # Check if sensor directory already exists.
        if self.check_if_sensor_exists(sensor_name):
            # Load metadata and populate the class attributes.
            with open(self.sensor_subdir_path / 'metadata.json', 'r') as f:
                self.metadata = json.load(f)
            self.sensor_name = self.metadata['sensor_name']
            self.description = self.metadata['description']
            self.sensor_make = self.metadata['sensor_make']
            self.sensor_model = self.metadata['sensor_model']
        else:
            # Create sensor subdirectory.
            os.makedirs(self.sensor_subdir_path)

            # Populate 
            self.sensor_name = sensor_name
            self.description = description
            self.sensor_make = sensor_make
            self.sensor_model = sensor_model

        self.sensor_path = sensor_path
        self.sensor_name = sensor_name
        
    
    def check_if_sensor_exists(self, sensor_name):
        """
        Check if the sensor directory already exists.

        :return: True if the sensor directory exists, False if not.
        """
        if os.path.exists(self.sensor_subdir_path):
            print(f">>> Loaded '{sensor_name}'")
            return True
        else:
            return False
    

    def list_records(self):
        """
        List all data records associated with the sensor.

        :return records: List of record names.
        """
        p = Path(self.sensor_subdir_path).glob('**/*.csv')

        print(f'Records of sensor {self.sensor_name}')
        records = []
        for file in p:
            print(f'> {file}')
            records.append(file)

        return records
    

    def get_record(self, record_name):
        """
        Gets the record associated with the record name.

        :return record_df: DataFrame of record .csv file.
        """
        # Construct file path to record.
        record_fp = self.sensor_subdir_path / f'{record_name}.csv'

        # Create data frame.
        record_df = pd.read_csv(record_fp)

        return record_df
    

    def read(self):

        return
    

    @staticmethod
    def sensor_directory_name(sensor_name):
        """
        Check if sensor name can form a valid directory name.
        - 20 or fewer characters.
        - Only contains letters, numbers, underscores, whitespace, and hyphens.
        Replace white spaces with underscores.

        :param sensor_name: Name of sensor.
        :return sensor_dir: Sensor directory name.
        """
        # Replace white spaces with underscores.
        sensor_dir = sensor_name.replace(' ', '_')

        # Check if it is a valid directory name.
        if len(sensor_name) > 20:
            raise ValueError(f'Sensor name must have 20 or fewer characters, "{sensor_name}" has {len(sensor_name)}')
        elif not re.match(r'^[a-zA-Z0-9_\s-]+$', sensor_name):
            raise ValueError('Sensor name can only contain letters, numbers, underscores, and hyphens.')

        return sensor_dir
    

    def _create_metadata(self):
        """
        Create metadata for the sensor in the file metadata.json.
        """
        # Metadata file path.
        file_path = self.sensor_subdir_path / 'metadata.json'

        # Save the updated JSON back to the file.
        with open(file_path, 'w') as file:
            json.dump(self.metadata, file, indent=4)
    

    def _update_metadata(self, name, value):
        """
        Updates metadata.json with a name : value pair. 

        :param name: Name of metadata attribute.
        :param value: Value of metadata attribute.
        """
        # Metadata file path.
        file_path = self.sensor_subdir_path / 'metadata.json'

        # Try to load existing json file.
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Update data.
            data.update({name: value})
        else:
            # Create new data if the file doesn't exist.
            data = {name: value}

        # Save the updated JSON back to the file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)


    def __setattr__(self, name, value):
        """Set the attribute and update the metadata file if the applicable."""
        super().__setattr__(name, value)  # Set the attribute

        # Save to metadata.json when a public attribute is updated.
        if name in Sensor.metadata_entries: 
            self._update_metadata(name, value)         
    

# ==================================================================
# Functions.
def list_sensors(sensor_path):
    """
    Lists available sensors from sensor directory.

    :param sensor_path: Sensor directory containing all sensor sub-directories.
    :return: None
    """
    # Create list of subdirectories.
    dirpath, dirnames, _ = next(os.walk(sensor_path))
    if len(dirnames) == 0:
        # If there are no subdirectories. Raise file not found error.
        raise FileNotFoundError(f'No subdirectories found under {sensor_path}')
    else:
        subdirs = [Path(os.path.join(dirpath, subdir)) for subdir in dirnames]
    
    # Print the sensor information.
    print('Available Sensors:')
    for subdir in subdirs:
        # Sensor metadata.
        with open(subdir / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        # Sensor name and type.
        sensor_name = metadata['sensor_name']

        # Print sensor information.
        print(f'> {sensor_name}')


def load_sensor(sensor_path, sensor_name):
    """
    Loads an existing sensor based on the sensor name.

    :param sensor_path: Path to top-level sensor directory.
    :param sensor_name: Name of sensor to load.

    :return sensor: Loaded instance of sensor class.
    """
    # Create sensor subdirectory name.
    sensor_dir = Sensor.sensor_directory_name(sensor_name)

    # Access specific sensor subdirectory.
    sensor_subdir_path = sensor_path / sensor_dir

    # If it doesn't exist, raise error.
    if not os.path.exists(sensor_subdir_path):
        raise ValueError(f'Sensor with name {sensor_name} not found at {sensor_subdir_path}')
    
    # Access metadata and populate sensor.
    with open(sensor_subdir_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    # Shared sensor metadata.
    description = metadata['description']
    sensor_make = metadata['sensor_make']
    sensor_model = metadata['sensor_model']

    # Create a Sensor instance.
    sensor = Sensor(sensor_path, sensor_name, description, sensor_make, sensor_model)

    return sensor