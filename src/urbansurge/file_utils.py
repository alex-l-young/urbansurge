# ========================================================
# Hacky utilities for editing SWMM input files.
# ========================================================

# Library imports.
from collections import defaultdict
import pandas as pd
import re
from typing import List, Union


def set_inp_section(in_filepath: str, section: str, column_name: str, component_name: int, new_value: Union[int, float],
                    out_filepath=None):
    """
    Set a value in the .inp file.
    
    :param in_filepath: Path to inp file.
    :param section: Section to edit.
    :param column_name: Column to edit.
    :param component_name: Component name to edit. From "name" column.
    :param new_value: New value to set.
    :param out_filepath: If specified, creates a new .inp file with the new value. Otherwise overwrites in_filepath.
    :return: None
    """
    # If out_filename is None, use in_filename.
    if out_filepath is None:
        out_filepath = in_filepath

    with open(in_filepath, 'r') as file:
        # Read the file into a list of lines
        lines = file.readlines()

        # Find the line number where the section table starts
        start_line = None
        for i, line in enumerate(lines):
            if line.startswith('[' + section + ']'):
                start_line = i + 3  # Skip the header lines
                break

        # Find the index of the "Name" and specified column in the header line
        header_line = lines[start_line-2]
        header_values = re.split(r" {2,}", header_line.strip())
        name_col_index = 0
        column_index = header_values.index(column_name)

        # Find the line number that corresponds to the specified "Name" value
        update_line = None
        for i in range(start_line, len(lines)):
            line_values = lines[i].strip().split()
            if line_values[name_col_index] == str(component_name):
                update_line = i
                break

        if update_line is not None:
            # Handle weird storage formatting.
            if section == 'STORAGE' and column_index > 4:
                column_index += 2

            # Update the specified column's value for the found line
            line_values = lines[update_line].strip().split()
            line_values[column_index] = str(new_value)
            lines[update_line] = " ".join(line_values) + "\n"

            # Write the updated file
            with open(out_filepath, 'w') as file:
                file.writelines(lines)
            print(f"Updated {column_name} value to {new_value} for {component_name} in {component_name}")
        else:
            print(f"No line found with Name value {component_name} in {component_name}")


def get_inp_section_from_lines(lines, section, column_name, component_name):
    """
    Gets the value from a specified section, column name, and component name from the SWMM .inp file lines.
    :param lines: Lines from inp file.
    :param section: Section to choose data from. E.g., XSECTIONS
    :param column_name: Name of column to get data from.
    :param component_name: Component name.
        Will use the first column of the section table which is either "Name" or "Link".
    :return: Returns the requested value as a string.
    """
    # Find the line number where the section table starts
    start_line = None
    for i, line in enumerate(lines):
        if line.startswith('[' + section + ']'):
            start_line = i + 3  # Skip the header lines
            break

    # Find the index of the "Name" and specified column in the header line
    header_line = lines[start_line-2]
    header_values = re.split(r" {2,}", header_line.strip())
    name_col_index = 0
    column_index = header_values.index(column_name)

    # Find the line number that corresponds to the specified "Name" value
    update_line = None
    for i in range(start_line, len(lines)):
        line_values = lines[i].strip().split()
        if line_values[name_col_index] == str(component_name):
            update_line = i
            break

    if update_line is not None:
        # Handle weird storage formatting.
        if section == 'STORAGE' and column_index > 4:
            column_index += 2

        # Update the specified column's value for the found line
        line_values = lines[update_line].strip().split()
        component_value = line_values[column_index]

        # print(f"Found {column_name} value to be {component_value} for {component_name} in {component_name}")
        return component_value

    else:
        raise ValueError(f"No line found with Name value {component_name} in {component_name}")


def get_inp_section(in_filepath, section, column_name, component_name):
    """
    Gets the value from a specified section, column name, and component name from the SWMM .inp file.
    :param in_filepath: Path to .inp file.
    :param section: Section to choose data from. E.g., XSECTIONS
    :param column_name: Name of column to get data from.
    :param component_name: Component name.
        Will use the first column of the section table which is either "Name" or "Link".
    :return: Returns the requested value as a string.
    """

    with open(in_filepath, 'r') as file:
        # Read the file into a list of lines
        lines = file.readlines()

        # Find the line number where the section table starts
        start_line = None
        for i, line in enumerate(lines):
            if line.startswith('[' + section + ']'):
                start_line = i + 3  # Skip the header lines
                break

        # Find the index of the "Name" and specified column in the header line
        header_line = lines[start_line-2]
        header_values = re.split(r" {2,}", header_line.strip())
        name_col_index = 0
        column_index = header_values.index(column_name)

        # Find the line number that corresponds to the specified "Name" value
        update_line = None
        for i in range(start_line, len(lines)):
            line_values = lines[i].strip().split()
            if line_values[name_col_index] == str(component_name):
                update_line = i
                break

        if update_line is not None:
            # Handle weird storage formatting.
            if section == 'STORAGE' and column_index > 4:
                column_index += 2

            # Update the specified column's value for the found line
            line_values = lines[update_line].strip().split()
            component_value = line_values[column_index]

            # print(f"Found {column_name} value to be {component_value} for {component_name} in {component_name}")
            return component_value

        else:
            raise ValueError(f"No line found with Name value {component_name} in {component_name}")
        

def inp_section_to_dataframe(inp_filepath, section):
    """
    Convert the section table from an inp file to a Pandas DataFrame.

    :param inp_filepath: Path to SWMM input file.
    :param section: Name of the section to convert.

    :return: DataFrame of the section with columns corresponding to the section headers.
    """
    with open(inp_filepath, 'r') as file:
        # Read the file into a list of lines
        lines = file.readlines()

        # Find the line number where the section table starts
        start_line = None
        for i, line in enumerate(lines):
            if line.startswith('[' + section + ']'):
                start_line = i + 3  # Skip the header lines
                break

        # Find the index of the "Name" and specified column in the header line
        header_line = lines[start_line-2]
        header_values = re.split(r" {2,}", header_line.strip())

        # Create empty DataFrame.
        df = pd.DataFrame(columns=header_values)

        # Find the line number that corresponds to the specified "Name" value
        dfs_to_concat = []
        for i in range(start_line, len(lines)):
            # Extract line values
            line_values = lines[i].strip().split()

            # Stop when the end of the section is reached.
            if len(line_values) == 0:
                break

            # Skip line if it starts with a comment.
            if line_values[0][0] == ';':
                continue

            # Pad line values with nan if it's shorter than the number of rows.
            if len(line_values) < len(df.columns):
                pad_values = [None for _ in range(len(df.columns) - len(line_values))]
                line_values += pad_values

            # Create a new DataFrame from the list
            print(df.columns)
            print(lines)
            new_row_df = pd.DataFrame([line_values], columns=df.columns) 
            dfs_to_concat.append(new_row_df)

        # Concatenate the original DataFrame with the new row DataFrame
        df = pd.concat(dfs_to_concat, ignore_index=True)
        
        return df
    

def check_for_section(inp_filepath, section):
    """
    Checks whether the EPA SWMM input section passed to the function exists.
    """
    with open(inp_filepath, 'r') as file:
        # Read the file into a list of lines
        lines = file.readlines()

        # Find the line number where the section table starts
        found_section = False
        for i, line in enumerate(lines):
            if line.startswith('[' + section + ']'):
                found_section = True
                break
        
        return found_section

        

def get_section_column_names(in_filepath, section):
    with open(in_filepath, 'r') as file:
        # Read the file into a list of lines
        lines = file.readlines()

        # Find the line number where the section table starts
        start_line = None
        for i, line in enumerate(lines):
            if line.startswith('[' + section + ']'):
                start_line = i + 3  # Skip the header lines
                break

        # Find the index of the "Name" and specified column in the header line
        header_line = lines[start_line-2]
        header_values = re.split(r" {2,}", header_line.strip())

        # Remove the two semicolons on the first column name.
        header_values[0] = header_values[0].strip(';')

    return header_values
        

def remove_inp_row(in_filepath, section, component_name):
    with open(in_filepath, 'r') as file:
        # Read the file into a list of lines
        lines = file.readlines()

        # Find the line number where the section table starts
        start_line = None
        for i, line in enumerate(lines):
            if line.startswith('[' + section + ']'):
                start_line = i + 3  # Skip the header lines
                break

        # Find the index of the "Name" and specified column in the header line
        header_line = lines[start_line-2]
        header_values = re.split(r" {2,}", header_line.strip())
        name_col_index = 0

        # Find the line number that corresponds to the specified "Name" value
        update_line = None
        for i in range(start_line, len(lines)):
            line_values = lines[i].strip().split()
            if line_values[name_col_index] == str(component_name):
                update_line = i
                break

        if update_line is not None:
            line_to_remove = lines[update_line]
            del lines[update_line]

            # Write the updated file
            with open(in_filepath, 'w') as file:
                file.writelines(lines)
            print(f"Removed Line: {line_to_remove}")
        else:
            print(f"No line found with Name value {component_name} in {component_name}")


def add_inp_row(in_filepath, section, row_dict, out_filepath=None, verbose=False):
    # If out_filename is None, use in_filename.
    if out_filepath is None:
        out_filepath = in_filepath

    # Column names from section.
    column_names = get_section_column_names(in_filepath, section)

    with open(in_filepath, 'r') as file:
        # Read the file into a list of lines
        lines = file.readlines()

        # Find the line number where the section table starts
        start_line = None
        for i, line in enumerate(lines):
            if line.startswith('[' + section + ']'):
                start_line = i + 3  # Skip the header lines
                break

        # Populate new line.
        new_line = []
        for i, column_name in enumerate(column_names):
            new_line.append(str(row_dict[column_name]))
        new_line = " ".join(new_line) + "\n"

        # Find the last line in the section.
        for i in range(start_line, len(lines)):
            if lines[i].strip() == '':
                insert_line_idx = i
                break

        # Insert the new line.
        lines.insert(insert_line_idx, new_line)

        # Add the new line.
        try:
            # Write the updated file
            with open(in_filepath, 'w') as file:
                file.writelines(lines)
            if verbose is True:
                print(f'Added new line: {new_line}')
        except Exception as e:
            print('Could not insert line.')
            print(e)


def get_component_names(in_filepath: str, section: str) -> List:
    """
    Retrieve component names from a section in the .inp file.
    :param in_filepath: Path to inp file.
    :param section: Section from which to get values from the "names" column.
    :return: Component names from the section.
    """

    with open(in_filepath, 'r') as file:
        # Read the file into a list of lines
        lines = file.readlines()

        # Find the line number where the section table starts
        start_line = None
        for i, line in enumerate(lines):
            if line.startswith('[' + section + ']'):
                start_line = i + 3  # Skip the header lines
                break

        # If section is not found, return None.
        if start_line is None:
            print(f'No section found with name {section}.')
            return None

        # Find the index of the "Name" and specified column in the header line
        header_line = lines[start_line-2]
        header_values = re.split(r" {2,}", header_line.strip())
        name_col_index = 0

        # Loop through rows and add to list of names.
        names = []
        # for i in range(start_line, len(lines)):
        i = start_line
        line_values = lines[i].strip().split()
        while line_values:
            names.append(line_values[name_col_index])
            i += 1
            line_values = lines[i].strip().split()

    return names


def get_components_by_tag(in_filepath, component_tag):
    with open(in_filepath, 'r') as file:
        # Read the file into a list of lines
        lines = file.readlines()

    # Find the line number where the section table starts
    start_line = None
    for i, line in enumerate(lines):
        if line.startswith('[TAGS]'):
            start_line = i + 1  # Skip the header lines
            break

    # If section is not found, return None.
    if start_line is None:
        print(f'No section found with name TAGS.')
        return None

    # Find the index of the "Name" and specified column in the header line
    name_col_index = 1

    # Loop through rows and add to list of names.
    names = []
    i = start_line
    line_values = lines[i].strip().split()
    while line_values:
        names.append(line_values[name_col_index])
        i += 1
        line_values = lines[i].strip().split()

    return names


def set_raingage(in_filepath, column_name, component_name, new_value, out_filepath=None):
    # If out_filepath is None, use in_filepath.
    if out_filepath is None:
        out_filepath = in_filepath

    with open(in_filepath, 'r') as file:
        # Read the file into a list of lines
        lines = file.readlines()

    # Find the start and end indices of the RAINGAGES table
    start_index = None
    end_index = None
    for i, line in enumerate(lines):
        if line.strip() == "[RAINGAGES]":
            start_index = i + 2
        elif start_index is not None and line.strip() == "":
            end_index = i
            break

    # Column indices.
    col_inds = {'Format': 1, 'Interval': 2, 'SCF': 3, 'Source': 4}

    # Search through the rain gages and find the line to update.
    update_line = None
    for i in range(start_index, end_index):
        if lines[i][0] == str(component_name):
            update_line = i

    if update_line is None:
        raise ValueError(f'Line with name "{component_name}" could not be found')

    # Create a new line.
    split_line = lines[update_line].split()
    new_line = []
    for i in range(5):
        if i == col_inds[column_name]:
            new_line.append(str(new_value))
        elif i == 4 and column_name != 'Source':
            # Handle source column when it doesn't need to be replaced.
            new_line.extend(split_line[-2:])
        elif i == 4 and column_name == 'Source':
            new_line.append(str(new_value))
        else:
            new_line.append(split_line[i])

    # Join the line.
    new_line = ' '.join(new_line) + '\n'

    # Update the line.
    del lines[update_line]
    lines[update_line:update_line] = new_line

    # Write the modified contents back to the file
    with open(out_filepath, "w") as file:
        file.writelines(lines)


def add_prcp_timeseries(in_filepath, ts_name, ts_description, times, values, dates=None, overwrite=False, out_filepath=None):

    # If out_filename is None, use in_filename.
    if out_filepath is None:
        out_filepath = in_filepath

    # Open the input file.
    with open(in_filepath, 'r') as f:
        lines = f.readlines()

    # Loop through lines.
    for i in range(len(lines)):

        # Find the correct section.
        if lines[i].strip().startswith('[TIMESERIES]'):
            # Line index of first row of timeseries table.
            table_start_idx = i + 3

            # Find last row of time series table.
            table_end_idx = i + 1
            while lines[table_end_idx].strip() != '':
                table_end_idx += 1

            # Get column start indices for Date, Time, and Value.
            column_split = [*lines[i + 1]]
            col_inds = [0] # Index for Name is 0.
            for c_name in ['Date', 'Time', 'Value']:
                ci, _ = string_index(lines[i+1], c_name)
                col_inds.append(ci)

            # Add description line.
            if overwrite is True:
                ts_lines = [] # New lines to add.
            else:
                ts_lines = [';\n']  # New lines to add.
            line_template = [' ' for _ in column_split] # Template line to populate.
            line_template[-1] = '\n'
            desc_line = ';' + ts_description
            desc_line = desc_line + ''.join([' ' for _ in range(len(column_split) - len(desc_line))]) + '\n'
            ts_lines.append(desc_line)
            for ti in range(len(times)): # No enumerate for readability.
                # Copy the line template.
                new_line = line_template.copy()

                # Add values for Name, Date, Time, and Value in correct positions.
                new_line[col_inds[0]:len(ts_name)+1] = ts_name # Timeseries name.
                if dates is not None:
                    new_line[col_inds[1]:len(str(dates[ti])) + 1] = [*str(dates[ti])] # Date.
                new_line[col_inds[2]:len(str(times[ti])) + 1] = [*str(times[ti])] # Time.
                new_line[col_inds[3]:len(str(values[ti])) + 1] = [*str(values[ti])] # Value

                # Join the line.
                new_line = ''.join(new_line)
                ts_lines.append(new_line)

            # Insert the new lines in the correct location within the file.
            if overwrite is True:
                # Remove the existing time series data.
                del lines[table_start_idx:table_end_idx]

                # Add in the new time series data.
                lines[table_start_idx:table_start_idx] = ts_lines

            elif overwrite is False:
                # Add in the new time series data.
                lines[table_end_idx:table_end_idx] = ts_lines

            else:
                raise ValueError('Overwrite not specified as True or False.')

            # Break out of looping through lines.
            break

    # write the modified contents back to the file
    with open(out_filepath, 'w') as f:
        f.writelines(lines)


def add_timeseries_file(in_filepath, ts_name, ts_description, file_path, overwrite=False, out_filepath=None):

    # If out_filename is None, use in_filename.
    if out_filepath is None:
        out_filepath = in_filepath

    # Open the input file.
    with open(in_filepath, 'r') as f:
        lines = f.readlines()

    # Loop through lines.
    for i in range(len(lines)):

        # Find the correct section.
        if lines[i].strip().startswith('[TIMESERIES]'):
            # Line index of first row of timeseries table.
            table_start_idx = i + 3

            # Find last row of time series table.
            table_end_idx = i + 1
            while lines[table_end_idx].strip() != '':
                table_end_idx += 1

            # Get column start indices for Date, Time, and Value.
            column_split = [*lines[i + 1]]
            col_inds = [0] # Index for Name is 0.
            for c_name in ['Date', 'Time', 'Value']:
                ci, _ = string_index(lines[i+1], c_name)
                col_inds.append(ci)

            # Add description line.
            if overwrite is True:
                ts_lines = [] # New lines to add.
            else:
                ts_lines = [';\n']  # New lines to add.
            line_template = [' ' for _ in column_split] # Template line to populate.
            line_template[-1] = '\n'
            desc_line = ';' + ts_description
            desc_line = desc_line + ''.join([' ' for _ in range(len(column_split) - len(desc_line))]) + '\n'
            ts_lines.append(desc_line)

            # Add new line with time series file.
            new_line = line_template.copy()
            new_line[col_inds[0]:len(ts_name)+1] = ts_name # Timeseries name.
            new_line[col_inds[1]:len('FILE') + 1] = ['FILE'] # File tag.
            new_line[col_inds[2]:len(str(file_path)) + 1] = f'"{file_path}"' # file path.

            # Join the line.
            new_line = ''.join(new_line)
            ts_lines.append(new_line)

            # Add a newline after the time series section.
            ts_lines.append('\n')

            # Insert the new lines in the correct location within the file.
            if overwrite is True:
                # Remove the existing time series data.
                del lines[table_start_idx:table_end_idx]

                # Add in the new time series data.
                lines[table_start_idx:table_start_idx] = ts_lines

            elif overwrite is False:
                # Add in the new time series data.
                lines[table_end_idx:table_end_idx] = ts_lines

            else:
                raise ValueError('Overwrite not specified as True or False.')

            # Break out of looping through lines.
            break

    # write the modified contents back to the file
    with open(out_filepath, 'w') as f:
        f.writelines(lines)


def string_index(full_string, match_string):
    """
    Find the start and end indices of a match_string in the full_string.
    string_index("The dog ran fast", "ran") -> (8, 10)
    Only returns indices from first match in string.
    :param full_string: Full string to search.
    :param match_string: String to match.
    :return: Start and end indices.
    """
    reg_search = re.search(match_string, full_string)
    start_idx, end_idx = reg_search.regs[0]
    return start_idx, end_idx


def parse_inp_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = defaultdict(list)
    section = None

    for line in lines:
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            section = line.strip('[]')
        elif section and line.startswith(';;') and not line.startswith(';;-'):
            data[section].append(line)
        elif section and line and not line.startswith(';'):
            data[section].append(line)

    return data


def parse_section_to_df(lines):

    header_names = {
        'TITLE': ['Project Title/Notes'],
        'OPTIONS': ['Option', 'Value'],
        'EVAPORATION': ['Data Source', 'Parameters'],
        'RAINGAGES': ['Name', 'Format', 'Interval', 'SCF', 'Source', 'Filename', 'Units'],
        'SUBCATCHMENTS': ['Name', 'Rain Gage', 'Outlet', 'Area', '%Imperv', 'Width', '%Slope', 'CurbLen', 'SnowPack'],
        'SUBAREAS': ['Subcatchment', 'N-Imperv', 'N-Perv', 'S-Imperv', 'S-Perv', 'PctZero', 'RouteTo', 'PctRouted']
    }
    
    # Find the header (first line starting with ";;")
    headers = []
    data_rows = []
    for line in lines:
        if line.startswith(';;'):
            headers = re.split(r'\s{2,}', line[2:].strip())
        else:
            row = re.split(r'\s{1,}', line.strip())
            # If first element of row starts with ";", skip since it is a comment.
            if row[0][0] == ";":
                continue
            data_rows.append(row)

    
    if data_rows:
        for row in data_rows:
            # Difference between length of row and headers.
            diff = len(row) - len(headers)

            # If there are fewer headers than data columns, pad headers.
            if diff > 0:
                [headers.append(f'col_{i}') for i in range(len(headers), len(headers) + diff)]
            
            # If there are fewer data columns than headers, pad rows.
            if diff < 0:
                [row.append('0') for _ in range(len(row), len(row) - diff)]

    # Fall back if no headers found
    if not headers:
        headers = [f'col_{i}' for i in range(len(data_rows[0]))]

    # Make data frame.
    df = pd.DataFrame(data_rows, columns=headers)

    return df


def inp_to_database(filepath):
    raw_sections = parse_inp_file(filepath)
    database = {}
    for section, lines in raw_sections.items():
        try:
            df = parse_section_to_df(lines)
            database[section] = df
        except Exception as e:
            print(f"Error parsing section [{section}]: {e}")
    return database



if __name__ == '__main__':
    in_filepath = r"C:\Users\ay434\Box\Research\Digital_Twin_Interpretable_AI\SWMM\SWMM_Files\SWMM\Canandaigua.inp"
    out_filepath = r"C:\Users\ay434\Box\Research\Digital_Twin_Interpretable_AI\SWMM\SWMM_Files\SWMM\Canandaigua - Copy.inp"
    # section = 'SUBCATCHMENTS'
    # column_name = 'Area'
    # component_name = 2
    # new_value = 200
    #
    # set_inp_section(in_filepath, section, column_name, component_name, new_value, out_filepath=out_filepath)

    # section = 'RAINGAGES'
    # rg_ts_name = 'TIMESERIES TS_TEST'
    # component_name = 1
    # column_name = 'Source'
    # set_raingage(in_filepath, column_name, component_name, rg_ts_name, out_filepath=out_filepath)

    # ts_name = 'TS_TEST'
    # ts_description = 'TS_TEST_DESC'
    # times = [0, 1, 2, 3, 4, 5, 6]
    # values = [0, 1, 0.8, 0.6, 0.4, 0.2, 0.0]
    # add_prcp_timeseries(in_filepath, ts_name, ts_description, times, values, dates=None, overwrite=True,
    #                     out_filepath=out_filepath)

    inp_path = r'C:\Users\ay434\Documents\urbansurge\analysis\Bellinge\7_SWMM\BellingeSWMM_v021_nopervious_tmp.inp'
    db = inp_to_database(inp_path)

    print(db.keys())

    # Example: access subcatchments
    print(db['RAINGAGES'].head())