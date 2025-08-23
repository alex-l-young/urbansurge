####################################################################
# Plotting functions for UrbanSurge
####################################################################

# Library imports.
import matplotlib.pyplot as plt
import numpy as np

from urbansurge import sensor_network

def linear_spaced_array(value1, value2, N=5, D=1):
    """
    Generate a linearly spaced array of N numbers between two values with numbers 
    rounded to D decimal places.

    Parameters:
        value1 (float): The starting value of the array.
        value2 (float): The ending value of the array.
        N (int): Number of values in the array.
        D (int): Maximum number of decimal places.

    Returns:
        list: A list of N linearly spaced numbers rounded to D decimal places.
    """
    # Generate a linearly spaced array with N numbers
    linear_array = np.linspace(value1, value2, N)

    # Round the numbers to the specified number of decimal places
    rounded_array = np.round(linear_array, D)

    # Ensure unique values by rounding edge cases
    rounded_array = np.unique(rounded_array)

    # Check if we achieved the desired N numbers; if not, regenerate
    while len(rounded_array) < N:
        step = (value2 - value1) / (N - 1)
        linear_array = np.array([value1 + i * step for i in range(N)])
        rounded_array = np.round(linear_array, D)
        rounded_array = np.unique(rounded_array)

    return rounded_array.tolist()


def plot_network(swmm, node_weights=None, figsize=(6,5)):
    """
    Plots the network.
    """
    # Inp filepath. 
    inp_filepath = swmm.inp_path

    # Link-nodes dictionary.
    link_nodes_dict = sensor_network.link_nodes(inp_filepath)

    # Link-coordinates dictionary.
    fig, ax = plt.subplots(figsize=figsize)
    for _, nodes in link_nodes_dict.items():
        from_node_coords = swmm.get_node_coordinates(nodes[0])
        to_node_coords = swmm.get_node_coordinates(nodes[1])
        coords = np.stack((from_node_coords, to_node_coords))
        ax.plot(coords[:,0], coords[:,1], 'k', linewidth=1)
        ax.scatter(coords[:,0], coords[:,1], c='k')

    
    # for i in range(0, coords.shape[0], 2):
    #     ax.plot(coords[i:i+2,0], coords[i:i+2,1], 'k')


def link_nodes_to_shp(swmm, link_nodes_dict):
    for _, nodes in link_nodes_dict.items():
        from_node_coords = swmm.get_node_coordinates(nodes[0])
        to_node_coords = swmm.get_node_coordinates(nodes[1])
        coords = np.stack((from_node_coords, to_node_coords))
    
    return nodes_shp, links_shp

if __name__ == '__main__':
    # Example usage
    result = linear_spaced_array(0.11, 0.36, N=4, D=1)
    print(result)  # Output: [0.1, 0.2, 0.3, 0.4]
