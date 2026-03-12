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


def plot_swmm_network(swmm):
    """
    Plots the swmm network nodes and conduits.

    :param swmm: swmm_model.SWMM object.
    """
    # Get input path.
    inp_path = swmm.inp_path

    nodes = {}
    conduits = {}

    # Parse the .inp file
    with open(inp_path, 'r') as file:
        lines = file.readlines()

    current_section = None

    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue

        # Track current section
        if line.startswith('['):
            current_section = line
            continue

        # Skip comment lines
        if line.startswith(';'):
            continue

        # Extract Node Coordinates
        if current_section == '[COORDINATES]':
            parts = line.split()
            if len(parts) >= 3:
                node_id = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                nodes[node_id] = (x, y)

        # Extract Conduit Connections
        elif current_section == '[CONDUITS]':
            parts = line.split()
            if len(parts) >= 3:
                conduit_id = parts[0]
                from_node = parts[1]
                to_node = parts[2]
                conduits[conduit_id] = (from_node, to_node)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot Conduits
    for cid, (from_n, to_n) in conduits.items():
        if from_n in nodes and to_n in nodes:
            x1, y1 = nodes[from_n]
            x2, y2 = nodes[to_n]

            # Draw line
            ax.plot([x1, x2], [y1, y2], color='royalblue', linewidth=2, zorder=1)

            # Label conduit in the middle
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, cid, color='darkblue', fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

    # Plot Nodes
    for nid, (x, y) in nodes.items():
        # Draw node point
        ax.scatter(x, y, color='firebrick', s=40, zorder=2)

        # Label node slightly offset from the point
        ax.annotate(nid, (x, y), xytext=(4, 4), textcoords='offset points',
                    color='darkred', fontsize=9, fontweight='bold')

    # Formatting the plot
    ax.set_aspect('equal')  # Ensures the network isn't warped
    plt.title('SWMM Network Map')
    plt.xlabel('X-Coordinate')
    plt.ylabel('Y-Coordinate')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Display the plot
    plt.show()


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
