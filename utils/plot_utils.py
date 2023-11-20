import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_contour(ax, func, x_bounds=(-5, 5), y_bounds=(-5, 5), mesh_density=400,
                 level_sets=[0], colors='r'):
    # Create a grid of x and y values
    x = np.linspace(x_bounds[0], x_bounds[1], mesh_density)
    y = np.linspace(y_bounds[0], y_bounds[1], mesh_density)

    # Create a meshgrid from x and y
    X, Y = np.meshgrid(x, y)

    # Evaluate the function on the meshgrid
    Z = func(X, Y)

    # Plot the contour of the function for specified level sets
    contour = ax.contour(X, Y, Z, levels=level_sets, colors=colors)

def plot_zero_level_sets(functions, bounds=(-5, 5), mesh_density=400,
                         x_label=r'$x$', y_label=r'$y$', legends=None,
                         cmap='tab10', font_size=12):
    if legends is None:
        legends = [f.__name__ for f in functions]

    # Choose a colormap for different colors
    cmap = get_cmap(cmap)

    # Create a figure and axes
    fig, ax = plt.subplots()

    colors = [cmap(i) for i in np.linspace(0, 1, len(functions))]

    for func, color, legend in zip(functions, colors, legends):
        plot_contour(ax, func, x_bounds=bounds, y_bounds=bounds, mesh_density=mesh_density,
                     level_sets=[0], colors=[color])

    # Add labels and title
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)

    # Set equal scaling for x and y axes
    ax.set_aspect('equal', adjustable='box')

    # Set font size for tick labels
    ax.tick_params(axis='both', which='both', labelsize=font_size)

    # Place the legend
    ax.legend(legends, fontsize=font_size)

    plt.show()




# Define two different functions
def function1(x, y):
    return np.sin(x) + np.cos(y)

def function2(x, y):
    return x**2 + y**2 - 4


# Example usage:
plot_zero_level_sets([function1, function2], bounds=(-2, 2), mesh_density=400,
                     x_label=r'$x$', y_label=r'$y$', legends=['Function 1', 'Function 2'],
                     font_size=14)