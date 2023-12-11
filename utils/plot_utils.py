import numpy as np
from matplotlib.cm import get_cmap
import torch
import matplotlib.pyplot as plt

def plot_contour(ax, func, x_bounds=(-5, 5), y_bounds=(-5, 5), mesh_density=200,
                 level_sets=[0], colors='r', funcs_are_torch=False, break_in_batch=0,
                 linestyle='dashed'):

    if funcs_are_torch:
        # Create a grid of x and y values
        x = torch.linspace(x_bounds[0], x_bounds[1], mesh_density)
        y = torch.linspace(y_bounds[0], y_bounds[1], mesh_density)

        # Create a meshgrid from x and y
        X, Y = torch.meshgrid(x, y)
        X_flat = torch.flatten(X)
        Y_flat = torch.flatten(Y)
        input = torch.vstack((X_flat, Y_flat)).t()
    else:
        # Create a grid of x and y values
        x = np.linspace(x_bounds[0], x_bounds[1], mesh_density)
        y = np.linspace(y_bounds[0], y_bounds[1], mesh_density)

        # Create a meshgrid from x and y
        X, Y = np.meshgrid(x, y)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        input = np.vstack((X_flat, Y_flat)).T
    # Evaluate the function on the meshgrid
    if break_in_batch == 0:
        Z = func(input)
    else:
        if funcs_are_torch:
            Z = [func(inp) for inp in input.split(break_in_batch)]
            Z = torch.hstack(Z)
        else:
            Z = [func(inp) for inp in np.array_split(input, break_in_batch)]
            Z = np.hstack(Z)

    if funcs_are_torch:
        X = X.detach().numpy()
        Y = Y.detach().numpy()
        Z = Z.view(mesh_density, mesh_density).detach().numpy()
    else:
        Z = Z.reshape(mesh_density, mesh_density)

    # Plot the contour of the function for specified level sets
    contour = ax.contour(X, Y, Z, levels=level_sets, colors=colors, linestyles=linestyle)
    return contour

def plot_zero_level_sets(functions, bounds=(-5, 5), mesh_density=200,
                         x_label=r'$x$', y_label=r'$y$', legends=None,
                         cmap='tab20', font_size=12, funcs_are_torch=False,
                         break_in_batch=0, plt_show=True, legend_dict=None,
                         linestyles=None, x_lim=None, y_lim=None):
    # if legends is None:
        # legends = [f.__name__ for f in functions]

    # Choose a colormap for different colors
    cmap = get_cmap(cmap)

    # Create a figure and axes
    fig, ax = plt.subplots()
    if linestyles is None:
        linestyles = ['dashed'] * len(functions)

    colors = [cmap(i) for i in np.linspace(0, 1, len(functions))]
    contours = [plot_contour(ax, func, x_bounds=bounds,
                             y_bounds=bounds, mesh_density=mesh_density,
                             level_sets=[0], colors=[color],
                             funcs_are_torch=funcs_are_torch,
                             break_in_batch=break_in_batch,
                             linestyle=linestyle)
                for func, color, linestyle in zip(functions, colors, linestyles)]



    # Add labels and title
    ax.set_xlabel(x_label, fontsize=font_size)
    ax.set_ylabel(y_label, fontsize=font_size)
    if x_lim is not None:
        ax.set_xlim(*x_lim)

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    # Set equal scaling for x and y axes
    ax.set_aspect('equal', adjustable='box')

    # Set font size for tick labels
    ax.tick_params(axis='both', which='both', labelsize=font_size)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Place the legend

    if legends:
        for contour, legend in zip(contours, legends):
            contour.collections[0].set_label(legend)
        if legend_dict is None:
            ax.legend()
        else:
            ax.legend(**legend_dict)

    if plt_show:
        plt.show()
    else:
        return fig, ax




# # Define two different functions
# def function1(x, y):
#     return np.sin(x) + np.cos(y)
#
# def function2(x, y):
#     return x**2 + y**2 - 4
#
#
# # Example usage:
# plot_zero_level_sets([function1, function2], bounds=(-2, 2), mesh_density=400,
#                      x_label=r'$x$', y_label=r'$y$', legends=['Function 1', 'Function 2'],
#                      font_size=14)