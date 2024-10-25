import numpy as np
from matplotlib.cm import get_cmap
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import math


def plot_contour(ax, func, x_bounds=(-5, 5), y_bounds=(-5, 5), mesh_density=200,
                 level_sets=[0], colors='r', funcs_are_torch=False, break_in_batch=0,
                 linestyle='dashed', filled=False):
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
    if not filled:
        contour = ax.contour(X, Y, Z, levels=level_sets, colors=colors, linestyles=linestyle)
    else:
        _ = ax.contourf(X, Y, Z, levels=[*level_sets, 10000], colors=colors, extend='neither', alpha=0.05)
        contour = ax.contour(X, Y, Z, levels=level_sets, colors=colors, linestyles=linestyle)
    return contour


def plot_zero_level_sets(functions, bounds=(-5, 5), mesh_density=200,
                         x_label=r'$x$', y_label=r'$y$', legends=None,
                         cmap='tab20', font_size=12, funcs_are_torch=False,
                         break_in_batch=0, plt_show=True, legend_dict=None,
                         linestyles=None, x_lim=None, y_lim=None, filled=None):
    # if legends is None:
    # legends = [f.__name__ for f in functions]

    # Choose a colormap for different colors
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
        colors = [cmap(i) for i in np.linspace(0, 1, len(functions))]
    else:
        colors = cmap

    # Create a figure and axes
    fig, ax = plt.subplots()
    if linestyles is None:
        linestyles = ['dashed'] * len(functions)

    if not isinstance(filled, list):
        filled = [filled] * len(functions)

    contours = [plot_contour(ax, func, x_bounds=bounds,
                             y_bounds=bounds, mesh_density=mesh_density,
                             level_sets=[0], colors=[color],
                             funcs_are_torch=funcs_are_torch,
                             break_in_batch=break_in_batch,
                             linestyle=linestyle, filled=fill)
                for func, color, linestyle, fill in zip(functions, colors, linestyles, filled)]

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


def from_mcolors_to_rgb_pallete(mcolor_pallete_name, mcolor_pallete_colors=None):
    pallete = getattr(mcolors, mcolor_pallete_name + '_COLORS')
    if mcolor_pallete_colors is not None:
        return [(*(mcolors.to_rgb(pallete[color])), 1.0) for color in mcolor_pallete_colors]

    return [(*(mcolors.to_rgb(color)), 1.0) for color in pallete.values()]


# visualize mcolors pallete. From matplotlib https://matplotlib.org/stable/gallery/color/named_colors.html
def plot_colortable(colors, *, ncols=4, sort_colors=True):
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - margin) / height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y - 9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )
    plt.show()
