import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from wavesolve.fe_solver import get_eff_index, plot_vector_mode

def create_distinct_colormap(n_needed=60):
    """Create colormap using tab20 and Set3 for maximum distinction"""
    # Get colors from tab20 (20 colors) and Set3 (12 colors)
    tab20b_colors = plt.cm.tab20b(np.linspace(0, 1, 20))
    tab20c_colors = plt.cm.tab20c(np.linspace(0, 1, 20))

    # Combine and repeat as needed
    base_colors = np.vstack([tab20b_colors, tab20c_colors])  # 40 colors

    # Repeat colors to get n_needed (0-n_needed)
    colors = []
    for i in range(n_needed):
        colors.append(base_colors[i % len(base_colors)])

    return ListedColormap(colors)


def plot_modes(w, v, m, wl, IOR_dict, target_radius=None,
               basefigsize=2, no_cols=6, circle_color='white',
               circle_linewidth=1, circle_linestyle='dashed'):
    """
    Plot vector modes in subplots with circles overlay.

    Parameters:
    w (array): Eigenvalues
    v (array): Eigenvectors
    m: Model/mesh object
    wl (float): Wavelength
    IOR_dict (dict): Index of refraction dictionary
    target_radius (float): Radius of the circle to plot (if None, extracted from fiber bundle)
    basefigsize (float): Base size for subplot dimensions (default: 2)
    no_cols (int): Number of columns in subplot grid (default: 6)
    circle_color (str): Color of the circle (default: 'white')
    circle_linewidth (float): Line width of the circle (default: 1)
    circle_linestyle (str): Line style of the circle (default: 'dashed')
    """
    # Calculate effective indices and IOR bounds internally
    eff_indexs = get_eff_index(wl, w)
    IORs = [ior[1] for ior in IOR_dict.items()]
    nmin, nmax = min(IORs), max(IORs)

    # Extract target_radius from fiber bundle if not provided
    if target_radius is None and hasattr(m, 'r_target_mmcore_size'):
        target_radius = m.r_target_mmcore_size
    # Calculate number of modes to plot
    n_modes = len([w_val for w_val, ne in zip(w, eff_indexs)
                   if w_val >= 0 and nmin <= ne <= nmax])

    # Create subplot grid
    n_rows = round(n_modes / no_cols)
    fig, axs = plt.subplots(n_rows, no_cols, sharey=True,
                            figsize=(basefigsize * no_cols, basefigsize * n_rows))

    # Handle case where there's only one row
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    # Create circle coordinates
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = target_radius * np.cos(theta)
    circle_y = target_radius * np.sin(theta)

    plot_index = 0
    for _w, _v, ne in zip(w, v, eff_indexs):
        if _w < 0:
            continue

        if not (nmin <= ne <= nmax):
            print("warning: spurious mode! stopping plotting ... ")
            break

        # Calculate row and column for current plot
        row = plot_index // no_cols
        col = plot_index % no_cols
        ax = axs[row, col]

        ax.set_aspect('equal')
        plot_vector_mode(m, _v, ax=ax, arrows=False)
        ax.plot(circle_x, circle_y, color=circle_color,
                linewidth=circle_linewidth, linestyle=circle_linestyle)

        # Optional: plot scalar mode
        # plot_scalar_mode(m, np.real(_v[Ne:]), ax=ax)

        plot_index += 1

    # Hide unused subplots
    for i in range(plot_index, n_rows * no_cols):
        row = i // no_cols
        col = i % no_cols
        axs[row, col].set_visible(False)

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()
