from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize
import plotly.graph_objects as go


def dict_to_namedtuple(dict):
    nt = namedtuple('nt', dict)
    tuple = nt(**dict)
    return tuple

def plot_gng_stats(stats, rolling_window=10, smoothed=False, colors=cm.Dark2(np.linspace(0, 1, 4, endpoint=False)), def_ax=None):
    if not def_ax:
        fig, ax = plt.subplots(nrows=3, figsize=(10, 9))
        ax = ax.flatten()
    else:
        ax = def_ax

    # Global error
    df = pd.Series(stats.global_error)
    if smoothed:
        ma = df.rolling(rolling_window).mean()
        mstd = df.rolling(rolling_window).std()

        lower_bound = ma-mstd
        upper_bound = ma+mstd
        ax[0].plot(ma.index, ma, color=colors[0], label="Global error")
        ax[0].fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15, color=colors[0])
    else:
        ax[0].plot(df, color=colors[0], label="Global error")

    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Global error")
    ax[0].set_title("Global error")

    # Global mean error
    df = pd.Series(np.array(stats.global_error)/np.array(stats.graph_order))
    if smoothed:
        ma = df.rolling(rolling_window).mean()
        mstd = df.rolling(rolling_window).std()

        lower_bound = ma-mstd
        upper_bound = ma+mstd
        ax[0].plot(ma.index, ma, color=colors[1], label="Global mean error")
        ax[0].fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15, color=colors[1])
    else:
        ax[0].plot(df, color=colors[1], label="Global mean error")
    
    ax[0].legend()

    # Global utility
    df = pd.Series(stats.global_utility)
    if smoothed:
        ma = df.rolling(rolling_window).mean()
        mstd = df.rolling(rolling_window).std()

        lower_bound = ma-mstd
        upper_bound = ma+mstd
        ax[1].plot(ma.index, ma, color=colors[1])
        ax[1].fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15, color=colors[1])
    else:
        ax[1].plot(df, color=colors[1])
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Global utility")
    ax[1].set_title("Global utility")

    # Graph size and order
    df = pd.Series(stats.graph_order)
    ma = df.rolling(rolling_window).mean()
    mstd = df.rolling(rolling_window).std()

    lower_bound = ma-mstd
    upper_bound = ma+mstd

    ax[2].plot(ma.index, ma, color=colors[2], label="Graph order, |V|")
    ax[2].fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15, color=colors[2])
    df = pd.Series(stats.graph_size)
    ma = df.rolling(rolling_window).mean()
    mstd = df.rolling(rolling_window).std()

    lower_bound = ma-mstd
    upper_bound = ma+mstd

    ax[2].plot(ma.index, ma, color=colors[3], label="Graph size, |E|")
    ax[2].fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15, color=colors[3])

    ax[2].set_xlabel("Iterations")
    ax[2].set_title("Graph properties")
    ax[2].legend()
    if not def_ax:
        fig.tight_layout()


def plot_q_table_3d(q_table, state_size, title="Q-table", axis_names=["X", "Y", "Z"]):
    x_range = np.arange(0, state_size[0])
    y_range = np.arange(0, state_size[1])
    X, Y = np.meshgrid(x_range, y_range)
    Z = np.max(q_table, axis=2)
    norm = Normalize(vmin = np.min(Z), vmax = np.max(Z), clip = False)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.CMRmap, norm=norm, rstride=1, cstride=1, vmin=-1.0, vmax=1.0)
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_zlabel(axis_names[2])
    ax.set_title(title)
    ax.view_init(ax.elev, -120)
    fig.colorbar(surf)
    fig.tight_layout()
    plt.show()

def plot_q_table(q_table, 
                action_names, 
                symbols, 
                colors, 
                title="Q-table", 
                cmap="CMRmap", 
                axis_names=["X", "Y"], 
                def_plot=None,
                figsize=(10, 10)):

    if not def_plot:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = def_plot

    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    q_actions = np.argmax(q_table, axis=2)  # best action for each state

    cax = ax.imshow(q_image, cmap=cmap)
    cbar = fig.colorbar(cax)

    for action in np.unique(q_actions):
        x, y = np.where(q_actions==action)
        ax.scatter(x, y, label='Action: {}'.format(action_names[action]), marker=symbols[action], s=10**2, c=[colors[action]])
            
    ax.grid(False)
    ax.set_title("{} \n size: {}".format(title, q_table.shape))
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.legend(bbox_to_anchor=(0, -0.125,1,0.2), loc="lower left",mode="expand", borderaxespad=0, ncol=3)


def plot_stats(stats, 
                rolling_window=10, 
                colors=None,
                def_plot=None,
                figsize=(10, 6),
                label=None,
                ):

    if def_plot is None:
        fig, ax = plt.subplots(nrows=len(stats), figsize=figsize)
    else:
        fig, ax = def_plot

    if colors is None:
        colors = cm.Dark2(np.linspace(0, 1, len(stats), endpoint=False))

    for i, (key, val) in enumerate(stats.items()):
        df = pd.Series(val)
        ma = df.rolling(rolling_window).mean()
        mstd = df.rolling(rolling_window).std()
        lower_bound = ma-mstd
        upper_bound = ma+mstd

        ax[i].plot(ma.index, ma, color=colors[i], label=label)
        ax[i].fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15, color=colors[i])
        ax[i].set_xlabel("episode")
        ax[i].set_ylabel(key.replace("_", " "))

    fig.tight_layout()

def plot_stats_comparison(stats_dict,
                            rolling_window=10,
                            def_plot=None,
                            figsize=(10, 6),
                            colors=None,
                            title=None):


    if def_plot is None:
        fig, ax = plt.subplots(nrows=len(stats_dict), figsize=figsize)
    else:
        fig, ax = def_plot

    if colors is None:
        colors = cm.tab20b(np.linspace(0, 1, len(stats_dict), endpoint=False))
    
    for i, (model_name, stats) in enumerate(stats_dict.items()):
        _colors = np.tile(colors[i], (len(stats), 1))
        plot_stats(stats, def_plot=(fig, ax), label=model_name, colors=_colors)

    for axe in ax:
        axe.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()


def visualize_samples(samples, discretized_samples, grid, low=None, high=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot discretization as a grid
    ax.xaxis.set_major_locator(plt.FixedLocator(grid[0]))
    ax.yaxis.set_major_locator(plt.FixedLocator(grid[1]))
    ax.grid(True)

    if low is None or high is None:
        x, y = grid[0], grid[1]
        low = [x[0], y[0]]
        high = [x[-1], y[-1]]
    else:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])

    # Map each discretized sample (which is really an index) to the center of corresponding grid cell
    grid_extended = np.hstack((np.array([low]).T, grid, np.array([high]).T)) 
    grid_centers = (grid_extended[:, 1:] + grid_extended[:, :-1]) / 2 
    x = discretized_samples[:, 0]
    y = discretized_samples[:, 1]

    gx = grid_centers[0, x] # pick grid centre for each x
    gy = grid_centers[1, y] # pick grid centre for each y
    locs = np.stack([gx, gy], axis=1)

    ax.plot(samples[:, 0], samples[:, 1], 'o', label='Original')
    ax.plot(locs[:, 0], locs[:, 1], 's', label='Discretized')
    ax.legend()
    ax.add_collection(LineCollection(list(zip(samples, locs)), colors='orange'))

def project_nodes(nodes, 
                    state_size, 
                    action_names, 
                    symbols, 
                    colors, 
                    title=None, 
                    unravel=False, 
                    round=False, 
                    axis_names=["X", "Y"], 
                    def_plot=None,
                    figsize=(10, 10),
                    legend=False,
                    labels=False):
    if not def_plot:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = def_plot

    if unravel:
        nodes = np.unravel_index(np.around(nodes).astype(int), state_size)
    elif round:
        nodes = np.around(nodes).astype(int)

    layered_nodes = [nodes[:, nodes[-1]==i] for i in range(len(action_names))]
    for i, nodes in enumerate(layered_nodes):
        x, y, _ = nodes
        label = None
        if labels:
            label = 'Action: {}'.format(action_names[i])
        ax.scatter(x, y, label=label, marker=symbols[i], c=[colors[i]], s=10**2)

    if legend:
        ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=3)
        ax.set_xlabel(axis_names[0], loc='left')
    else: 
        ax.set_xlabel(axis_names[0])

    ax.grid(True)
    ax.set_ylabel(axis_names[1])

    ax.set_xlim(-0.5, state_size[0]-0.5)
    ax.set_ylim(-0.5, state_size[1]-0.5)
    
    ax.set_xticks(np.arange(state_size[0]))
    ax.set_yticks(np.arange(state_size[1]))
    if title:
        ax.set_title(title)
    
    if not def_plot:
        fig.tight_layout()


def plot_nodes_3d(nodes, 
                state_size, 
                action_names, 
                symbols, 
                colors, 
                title="ML-GNG layers", 
                unravel=False, 
                round=False, 
                axis_names=["X", "Y", "Z"],
                def_plot=None,
                figsize=(10, 10),
                legend=False):
    
    if not def_plot:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = def_plot

    ax = fig.add_axes([0,0,1,1], projection='3d')

    if unravel:
        # We're unraveling because the state_space is 1D and we want to make it 2D (eg gridworld)
        nodes = np.unravel_index(np.around(nodes).astype(int), state_size)
    elif round:
        nodes = np.around(nodes).astype(int)
    
    gx = np.arange(state_size[0]+1)
    gy = np.arange(state_size[1]+1)

    layered_nodes = [nodes[:, nodes[-1]==i] for i in range(len(action_names))]
    for i, nodes in enumerate(layered_nodes):
        x, y, _ = nodes
        ax.scatter(x, y, i, marker=symbols[i], s=100, c=[colors[i]], label="Action: {}".format(action_names[i]))
        
        p = Rectangle((0,0), state_size[0], state_size[1], color=colors[i], alpha=0.15)
        # grid lines
        for x in gx:
            ax.plot3D([x, x], [gy[0], gy[-1]], i, color=colors[i], alpha=.7, linestyle=':')
        for y in gy:
            ax.plot3D([gx[0], gx[-1]], [y, y], i, color=colors[i], alpha=.7, linestyle=':')

        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=i, zdir="z")

    # plot connection lines
    # ax.plot([x[0],y[0],x[0]],[y[0],x[0],y[0]],[0.4,0.9,1.6], color="k")
    # ax.plot([x[2],y[2],x[2]],[y[2],x[2],y[2]],[0.4,0.9,1.6], color="k")
    
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_zlabel(axis_names[2])

    ax.set_aspect('auto')

    # ax.set_xticks(np.arange(state_size[1]))
    # ax.set_yticks(np.arange(state_size[0]))
    ax.set_zticks(np.arange(len(action_names)))

    ax.set_xlim(0, state_size[0])
    ax.set_ylim(0, state_size[1])
    # ax.set_xticks(np.arange(state_size[0]))
    # ax.set_yticks(np.arange(state_size[1]))
    
    ax.grid(False)
    #ax.view_init(elev=15., azim=60)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_nodes_changes(stats_series, action_names, symbols, colors, figsize=(600, 600)):

    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    for idx, nodes in enumerate(stats_series):
        if idx % 10 == 0:
            for i in range(len(action_names)):
                x, y, _ = nodes[:, nodes[-1]==i]
                fig.add_trace(
                    go.Scatter(
                        visible=False,
                        mode="markers",
                        name=action_names[i],
                        marker_symbol=symbols[i],
                        marker=dict(size=12,
                                    color="rgba(" + ",".join([str(int(item)) for item in colors[i]]) + ")"),
                        x=x,
                        y=y)
                        )

    fig.data[0].visible = True
    fig.data[1].visible = True
    fig.data[2].visible = True

    # Create and add slider
    steps = []
    i = 0
    for idx in range(len(fig.data)//3):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Episode: " + str(idx*10)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][i+2] = True  # Toggle i'th trace to "visible"
        steps.append(step)
        i+=3

    sliders = [dict(
        active=0,
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        sliders=sliders,
        yaxis=dict( range=[-0.5, 9.5],
                    tickmode="linear"),
        xaxis=dict( range=[-0.5, 9.5],
                    tickmode="linear"),
    )

    fig.show()