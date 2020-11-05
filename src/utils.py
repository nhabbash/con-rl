from collections import namedtuple, defaultdict
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import mpl_toolkits.mplot3d.art3d as art3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle, PathPatch
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns


EpisodeStats = namedtuple("Stats", ["episode_lengths", 
                                    "episode_rewards",
                                    "selector_dist"])

def project_best_actions(state_actions, action_names, symbols, colors, axis_names=["X", "Y"], title="Best actions", ax=None, legend=True):

    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))

    for k, v in state_actions.items():
        x, y = zip(*v)
        ax.scatter(x, y, label='Action: {}'.format(action_names[k]), marker=symbols[k], color=colors[k], s=10**2)
        if legend:
            plt.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=3)
            ax.set_xlabel(axis_names[0], loc='left')
        else: 
            ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
        ax.set_title(title)

    if not ax:
        fig.tight_layout()

def project_mlgng_actions(mlgng, state_size, action_names, symbols, colors, title="ML-GNG layers projection", unravel=False, round=False, axis_names=["X", "Y"], ax=None, legend=True):
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 10))
    # Plot discretization as a grid
    ax.grid(True)
    for i in range(len(action_names)):
        if unravel:
            x, y = np.unravel_index(np.around(mlgng[i].g.vp.pos.get_2d_array(pos=[0])[0]).astype(int), state_size)
        else:
            if round:
                x, y = np.around(mlgng[i].g.vp.pos.get_2d_array(pos=[0, 1])).astype(int)
            else:
                x, y = mlgng[i].g.vp.pos.get_2d_array(pos=[0, 1])
        ax.scatter(x, y, label='Action: {}'.format(action_names[i]), marker=symbols[i], c=[colors[i]], s=10**2)

    if legend:
        ax.legend(bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=3)
        ax.set_xlabel(axis_names[0], loc='left')
    else: 
        ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_xlim(0, state_size[0])
    ax.set_ylim(0, state_size[1])
    ax.set_xticks(np.arange(state_size[1]+1))
    ax.set_yticks(np.arange(state_size[0]+1))
    ax.set_title(title)
    
    if not ax:
        fig.tight_layout()

def plot_mlgng_actions_3d(mlgng, state_size, action_names, symbols, colors, title="ML-GNG layers", unravel=False, round=False, axis_names=["X", "Y", "Z"], full_projection=False):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0,0,1,1], projection='3d')
    
    gx = np.arange(state_size[0]+1)
    gy = np.arange(state_size[1]+1)

    for i in range(len(action_names)):
        if unravel:
            # We're unraveling because the state_space is 1D and we want to make it 2D (eg gridworld)
            x, y = np.unravel_index(np.around(mlgng[i].g.vp.pos.get_2d_array(pos=[0])[0]).astype(int), state_size)
        else:
            if round:
                x, y = np.around(mlgng[i].g.vp.pos.get_2d_array(pos=[0, 1])).astype(int)
            else:
                x, y = mlgng[i].g.vp.pos.get_2d_array(pos=[0, 1])

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
    
    ax.grid(False)
    #ax.view_init(elev=15., azim=60)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_stats_comparison(stats_dict, rolling_window=10):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))

    for name, stats in stats_dict.items():

        df = pd.Series(stats.episode_lengths)
        ma = df.rolling(rolling_window).mean()
        mstd = df.rolling(rolling_window).std()
        lower_bound = ma-mstd
        upper_bound = ma+mstd

        ax1.plot(ma.index, ma, label=name)
        ax1.fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15)
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Steps")
        ax1.set_title("Episode Length over Time")
        ax1.legend()

        df = pd.Series(stats.episode_rewards)
        ma = df.rolling(rolling_window).mean()
        mstd = df.rolling(rolling_window).std()

        lower_bound = ma-mstd
        upper_bound = ma+mstd

        ax2.plot(ma.index, ma, label=name)
        ax2.fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Episode Reward")
        ax2.set_title("Episode Reward over Time")
        ax2.legend()
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

def plot_q_table(q_table, action_names, symbols, colors, title="Q-table", cmap="CMRmap", axis_names=["X", "Y"]):
    """Visualize max Q-value for each state and corresponding action."""
    q_image = np.max(q_table, axis=2)       # max Q-value for each state
    q_actions = np.argmax(q_table, axis=2)  # best action for each state

    fig, ax = plt.subplots(figsize=(10, 10))
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

def plot_stats(stats, rolling_window=10):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(10, 6))

    # ax1.set_prop_cycle('color', [plt.cm.coolwarm(i) for i in np.linspace(0, 1, 2)])
    # ax2.set_prop_cycle('color', [plt.cm.coolwarm(i) for i in np.linspace(0, 1, 2)])

    ax1.plot(stats.episode_lengths)
    rm = pd.Series(stats.episode_lengths).rolling(rolling_window).mean()
    ax1.plot(rm)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Steps")
    ax1.set_title("Episode Length over Time")

    ax2.plot(stats.episode_rewards)
    rm = pd.Series(stats.episode_rewards).rolling(rolling_window).mean()
    ax2.plot(rm)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Episode Reward")
    ax2.set_title("Episode Reward over Time")
    fig.tight_layout()

def plot_stats2(stats, rolling_window=10, smoothed=True, selector=False, sel_norm=False, colors=cm.Dark2(np.linspace(0, 1, 3, endpoint=False))):
    
    if selector:
        fig, ax = plt.subplots(nrows=3, figsize=(10, 8))
    else:
        
        fig, ax = plt.subplots(nrows=2, figsize=(10, 6))

    # Steps
    df = pd.Series(stats.episode_lengths)
    ma = df.rolling(rolling_window).mean()
    mstd = df.rolling(rolling_window).std()

    lower_bound = ma-mstd
    upper_bound = ma+mstd

    ax[0].plot(ma.index, ma, color=colors[0])
    ax[0].fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15, color=colors[0])
    ax[0].set_xlabel("Episode")
    ax[0].set_ylabel("Steps")
    ax[0].set_title("Episode Length over time")

    # Rewards
    df = pd.Series(stats.episode_rewards)
    ma = df.rolling(rolling_window).mean()
    mstd = df.rolling(rolling_window).std()

    lower_bound = ma-mstd
    upper_bound = ma+mstd

    ax[1].plot(ma.index, ma, color=colors[1])
    ax[1].fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15, color=colors[1])
    ax[1].set_xlabel("Episode")
    ax[1].set_ylabel("Episode Reward")
    ax[1].set_title("Episode Reward over time")

    # Selector choice
    if selector:
        df = pd.DataFrame(stats.selector_dist)
        if sel_norm:
            frequencies = df.apply(pd.value_counts, axis=1, **{"normalize": True}).fillna(0)
        else:
            frequencies = df.apply(pd.value_counts, axis=1).fillna(0)

        f = frequencies[0]

        ma = f.rolling(rolling_window).mean()
        mstd = f.rolling(rolling_window).std()

        lower_bound = ma-mstd
        upper_bound = ma+mstd

        ax[2].plot(ma.index, ma, color=colors[2])
        ax[2].fill_between(mstd.index, lower_bound, upper_bound, alpha=0.15, color=colors[2])
        ax[2].set_xlabel("Episode")
        ax[2].set_ylabel("Selector choice")
        ax[2].set_title("Selector distribution over time")
    fig.tight_layout()

def create_discretization_grid(low, high, bins=[10]):
    if len(bins) == 1:
        bins = np.repeat(bins, len(low))

    assert len(low) == len(high) == len(bins)

    grid = [np.linspace(low[i], high[i], bins[i], endpoint=False)[1:] for i, _ in enumerate(low)]
    return np.array(grid)

def get_discrete_state(state, window_size, env):
    discrete_state = (state - env.observation_space.low)/window_size
    return tuple(discrete_state.astype(np.int))

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
    ax.add_collection(mc.LineCollection(list(zip(samples, locs)), colors='orange'))