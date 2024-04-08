import numpy as np

from   matplotlib.animation import FuncAnimation
import matplotlib.pyplot    as     plt
import matplotlib.colors    as     colors
import matplotlib.cm        as     cm


def replace_tag(filename, tokens_in, token_out):
    if isinstance(tokens_in, str):
        # "a b" will turn into ["a", "b"]
        # "a" will turn into ["a"]
        tokens_in = tokens_in.split()

    if not any(token in filename for token in tokens_in):
        raise ValueError("None of the input tags is present in the filename")

    for token_in in tokens_in:
        filename = filename.replace(token_in, token_out)
    return filename


def find_highest_wf(wfs, event_number):
    assert len(wfs.shape) == 3, "input must be 3-dimensional"

    wfs   = wfs[event_number]
    index = np.argmax(np.max(wfs, axis=1))
    return wfs[index]


def find_pmap_with_s1_s2(pmaps):
    for evt_no, pmap in pmaps.items():
        if pmap.s1s and pmap.s2s:
            return evt_no, pmap
    raise RuntimeError("There are no events with S1 and S2")


def sphere(ax, x0, y0, z0, r,  N=50, color="r"):
    u = np.linspace(0, 2*np.pi, N)
    v = np.linspace(0,   np.pi, N)
    x = np.outer(np.cos (u), np.sin(v)) * r + x0
    y = np.outer(np.sin (u), np.sin(v)) * r + y0
    z = np.outer(np.ones(N), np.cos(v)) * r + z0
    ax.plot_surface(x, y, z, cstride=5, rstride=5, color=color, edgecolor=color, lw=0.1, alpha=0.1)


def create_hits_animation(all_hits, hitsize):
    colorbar = True
    def animate(group):
        nonlocal colorbar

        z, hits = group
        if len(hits) <= 3: return
        plt.cla()
        plt.scatter(hits.X, hits.Y, marker="s", s=hitsize, c=hits.E, vmin=0, vmax=all_hits.E.max())
        plt.gca().set_facecolor("whitesmoke")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        plt.title(f"z = {z:.0f} mm")
        plt.xlim(all_hits.X.min() - 5, all_hits.X.max() + 5)
        plt.ylim(all_hits.Y.min() - 5, all_hits.Y.max() + 5)
        if colorbar:
            plt.colorbar()
            colorbar=False

    fig = plt.figure()
    plt.ioff()
    return FuncAnimation(fig, animate, frames=all_hits.groupby("Z"))


def plot_hits_voxelized(hits):
    nx = hits.X.unique().size//4
    ny = hits.Y.unique().size//4
    nz = hits.Z.unique().size//1

    e, (z,x,y) = np.histogramdd(hits.filter(list("ZXY")).values, (nz, nx, ny), weights=hits.E)
    z,x,y = np.meshgrid(z,x,y, indexing="ij")

    mapper = cm.ScalarMappable(cmap=cm.gnuplot2_r)
    colors = mapper.to_rgba(e.flatten()/e.max(), alpha=0.5).reshape(e.shape + (4,))

    plt.ion()
    ax = plt.figure(figsize=(10,8)).add_subplot(projection='3d')
    ax.voxels(z, x, y, e,
              facecolors=colors,
              edgecolors=colors,
              linewidth=0.5)
    ax.set_xlabel("z (mm)", labelpad=50)
    ax.set_ylabel("x (mm)", labelpad=50)
    ax.set_zlabel("y (mm)", labelpad=50)
