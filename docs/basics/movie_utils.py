# some plotting utilities to make movies

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def movie_profile(
    y_tx,
    x,
    out,
    *,
    fps=30,
    t=None,
    xlabel="x",
    ylabel="value",
    title=None,
    ylim=None,
    xlim=None,
    sort_x=True,
    dpi=150,
    bitrate=2000,
    codec="h264",
    writer="ffmpeg",
    extra_ffmpeg_args=("-pix_fmt", "yuv420p"),
):
    """
    Save an MP4 (or GIF) movie of a 1D profile y(x) evolving over time.

    Parameters
    ----------
    y_tx : array-like, shape (n_timesteps, n_locations)
        Values over time (rows) and space (columns).
    x : array-like, shape (n_locations,)
        Spatial coordinates corresponding to columns of y_tx.
    out : str or pathlib.Path
        Output filename, e.g. "profile.mp4" or "profile.gif".
    fps : int
        Frames per second.
    t : array-like, optional
        Time values for each frame (n_timesteps,). If provided, used in the title.
    xlabel, ylabel : str
        Axis labels.
    title : str, optional
        Base title. If None, only the time/frame is shown.
    ylim, xlim : tuple, optional
        Axis limits. If None, computed from data.
    sort_x : bool
        If True, sort x (and corresponding columns) for a clean line plot.
    dpi : int
        Output DPI.
    bitrate : int
        Video bitrate (kbps-ish; matplotlib passes through to ffmpeg).
    codec : str
        ffmpeg codec, default "h264".
    writer : {"ffmpeg","pillow"}
        Writer backend. Use "ffmpeg" for mp4, "pillow" for gif.
    extra_ffmpeg_args : tuple[str, ...]
        Extra arguments passed to ffmpeg (helps browser compatibility).

    Returns
    -------
    str
        The output path as a string.
    """
    y_tx = np.asarray(y_tx)
    x = np.asarray(x)

    if y_tx.ndim != 2:
        raise ValueError(
            f"y_tx must be 2D (n_timesteps, n_locations); got shape {y_tx.shape}"
        )
    if x.ndim != 1 or x.shape[0] != y_tx.shape[1]:
        raise ValueError(
            f"x must be 1D with length n_locations={y_tx.shape[1]}; got shape {x.shape}"
        )

    if t is not None:
        t = np.asarray(t)
        if t.shape[0] != y_tx.shape[0]:
            raise ValueError(
                f"t must have length n_timesteps={y_tx.shape[0]}; got shape {t.shape}"
            )

    if sort_x:
        order = np.argsort(x)
        x = x[order]
        y_tx = y_tx[:, order]

    # Robust limits (ignore NaNs)
    if xlim is None:
        xlim = (np.nanmin(x), np.nanmax(x))
    if ylim is None:
        y0 = np.nanmin(y_tx)
        y1 = np.nanmax(y_tx)
        pad = 0.05 * (y1 - y0 if y1 > y0 else 1.0)
        ylim = (y0 - pad, y1 + pad)

    fig, ax = plt.subplots()
    (line,) = ax.plot(x, y_tx[0], lw=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    title_text = ax.set_title("")

    interval_ms = 1000 / fps

    def _frame_title(i):
        base = "" if title is None else str(title)
        if t is None:
            time_part = f"frame {i}"
        else:
            time_part = f"t = {t[i]:.6g}"
        return f"{base} ({time_part})" if base else time_part

    def update(i):
        line.set_ydata(y_tx[i])
        title_text.set_text(_frame_title(i))
        return line, title_text

    ani = FuncAnimation(
        fig, update, frames=y_tx.shape[0], interval=interval_ms, blit=True
    )

    out = str(out)
    ext = out.lower().rsplit(".", 1)[-1]

    if writer == "ffmpeg" or ext == "mp4":
        try:
            from matplotlib.animation import FFMpegWriter
        except Exception as e:
            plt.close(fig)
            raise RuntimeError(
                "FFMpegWriter unavailable. Install ffmpeg (and ensure it's on PATH), "
                "or set writer='pillow' to save a GIF."
            ) from e

        w = FFMpegWriter(
            fps=fps,
            codec=codec,
            bitrate=bitrate,
            extra_args=list(extra_ffmpeg_args) if extra_ffmpeg_args else None,
        )
        ani.save(out, writer=w, dpi=dpi)
    else:
        # GIF via Pillow
        try:
            from matplotlib.animation import PillowWriter
        except Exception as e:
            plt.close(fig)
            raise RuntimeError(
                "PillowWriter unavailable. Install pillow, or use writer='ffmpeg' for mp4."
            ) from e

        w = PillowWriter(fps=fps)
        ani.save(out, writer=w, dpi=dpi)

    plt.close(fig)
    return out
