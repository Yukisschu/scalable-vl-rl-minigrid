
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch

ACTION_NAMES = ["turn_left", "turn_right", "forward", "pickup", "drop", "toggle", "done"]


# -- Training log formatting ---------------------------------------------------

def format_update_log(update, num_frames, fps, duration, logs, upd_logs):
    """
    Return a single compact log line in the style:
      U 4 | F 008192 | FPS 1851 | D 4 | rR:μσmM 0.37 0.40 0.00 0.95 | ...

    Parameters
    ----------
    update     : int   update index
    num_frames : int   total frames so far
    fps        : float frames-per-second for this update
    duration   : float wall-clock seconds for this update
    logs       : dict  from algo.collect_experiences()
    upd_logs   : dict  from algo.update_parameters()
    """
    rets = np.array(logs.get("return_per_episode", []), dtype=float)
    lens = np.array(logs.get("num_frames_per_episode", []), dtype=float)

    def _stats(arr):
        if len(arr) == 0:
            return "nan nan nan nan"
        return f"{arr.mean():.2f} {arr.std():.2f} {arr.min():.2f} {arr.max():.2f}"

    return (
        f"U {update} | "
        f"F {num_frames:06d} | "
        f"FPS {int(fps)} | "
        f"D {int(duration)} | "
        f"rR:\u03bc\u03c3mM {_stats(rets)} | "
        f"F:\u03bc\u03c3mM {_stats(lens)} | "
        f"H {upd_logs['entropy']:.3f} | "
        f"V {upd_logs['value']:.3f} | "
        f"pL {upd_logs['policy_loss']:.3f} | "
        f"vL {upd_logs['value_loss']:.3f} | "
        f"\u2207 {upd_logs['grad_norm']:.3f}"
    )


# -- Training curves -----------------------------------------------------------

def plot_training_curves(csv_path, save_dir=None, smooth_window=30):
    """
    Read training_log.csv and produce a 2x3 metrics figure.
    Saves to save_dir/training_curves.png if save_dir is given.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No training data in CSV yet.")
        return

    def col(key):
        return np.array([float(r[key]) for r in rows if r.get(key) not in (None, "")])

    def smooth(arr, w):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w) / w, mode="valid")

    steps   = col("global_step") / 1e6
    returns = col("mean_ep_return")
    lengths = col("mean_ep_len")
    entropy = col("entropy")
    pg_loss = col("policy_loss")
    v_loss  = col("value_loss")
    W = smooth_window

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("PPO Training -- MiniGrid-Fetch", fontsize=13, fontweight="bold")

    def _panel(ax, y, color, title, ylabel, hlines=None):
        x = steps[:len(y)]
        ax.plot(x, y, alpha=0.25, color=color, linewidth=0.9)
        if len(y) >= W:
            ax.plot(x[W - 1:], smooth(y, W), color=color, linewidth=2,
                    label=f"MA({W})")
        if hlines:
            for val, lc, ls, lbl in hlines:
                ax.axhline(val, color=lc, linestyle=ls, linewidth=0.9, label=lbl)
        ax.set_xlabel("Steps (M)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        # y-axis: fit to data range with 5% padding, ignoring hlines
        if len(y) > 0:
            pad = (y.max() - y.min()) * 0.05 or abs(y.mean()) * 0.1 or 0.1
            ax.set_ylim(y.min() - pad, y.max() + pad)

    max_H = float(np.log(7))
    _panel(axes[0, 0], returns, "steelblue",  "Episode Return",        "Mean Return",
           [(0, "black", "--", "y=0")])
    _panel(axes[0, 1], lengths, "darkorange", "Episode Length  (down)", "Steps / Episode",
           [(320, "red", "--", "timeout 320")])
    _panel(axes[0, 2], entropy, "seagreen",   "Entropy  (keep > 0.5)", "H(pi)",
           [(max_H, "gray", "--", f"max H~{max_H:.2f}"),
            (0.5,   "red",  ":",  "collapse < 0.5")])
    _panel(axes[1, 0], pg_loss, "crimson",    "Policy Loss",           "Loss")
    _panel(axes[1, 1], v_loss,  "navy",       "Value Loss",            "MSE Loss")
    axes[1, 2].axis("off")

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {path}")

    plt.show()

    if len(returns):
        print("\nFinal stats (last 50 updates):")
        for name, arr in [("Return", returns), ("Length", lengths),
                          ("Entropy", entropy), ("PG Loss", pg_loss)]:
            tail = arr[-50:]
            print(f"  {name:<10}: {tail.mean():.4f}")


# -- Episode recording ---------------------------------------------------------

def record_episode(model, preprocess_fn, env_id, device,
                   seed=0, deterministic=False, dense_reward=False):
    """
    Roll out one episode using ACModel.

    Parameters
    ----------
    model         : ACModel (already on `device`)
    preprocess_fn : callable returned by make_preprocess_obss(vocab)
    env_id        : str, e.g. "MiniGrid-Fetch-8x8-N3-v0"
    device        : torch.device
    seed          : int  -- passed to env.reset(seed=seed)
    deterministic : bool -- argmax vs sample
    dense_reward  : bool -- wrap with FetchDenseRewardWrapper

    Returns
    -------
    frames, acts, success, mission, total_ret
    """
    from utils.env import make_env

    env     = make_env(env_id, dense_reward=dense_reward, render_mode="rgb_array")
    ob, _   = env.reset(seed=seed)
    mission = ob["mission"]

    frames, acts = [env.render()], []
    done, total_ret = False, 0.0
    memory = torch.zeros(1, model.memory_size, device=device)

    while not done:
        preprocessed = preprocess_fn([ob], device=device)
        with torch.no_grad():
            dist, _, memory = model(preprocessed, memory)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()

        ob, r, term, trunc, _ = env.step(action.item())
        frames.append(env.render())
        acts.append(ACTION_NAMES[action.item()])
        total_ret += r
        done = term or trunc

    env.close()
    return frames, acts, bool(total_ret > 0), mission, total_ret


# -- Inline animations ---------------------------------------------------------

def _make_animation(frames, acts, success, mission, interval=120):
    fig, ax = plt.subplots(figsize=(4, 4.5))
    ax.axis("off")
    im    = ax.imshow(frames[0])
    color = "green" if success else "red"
    title = ax.set_title(f"{mission}\nstep 0", fontsize=8, color=color)

    def update(i):
        im.set_data(frames[i])
        label = acts[i - 1] if i > 0 else "--"
        title.set_text(f"{mission}\nstep {i}  |  {label}")
        return [im, title]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=interval, blit=False, repeat=True,
    )
    plt.close(fig)
    return ani


def show_animations(episodes, n=5, interval=120):
    """Display the first n episodes as inline HTML animations in a notebook."""
    from IPython.display import HTML, display as ipy_display
    for i, (frames, acts, success, mission, ret) in enumerate(episodes[:n]):
        tag = "SUCCESS" if success else "failed"
        print(f"\n-- Episode {i + 1}/{min(n, len(episodes))}  "
              f"[{tag}]  '{mission}'  {len(acts)} steps  ret={ret:.3f} --")
        ani = _make_animation(frames, acts, success, mission, interval)
        ipy_display(HTML(ani.to_jshtml()))


# -- GIF export ----------------------------------------------------------------

def save_gifs(episodes, gif_dir, fps=8, resize=256):
    """Save each episode as a .gif under gif_dir."""
    from PIL import Image as PILImage
    os.makedirs(gif_dir, exist_ok=True)
    for idx, (frames, acts, success, mission, ret) in enumerate(episodes):
        tag   = "ok" if success else "fail"
        fname = os.path.join(gif_dir, f"ep{idx + 1:02d}_{tag}.gif")
        pil   = [PILImage.fromarray(f).resize((resize, resize), PILImage.NEAREST)
                 for f in frames]
        pil[0].save(fname, save_all=True, append_images=pil[1:],
                    duration=int(1000 / fps), loop=0)
        print(f"  Saved {fname}  ({len(frames)} frames)")


# -- Static grid of final frames -----------------------------------------------

def show_episode_grid(episodes, save_path=None, ncols=3):
    """Show the final frame of each episode in a grid."""
    n     = len(episodes)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, (frames, acts, success, mission, ret) in zip(axes_flat, episodes):
        ax.imshow(frames[-1])
        color  = "green" if success else "red"
        status = "SUCCESS" if success else "failed"
        ax.set_title(f"{mission}\n{len(acts)} steps  [{status}]  ret={ret:.3f}",
                     fontsize=8, color=color)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Agent Behaviour -- Final Frame", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")

    plt.show()
