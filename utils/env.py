import gymnasium as gym
import numpy as np
import torch
from torch_ac.utils import DictList
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

# Row and column of the agent's own cell inside the 7×7 egocentric observation
# grid (bottom-centre of the view, used as the reference point for distance
# calculations in the dense reward shaping).
_AGENT_FOV_ROW = 6   # agent position in 7×7 egocentric view
_AGENT_FOV_COL = 3


# ── Dense reward wrapper ──────────────────────────────────────────────────────
class FetchDenseRewardWrapper(gym.Wrapper):
    """
    Dense reward shaping for MiniGrid-Fetch:
      - step penalty          (discourages dawdling)
      - first-sight bonus     (one-off when target becomes visible)
      - approach shaping      (potential-based: reward for getting closer)
      - success bonus         (added on top of env's sparse reward)
      - useless action penalty (drop-while-empty and done action)
    """

    def __init__(self, env,
                 step_penalty=-0.01,
                 first_sight_bonus=0.5,
                 approach_scale=0.3,
                 success_reward=5.0,
                 useless_action_penalty=-0.05):
        super().__init__(env)
        self.step_penalty           = step_penalty
        self.first_sight_bonus      = first_sight_bonus
        self.approach_scale         = approach_scale
        self.success_reward         = success_reward
        self.useless_action_penalty = useless_action_penalty
        # Tracks the closest the agent has ever been to the target this episode;
        # used to compute potential-based approach shaping rewards.
        self._min_dist_ever   = float("inf")
        # Ensures the first-sight bonus is awarded only once per episode.
        self._first_sight_done = False

    @property
    def target_color(self):
        return self.env.unwrapped.targetColor

    @property
    def target_type(self):
        return self.env.unwrapped.targetType

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset per-episode shaping state so each episode starts fresh
        self._min_dist_ever    = float("inf")
        self._first_sight_done = False
        return obs, info

    def step(self, action):
        # Capture carrying state before the step; used to detect a drop on empty hand
        before_carrying = self.env.unwrapped.carrying is not None

        obs, orig_reward, terminated, truncated, info = self.env.step(action)
        # Start the shaped reward from the baseline step penalty
        dense = self.step_penalty

        # Penalise useless actions
        if action == 4 and not before_carrying:   # drop while not carrying anything
            dense += self.useless_action_penalty
        elif action == 6:                          # done action (never useful in Fetch)
            dense += self.useless_action_penalty

        if terminated and float(orig_reward) > 0:
            # Correct pickup — add success bonus on top of the environment's sparse reward
            dense += self.success_reward
            # Reset shaping state so subsequent episodes are not affected
            self._min_dist_ever    = float("inf")
            self._first_sight_done = False
        elif not terminated:
            # Approach shaping: reward progress towards the target
            d = self._target_dist(obs["image"])
            if d is not None:
                if not self._first_sight_done:
                    # One-off bonus the first time the target enters the field of view
                    dense += self.first_sight_bonus
                    self._first_sight_done = True
                    self._min_dist_ever    = d
                else:
                    # Potential-based shaping: reward only genuine improvement
                    improvement = self._min_dist_ever - d
                    if improvement > 0:
                        dense += self.approach_scale * improvement
                        self._min_dist_ever = d

        return obs, dense, terminated, truncated, info

    def _target_dist(self, image):
        """Return the minimum Euclidean distance from the agent to the target object
        in the egocentric observation grid, or None if the target is not visible."""
        # Channel 0 encodes object type; channel 1 encodes object colour
        obj_idx = OBJECT_TO_IDX[self.target_type]
        col_idx = COLOR_TO_IDX[self.target_color]
        mask    = (image[:, :, 0] == obj_idx) & (image[:, :, 1] == col_idx)
        if not mask.any():
            # Target is not visible in the current field of view
            return None
        # Find all grid cells occupied by matching pixels and compute distances
        pos   = np.argwhere(mask).astype(float)
        agent = np.array([_AGENT_FOV_ROW, _AGENT_FOV_COL], dtype=float)
        return float(np.linalg.norm(pos - agent, axis=1).min())


# ── Environment factory ─────────────────────────────────────────────────────

def make_env(env_id, dense_reward=True, render_mode=None, seed=None, **reward_kwargs):
    """Create a single (optionally wrapped) MiniGrid-Fetch environment.

    If dense_reward is True, the environment is wrapped with
    FetchDenseRewardWrapper using the provided reward_kwargs.
    Any extra keyword arguments are forwarded directly to the wrapper.
    """
    # Import minigrid here to ensure all custom environments are registered
    # with gymnasium before gym.make is called.
    import minigrid  # noqa: ensure env is registered
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        # Gymnasium environments are seeded via the first reset call
        env.reset(seed=seed)
    if dense_reward:
        env = FetchDenseRewardWrapper(env, **reward_kwargs)
    return env


# ── Vocabulary & observation preprocessing ───────────────────────────────────
def build_vocab(env_id, n_samples=200):
    """
    Sample missions from the environment and build a word -> index vocabulary.
    Always includes '<pad>' at index 0.
    """
    import minigrid  # noqa
    env   = gym.make(env_id, render_mode=None)
    # Index 0 is reserved for the padding token used in fixed-length sequences
    vocab = {"<pad>": 0}
    for _ in range(n_samples):
        obs, _ = env.reset()
        # Add each unseen word with the next available integer index
        for word in obs['mission'].split():
            if word not in vocab:
                vocab[word] = len(vocab)
    env.close()
    return vocab


def tokenize(mission, vocab, max_len=10):
    """Convert a mission string to a fixed-length list of token indices.

    Uses simple split and lowercasing to be robust to case differences.
    Unknown words are mapped to the '<pad>' index (0).
    Sequences longer than max_len are truncated; shorter ones are zero-padded.
    """
    tokens = [vocab.get(w.lower(), 0) for w in mission.split()]
    # Truncate to max_len, then right-pad with zeros to ensure a fixed length
    return tokens[:max_len] + [0] * max(0, max_len - len(tokens))


def make_preprocess_obss(vocab):
    """
    Return a preprocess_obss(obss, device=None) closure compatible with
    torch_ac.PPOAlgo (which calls it as preprocess_obss(obs_list, device=...)).
    """
    def preprocess_obss(obss, device=None):
        # Stack raw pixel grids from all observations into a single float array
        images = np.array([obs["image"] for obs in obss])
        # Tokenise each mission string using the shared vocabulary
        texts  = np.array([tokenize(obs["mission"], vocab) for obs in obss])
        # Return a DictList so downstream model code can access fields by name
        return DictList({
            "image": torch.tensor(images, dtype=torch.float, device=device),
            "text":  torch.tensor(texts,  dtype=torch.long,    device=device),
        })
    return preprocess_obss
