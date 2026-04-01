import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    """
    Actor-Critic model for MiniGrid.

    obs_space must be a plain dict:
        {"image": (H, W, C), "text": vocab_size}
    """

    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()
        self.use_text   = use_text
        self.use_memory = use_memory

        # ── Image encoder ──────────────────────────────────────────────
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
        )
        n, m = obs_space["image"][0], obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        # ── Optional LSTM memory ────────────────────────────────────────
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # ── Optional text encoder + FiLM ────────────────────────────────
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding      = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn            = nn.GRU(self.word_embedding_size, self.text_embedding_size,
                                               batch_first=True)
            self.film_generator = nn.Sequential(
                nn.Linear(self.text_embedding_size, self.text_embedding_size),
                nn.ReLU(),
                nn.Linear(self.text_embedding_size, 2 * self.image_embedding_size),
            )

        # ── Actor / Critic heads ────────────────────────────────────────
        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        # Image: (B, H, W, C) → conv expects (B, C, H, W)
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        # LSTM memory
        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size],
                      memory[:, self.semi_memory_size:])
            hidden      = self.memory_rnn(x, hidden)
            visual_feat = hidden[0]
            memory      = torch.cat(hidden, dim=1)
        else:
            visual_feat = x

        # FiLM conditioning on mission text
        if self.use_text:
            _, h = self.text_rnn(self.word_embedding(obs.text))
            text_feat        = h[-1]
            film_params      = self.film_generator(text_feat)
            gamma, beta      = film_params.chunk(2, dim=1)
            visual_feat      = gamma * visual_feat + beta

        dist  = Categorical(logits=F.log_softmax(self.actor(visual_feat), dim=1))
        value = self.critic(visual_feat).squeeze(1)
        return dist, value, memory
