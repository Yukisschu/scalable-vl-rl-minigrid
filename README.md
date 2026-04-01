# MiniGrid-Fetch — PPO Variant Comparison

Reinforcement learning final assignment. Trains and compares four PPO-based agent variants on the **MiniGrid-Fetch** task across three grid sizes (5×5, 6×6, 8×8), with additional experiments in curriculum learning (CL) and imitation learning (IL).

---

## Task

The agent receives a partial (7×7 egocentric) RGB observation and a natural-language mission string (e.g. *"go fetch a blue ball"*). It must navigate the grid and pick up the target object within a fixed step budget.

---

## Variants

| Variant | Fusion | memo + text | Reward |
|---------|-------|----------|--------|
| `baseline`  | Non | OFF  | sparse |
| `memo_text`  | Concat | ON | sparse |
| `film_dense`   | FiLM ($\gamma \cdot x$+$\beta$)          | ON  | dense  |
| `film_sparse`  | FiLM ($\gamma \cdot x$+$\beta$)          | ON  | sparse |

**FiLM** modulates CNN image features with GRU-encoded text, replacing simple concatenation with feature-wise linear modulation.
**Dense reward** adds a step penalty, first-sight bonus, approach shaping, and a success reward of +10.

---

## Repository Structure

```
├── config.YAML                   # Shared hyperparameters (PPO, reward, eval)
├── train_compare_5x5.ipynb       # Train all 4 variants on MiniGrid-Fetch-5x5-N2-v0
├── train_compare_6x6.ipynb       # Train all 4 variants on MiniGrid-Fetch-6x6-N2-v0
├── train_compare_8x8_CL.ipynb    # Train with curriculum learning (warm-start from 6x6)
├── train_compare_8x8_IL.ipynb    # Train with behavioural cloning pre-training
├── results_combined.ipynb        # Aggregate plots and post-CL agent visualisation
├── runs/                         # Training outputs 
└── utils/
    ├── env.py                    # FetchDenseRewardWrapper, make_env, tokeniser
    ├── format.py                 # Observation preprocessor, Vocabulary
    ├── model_baseline.py         # Baseline ACModel (CNN + optional LSTM + concat)
    ├── model_film.py             # FiLM ACModel (CNN + BiGRU + FiLM + optional LSTM)
    └── plotting.py               # Training curves, episode recording, GIF export
```

---

## Quickstart

```bash
pip install -r requirements.txt
```

Open and run the notebooks in order:

1. `train_compare_5x5.ipynb`
2. `train_compare_6x6.ipynb`
3. `train_compare_8x8_CL.ipynb` (requires 6×6 checkpoints for CL warm-start)
4. `train_compare_8x8_IL.ipynb`
5. `results_combined.ipynb`

Hyperparameters are controlled via `config.YAML`.


