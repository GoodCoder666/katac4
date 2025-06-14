# katac4

An AlphaZero engine for [Saiblo Connect4](https://www.saiblo.net/game/3), featuring a pure Python implementation of key [KataGo](https://github.com/lightvector/KataGo) techniques.

## Project Structure

```
katac4/
├── README.md        # This file
│
├── runs/            # TensorBoard logs
├── weights/         # Model checkpoints for each training run
│
├── docs/            # Documentation files
│   └── methods.md    # Overview of improvements over original AlphaZero
│
├── saiblo/          # Saiblo Connect4 submission package
│   ├── main.py       # Entry point
│   ├── game.py       # Standalone game environment (with zobrist hash)
│   └── search.py     # Monte Carlo Graph Search, used during online play
│
├── train.py         # Main training script
├── model.py         # Neural network model definition (b3c128nbt)
├── game.py          # Game environment
├── mcts.py          # Monte Carlo Tree Search, used during training
├── elo_eval.py      # Script for evaluating model ELO ratings
├── elo.json         # ELO ratings generated by elo_eval.py
├── elo_plot.py      # Plots ELO rating progression
├── export_model.py  # Converts model to TorchScript for deployment
├── benchmark.py     # Benchmarks model performance
└── human_play.py    # Allows human interaction with the model
```

## Quick Start

### Self-Play Training

Before starting self-play training, you may want to adjust a few parameters in `train.py`:

* `epochs`: Total number of training epochs
* `epoch_size`: Number of self-play games per epoch
* `parallel_games`: Number of games run in parallel
* `num_gpus`: Number of GPUs to utilize

> [!IMPORTANT]
>
> To exactly reproduce the most recent training run, **do not change** the default parameter values in the code.

To begin training, run:

```bash
python3 train.py
```

Model checkpoints will be saved to the `weights/` directory. To monitor progress with TensorBoard:

```bash
tensorboard --logdir runs
```

### ELO Evaluation

To evaluate model performance using ELO, edit the `ai_list` and `weight_format` variables in `elo_eval.py` as needed. Then run:

```bash
python3 elo_eval.py
```

Models will compete in randomized pairings, with results saved to `elo.json`. To visualize the rating progression:

```bash
python3 elo_plot.py
```

> [!NOTE]
>
> The ELO evaluation process is computationally intensive. For reference, the most recent evaluation was based on approximately 70,000 games and required 2.5 days to complete on four RTX 4090 GPUs.

### Submitting to Saiblo

Select your best-performing model checkpoint (typically the final one or the one with highest ELO), and set its path in `export_model.py`. Export it to TorchScript format by running:

```bash
python3 export_model.py
```

This will generate `model.pt` in the `saiblo/` directory. To create your submission, zip **only the files** inside the `saiblo/` folder—**do not include the folder itself** in the archive.
