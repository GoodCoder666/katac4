import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List

import numpy as np
import torch
import torch.multiprocessing as mp

from game import ConnectFour
from mcts import MCTS

# --- Configuration ---

@dataclass
class RunConfig:
    name: str
    weight_fmt: str
    iters: List[int]
    loader: Callable = None

# legacy v1 model:
# https://github.com/GoodCoder666/katac4/blob/67d14d501b8d0f839c86e7b0b9f64ac0cfeae5cf/model.py
from model_v1 import InferenceGraph as InferenceGraphV1, Net as NetV1

def load_b3c128v1(path, board_size, device):
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    net = NetV1()
    net.load_state_dict(state_dict)
    graph = InferenceGraphV1(net, device, *board_size)
    return graph.policy_value_fn

# current v2 model
from model import InferenceGraph as InferenceGraphV2, Net as NetV2

def load_b3c128v2(path, board_size, device):
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    state_dict['policy_head.conv2.conv.weight'] = \
        state_dict['policy_head.conv2.conv.weight'][0:1]
    net = NetV2(c_policy=1)
    net.load_state_dict(state_dict)
    graph = InferenceGraphV2(net, device, *board_size)
    return graph.policy_value_fn

RUNS = [
    RunConfig(
        name='b3c128_v1',
        weight_fmt='./weights/b3c128nbt_2025-05-24_20-47-22/katac4_b3c128nbt_{it}.pth',
        iters=list(range(0, 30001, 500)),
        loader=load_b3c128v1
    ),
    RunConfig(
        name='b3c128_v2',
        weight_fmt='./weights/b3c128nbt_2025-08-18_19-22-00/katac4_b3c128nbt_{it}.pth',
        iters=list(range(500, 30001, 500)),  # skipping 0; using b3c128_v1:0 as benchmark
        loader=load_b3c128v2
    )
]

MCTS_KWARGS = {
    'exploration': False,
    'c_puct': 1.1,
    'c_fpu': 0.2,
    'n_playout': 200
}

ELO_K_FACTOR = 32
K_DECAY = 0.9995
SAVE_FILENAME = 'elo.json'
SAVE_FREQ = 16

# --- Helper Functions ---

def random_policy_value_fn(game, *args):
    moves = game.sensible_moves()
    logits = np.random.randn(len(moves)) * 0.4
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    return moves, probs, 0.0

def load_policy(run_cfg, iteration, game, device):
    if iteration == 0:
        return random_policy_value_fn
    path = run_cfg.weight_fmt.format(it=iteration)
    return run_cfg.loader(path, (game.height, game.width), device)

def apply_temperature(probs, temp):
    logits = np.log(np.clip(probs, 1e-10, None)) / temp
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def one_game(game, mcts1, mcts2):
    step = 0
    board_size = game.height * game.width
    while not game.is_terminal():
        temp = max(1.03, 1.35 * pow(0.66, step / board_size))
        mcts = mcts1 if game.player == 1 else mcts2
        acts, probs = mcts.get_move_probs(game, root_prior_temp=temp)
        act_temp = 0.5 * pow(0.8, step / game.width)
        move = np.random.choice(acts, p=apply_temperature(probs, act_temp))
        game.step(move)
        mcts1.apply_move(move)
        mcts2.apply_move(move)
        step += 1
    return game.winner

def one_test(game, policy1, policy2):
    game2 = deepcopy(game)
    mcts1 = MCTS(policy1, **MCTS_KWARGS)
    mcts2 = MCTS(policy2, **MCTS_KWARGS)
    result1 = one_game(game, mcts1, mcts2)
    mcts1.reset()
    mcts2.reset()
    result2 = one_game(game2, mcts2, mcts1)
    return result1, -result2

def elo_worker(device, result_queue, runs_dict):
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    all_players = [
        (run_name, it)
        for run_name, run_cfg in runs_dict.items()
        for it in run_cfg.iters
    ]

    while True:
        (r1, it1), (r2, it2) = random.sample(all_players, 2)

        game = ConnectFour()
        p1_fn = load_policy(runs_dict[r1], it1, game, device)
        p2_fn = load_policy(runs_dict[r2], it2, game, device)

        result = one_test(game, p1_fn, p2_fn)

        k1, k2 = f'{r1}:{it1:05d}', f'{r2}:{it2:05d}'
        result_queue.put((k1, k2, (sum(result) + 2) / 4))

def expected_score(elo1, elo2):
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))

def elo_update(elo1, elo2, outcome):
    elo1 += ELO_K_FACTOR * (outcome - expected_score(elo1, elo2))
    elo2 += ELO_K_FACTOR * ((1 - outcome) - expected_score(elo2, elo1))
    return elo1, elo2

def main():
    global ELO_K_FACTOR
    num_gpus = torch.cuda.device_count()
    num_workers = 2 * num_gpus
    print(f'{num_workers} workers on {num_gpus} GPUs')

    runs_dict = {r.name: r for r in RUNS}

    elos = {}
    if os.path.exists(SAVE_FILENAME):
        print('Initializing with existing ELO ratings')
        with open(SAVE_FILENAME, 'r', encoding='utf-8') as file:
            elos = json.load(file)

    result_queue = mp.Queue()
    workers = [mp.Process(target=elo_worker, args=(f'cuda:{i % num_gpus}', result_queue, runs_dict))
               for i in range(num_workers)]
    for worker in workers:
        worker.start()

    game_count = 0
    benchmark_key = f'{RUNS[0].name}:00000'
    while True:
        p1, p2, result = result_queue.get()
        elos[p1], elos[p2] = elo_update(elos.get(p1, 0), elos.get(p2, 0), result)
        game_count += 2
        if game_count % SAVE_FREQ == 0:
            ELO_K_FACTOR *= K_DECAY
            benchmark = elos[benchmark_key]
            for player in elos:
                elos[player] -= benchmark
            print(f'Games played: {game_count}, K: {ELO_K_FACTOR:.3f}')
            with open(SAVE_FILENAME, 'w', encoding='utf-8') as file:
                json.dump(elos, file, indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
