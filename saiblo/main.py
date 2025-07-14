import time

load_begin = time.time()

import sys

import numpy as np
import torch
import torch.nn.functional as F
from search import MCGS

from game import ConnectFour

input = sys.stdin.readline

c_puct = 1.1
c_fpu = 0.2

device = torch.device('cpu')
model = torch.jit.load('model.pt', map_location=device)

z_table = np.load('z_lookup.npy', allow_pickle=False)

load_time = time.time() - load_begin

print('Load time:', load_time, file=sys.stderr)

def policy_value_fn(game):
    state = torch.FloatTensor(game.state()).unsqueeze(0)
    policy_logits, value_logits = model(state)
    sensible_moves = game.sensible_moves()
    policy_logits = policy_logits.squeeze(0)[game.top[sensible_moves], sensible_moves]
    policy = F.softmax(policy_logits, dim=0).numpy()
    win_rate, loss_rate, _ = F.softmax(value_logits.squeeze(0), dim=0).tolist()
    return sensible_moves, policy, win_rate - loss_rate

def send(msg):
    n = len(msg)
    chrs = map(
        lambda x: chr(n >> x & 255),
        [24, 16, 8, 0]
    )
    sys.stdout.write(''.join(chrs) + msg)
    sys.stdout.flush()

def main():
    height, width, noX, noY = map(int, input().split())
    noX = height - 1 - noX
    game = ConnectFour(height, width, (noX, noY))
    mcgs = MCGS(policy_value_fn, z_table.tolist(), c_puct, c_fpu)
    first_move = True
    timeout = max(0.1, 2.7 - load_time)
    while True:
        lastY = int(input().split()[1])
        if lastY != -1:
            game.step(lastY)
        print('\n' + str(game), file=sys.stderr)
        action, lcb = mcgs.search(game, timeout)
        move_x, move_y = height - 1 - int(game.top[action]), action
        send(f'{move_x} {move_y}')
        if first_move:
            first_move = False
            timeout = 2.8
        N, Q = mcgs.root.N, mcgs.root.Q
        print(f'\n{(Q+1)/2:.1%} N={N} Q={Q:.3f} LCB={lcb:.3f}', file=sys.stderr)
        game.step(action)
        print(game, file=sys.stderr)

with torch.inference_mode():
    main()