import os
import random
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from game import ConnectFour
from mcts import MCTS
from model import InferenceGraph, Net

model_name = 'b3c128nbt'
run_name = f'{model_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
weights_dir = f'./weights/{run_name}'

# training hyperparameters
batch_size = 256
buffer_config = {
    'alpha': 0.75,
    'beta': 0.4,
    'c': 250000
}
epochs = 30000
epoch_size = 16
momentum = 0.9
lr = 6e-5 * batch_size

c_puct = 1.1
c_fpu = 0.2
vloss_scaler = 1.5
l2_const = 6e-5

pcr_rate = 0.25
tiny_playouts = 160
large_playouts = 800

parallel_games = 16
num_gpus = 4


class ReplayBuffer:
    '''
    Dynamic-sized replay buffer.
    
    window_size = c * (1 + beta * ((N / c) ** alpha - 1) / alpha)
    '''
    outcome_trans = {1: 0, -1: 1, 0: 2}

    def __init__(self, alpha, beta, c):
        self.buffers = {}
        for h in range(9, 13):
            for w in range(9, 13):
                self.buffers[(h, w)] = deque()
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.count = 0

    def push(self, height, width, history, outcome):
        buffer = self.buffers[(height, width)]
        for state, probs in history:
            buffer.append((state, probs, self.outcome_trans[outcome]))
            outcome = -outcome
        self.count += len(history)
        window_size = self.c * (1 + ((self.count / self.c) ** self.alpha - 1) / self.alpha * self.beta)
        window_size /= 16.0
        while len(buffer) > window_size:
            buffer.popleft()

    def is_samplable(self, batch_size):
        samples_per_size = batch_size // 16
        return all(len(buffer) >= samples_per_size for buffer in self.buffers.values())

    def sample(self, batch_size):
        samples_per_size = batch_size // 16
        for buffer in self.buffers.values():
            batch = random.sample(buffer, samples_per_size)
            flip_indices = random.sample(range(len(batch)), len(batch) // 2)
            for i in flip_indices:
                state, probs, outcome = batch[i]
                batch[i] = (np.flip(state, axis=2), np.fliplr(probs), outcome)
            states, probs, outcomes = zip(*batch)
            yield np.stack(states), np.stack(probs), np.array(outcomes, dtype=np.long)


def apply_temperature(probs, temp):
    logits = np.log(np.clip(probs, 1e-10, None)) / temp
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

def sample_discrete_exp(mean):
    p = 1 - np.exp(-1.0 / mean)
    return np.random.geometric(p) - 1

def selfplay_worker(worker_id, shared_model, replay_queue):
    gpu_id = worker_id % num_gpus
    first_h, first_w = divmod(worker_id % 16, 4)
    first_h, first_w = first_h + 9, first_w + 9
    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    net = Net()

    def selfplay(board_height, board_width):
        game = ConnectFour(board_height, board_width)
        fast_game = random.random() > pcr_rate
        base_act_temp = 0.05 if fast_game else 0.8
        n_playout = tiny_playouts if fast_game else large_playouts
        net.load_state_dict(shared_model.state_dict())
        direct_moves = sample_discrete_exp((0.02 if fast_game else 0.04) * board_width * board_width)
        graph = InferenceGraph(net, device, board_height, board_width)
        mcts = MCTS(graph.policy_value_fn, exploration=not fast_game,
                    c_puct=c_puct, c_fpu=c_fpu, n_playout=n_playout)
        history = []
        step = 0
        board_size = game.height * game.width
        while not game.is_terminal():
            state = game.state()
            temp = 1.0 if fast_game else max(1.03, 1.35 * pow(0.66, step / board_size))
            acts, probs = mcts.get_move_probs(game, root_prior_temp=temp)
            acts = np.array(acts, dtype=np.int32)
            data_prob = np.zeros((game.height, game.width), dtype=np.float32)
            data_prob[game.top[acts], acts] = probs

            # drop policy samples for fast games; done by setting mcts_probs to zero
            if fast_game:
                placeholder = np.zeros((game.height, game.width), dtype=np.float32)
                history.append((state, placeholder))
            else:
                history.append((state, data_prob))

            if step < direct_moves:
                acts, probs, _ = graph.policy_value_fn(game)
                action = np.random.choice(acts, p=probs)
            else:
                act_temp = base_act_temp * pow(0.8, (step - 0.5 * direct_moves) / board_width)
                action = np.random.choice(acts, p=apply_temperature(probs, act_temp))
            game.step(action)
            mcts.apply_move(action)
            step += 1
        replay_queue.put((game.height, game.width, history, game.winner))

    selfplay(first_h, first_w)
    selfplay(first_h, first_w)
    while True:
        try:
            selfplay(random.randint(9, 12), random.randint(9, 12))
        except RuntimeError as e:
            if 'out of memory' not in str(e):
                raise
            print(f'[Worker {worker_id}] OOM on GPU {gpu_id}, pausing for 60s')
            torch.cuda.empty_cache()
            time.sleep(60)
            print(f'[Worker {worker_id}] Resuming selfplay on GPU {gpu_id}')

def train():
    # net and device
    device = torch.device('cuda:0')
    net = Net()
    net = net.to(device).train()
    print('Main device:', device)

    # optimizer, scheduler, tensorboard
    def lr_lambda(epoch):
        if epoch < 0.05 * epochs:
            return 1.0
        if epoch < 0.72 * epochs:
            return 3.0
        return 0.3

    optimizer = optim.SGD(net.parameters(), lr=lr/3, momentum=momentum, weight_decay=l2_const)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    writer = SummaryWriter(f'./runs/{run_name}')
    print('Run:', run_name)

    # replay buffer
    buffer = ReplayBuffer(**buffer_config)

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # shared model on CPU
    shared_model = Net()
    shared_model = shared_model.to('cpu').eval()
    shared_model.load_state_dict(net.state_dict())
    shared_model.share_memory()

    # start self-play workers
    mp.set_start_method('spawn')
    replay_queue = mp.Queue()
    workers = [mp.Process(target=selfplay_worker, args=(i, shared_model, replay_queue))
               for i in range(parallel_games)]
    for worker in workers:
        worker.start()

    for epoch in range(1, epochs + 1):
        running_loss = running_entropy = 0.0
        running_ploss = running_vloss = 0.0
        running_episode_len = 0
        iterations = 0

        print('Epoch:', epoch)
        while iterations < epoch_size:
            # move data into buffer
            game = replay_queue.get()
            buffer.push(*game)
            episode_len = len(game[2])

            # sample data
            if not buffer.is_samplable(batch_size):
                continue

            policy_loss = value_loss = entropy = 0.0
            policy_count = 0

            # compute metrics
            optimizer.zero_grad()

            for state_batch, mcts_batch, outcome_batch in buffer.sample(batch_size):
                states = torch.tensor(state_batch, dtype=torch.float32, device=device)
                mcts_probs = torch.tensor(mcts_batch, dtype=torch.float32, device=device).flatten(1)
                outcomes = torch.tensor(outcome_batch, dtype=torch.long, device=device)

                policy_logits, value_logits = net(states)
                policy_logits = policy_logits.flatten(1)
                log_act_probs = F.log_softmax(policy_logits, dim=1)

                # policy loss: leave fast samples unconstrained
                with torch.no_grad():
                    policy_mask = mcts_probs.sum(dim=1).ge(0.9)
                    n_policy = policy_mask.sum().item()
                if n_policy > 0:
                    policy_loss -= torch.sum(mcts_probs[policy_mask] * log_act_probs[policy_mask])
                    policy_count += n_policy

                # value loss: utilize all samples
                value_loss += F.cross_entropy(value_logits, outcomes, reduction='sum')

                # entropy: OK here to keep fast samples; only for monitoring use
                with torch.no_grad():
                    entropy -= torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1).mean()

            if policy_count > 0:
                policy_loss /= policy_count
            value_loss /= batch_size
            entropy /= 16.0

            loss = policy_loss + vloss_scaler * value_loss

            # update model
            loss.backward()
            optimizer.step()
            iterations += 1
            shared_model.load_state_dict(net.state_dict())

            # log metrics
            loss, entropy = loss.item(), entropy.item()
            policy_loss, value_loss = policy_loss.item(), value_loss.item()
            print(f'{iterations:2d}/{epoch_size} episode_len: {episode_len}, '
                f'loss: {loss:.3f}, entropy: {entropy:.3f}, '
                f'policy_loss: {policy_loss:.3f}, value_loss: {value_loss:.3f}')
            running_loss += loss
            running_entropy += entropy
            running_ploss += policy_loss
            running_vloss += value_loss
            running_episode_len += episode_len

        writer.add_scalar('loss', running_loss / iterations, epoch)
        writer.add_scalar('entropy', running_entropy / iterations, epoch)
        writer.add_scalar('loss/policy', running_ploss / iterations, epoch)
        writer.add_scalar('loss/value', running_vloss / iterations, epoch)
        writer.add_scalar('episode_len', running_episode_len / iterations, epoch)
        scheduler.step()

        # save checkpoint
        if epoch % 500 == 0:
            torch.save(net.state_dict(), f'{weights_dir}/katac4_{model_name}_{epoch}.pth')

    torch.save(net.state_dict(), f'{weights_dir}/katac4_{model_name}_final.pth')
    writer.close()
    for worker in workers:
        worker.terminate()

if __name__ == '__main__':
    train()
