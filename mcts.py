import copy
import math

import numpy as np


class TreeNode:
    __slots__ = ('children', 'N', 'Q', 'P')

    def __init__(self, prior):
        self.children = {}
        self.N = 0
        self.Q = 0
        self.P = prior

    def select(self, c_puct, c_fpu, forced_k=None):
        if forced_k and self.N > 1:
            total_visits = self.N - 1
            forced_child = next((
                (act, child) for act, child in self.children.items()
                if child.N ** 2 < forced_k * child.P * total_visits
            ), None)
            if forced_child:
                return forced_child
        p_explored = sum(
            child.P
            for child in self.children.values() if child.N > 0
        )
        fpu_penalty = c_fpu * math.sqrt(p_explored)
        def uct(edge):
            _, child = edge
            edge_Q = -child.Q if child.N else self.Q - fpu_penalty
            return edge_Q + c_puct * child.P * math.sqrt(self.N) / (1 + child.N)
        return max(self.children.items(), key=uct)

    def expand(self, actions, probs, dirichlet_alpha=None):
        self.children = {action: TreeNode(prob)
                         for action, prob in zip(actions.tolist(), probs.tolist())}
        if dirichlet_alpha is not None:
            noise_values = np.random.dirichlet(np.full(len(self.children), dirichlet_alpha))
            for (_, child), noise in zip(self.children.items(), noise_values):
                child.P = 0.75 * child.P + 0.25 * float(noise)

    def update(self, value):
        self.N += 1
        self.Q += (value - self.Q) / self.N


class MCTS:
    __slots__ = ('c_puct', 'c_fpu', 'policy', 'root', 'exploration', 'n_playout')

    def __init__(self, policy_value_fn, exploration=False, c_puct=1.0, c_fpu=0.2, n_playout=1600):
        self.c_puct = c_puct
        self.c_fpu = c_fpu
        self.n_playout = n_playout
        self.policy = policy_value_fn
        self.exploration = exploration
        self.root = TreeNode(1.0)

    def _playout(self, state):
        state = copy.deepcopy(state)
        node = self.root
        path = [node]
        if self.exploration:
            action, node = node.select(self.c_puct, 0, forced_k=2.0)
            state.step(action)
            path.append(node)
        while node.children:
            action, node = node.select(self.c_puct, self.c_fpu)
            state.step(action)
            path.append(node)
        if state.is_terminal():
            value = float(state.winner * state.player)
        else:
            acts, probs, value = self.policy(state)
            node.expand(acts, probs)
        for node in reversed(path):
            node.update(value)
            value = -value

    def _get_pruned_policy(self):
        max_visits, best_q, best_p = max((node.N, -node.Q, node.P)
                                         for node in self.root.children.values())
        best_puct = best_q + self.c_puct * best_p * math.sqrt(self.root.N) / (1 + max_visits)
        total_visits = self.root.N - 1

        def get_pruned_visits(child):
            N = child.N
            if N == max_visits:
                return N
            T = child.P * math.sqrt(self.root.N) / (best_puct + child.Q) - 1
            n_forced = math.floor(math.sqrt(2.0 * child.P * total_visits))
            N = max(N - n_forced, min(N, math.ceil(T)))
            return 0 if N == 1 else N

        acts, children = zip(*self.root.children.items())
        visits = np.array(list(map(get_pruned_visits, children)), dtype=np.float32)
        return acts, visits / visits.sum()

    def get_move_probs(self, state, root_prior_temp=1.0):
        # root_prior_temp works only in exploration games
        if self.exploration:
            acts, probs, value = self.policy(state, root_prior_temp)
            self.root.expand(acts, probs, dirichlet_alpha=0.8)
            self.root.update(value)
        for _ in range(self.n_playout):
            self._playout(state)
        if self.exploration:
            return self._get_pruned_policy()
        act_visits = [(act, node.N) for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        visits = np.array(visits, dtype=np.float32)
        act_probs = visits / visits.sum()
        return acts, act_probs

    def apply_move(self, move):
        if not self.exploration and move in self.root.children:
            self.root = self.root.children[move]
            self.root.P = 1.0
        else:
            self.reset()

    def reset(self):
        self.root = TreeNode(1.0)
