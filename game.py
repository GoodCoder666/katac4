import numpy as np
import random
import secrets

ZOBRIST_TABLE = {
    1: [[secrets.randbits(128) for _ in range(12)] for _ in range(12)],
    -1: [[secrets.randbits(128) for _ in range(12)] for _ in range(12)],
}

class ConnectFour:
    def __init__(self, height=None, width=None, forbidden_point=None):
        self.height = height or random.randint(9, 12)
        self.width = width or random.randint(9, 12)
        self.forbidden_point = forbidden_point or \
            (random.randint(0, self.height-1), random.randint(0, self.width-1))
        self.board = np.zeros((self.height, self.width), dtype=np.float32)
        self.top = np.zeros(self.width, dtype=np.int32)
        if self.forbidden_point[0] == 0:
            self.top[self.forbidden_point[1]] = 1
        self.player = 1
        self.winner = None
        self.board[self.forbidden_point] = -2
        self.last_opp_move = None
        self.last_ths_move = None
        self._hash = 0

    @property
    def hash(self):
        return self._hash

    def state(self):
        state = np.zeros((6, self.height, self.width), dtype=np.float32)
        state[0, :, :] = (self.board == self.player).astype(np.float32)
        state[1, :, :] = (self.board == -self.player).astype(np.float32)
        mask = self.top < self.height
        state[2, self.top[mask], np.nonzero(mask)[0]] = 1.0
        state[3, :, :] = 1.0
        state[3, *self.forbidden_point] = 0.0
        if self.last_opp_move:
            state[4, *self.last_opp_move] = 1
        if self.last_ths_move:
            state[5, *self.last_ths_move] = 1
        return state

    def _winning_moves(self, player, candidates):
        winning_moves = []
        for col in candidates:
            row = self.top[col]
            self.board[row, col] = player
            if self.check_win(row, col):
                winning_moves.append(col)
            self.board[row, col] = 0
        return winning_moves

    def sensible_moves(self):
        candidates = np.where(self.top < self.height)[0].astype(np.int32)
        if ths_win := self._winning_moves(self.player, candidates):
            return np.array(ths_win, dtype=np.int32)
        if opp_win := self._winning_moves(-self.player, candidates):
            return np.array(opp_win, dtype=np.int32)
        return candidates

    def _count(self, r, c, dr, dc):
        player = self.board[r, c]
        count = 0
        while 0 <= r < self.height and 0 <= c < self.width and self.board[r, c] == player:
            r, c, count = r + dr, c + dc, count + 1
        return count

    def check_win(self, r, c):
        count2 = lambda dr, dc: self._count(r, c, dr, dc) + self._count(r, c, -dr, -dc) - 1
        return count2(1, 1) >= 4 or count2(1, -1) >= 4 or \
            count2(0, 1) >= 4 or self._count(r, c, -1, 0) >= 4

    def is_terminal(self):
        return self.winner is not None

    def step(self, col):
        if self.top[col] == self.height:
            raise ValueError('Invalid move')
        row = self.top[col]
        self.board[row, col] = self.player
        self._hash ^= ZOBRIST_TABLE[self.player][row][col]
        self.last_ths_move = self.last_opp_move
        self.last_opp_move = (row, col)
        self.top[col] += 2 if (row + 1, col) == self.forbidden_point else 1
        if self.check_win(row, col):
            self.winner = self.player
        elif np.all(self.top == self.height):
            self.winner = 0
        self.player *= -1

    def __str__(self):
        marks = np.full((self.height, self.width), '-')
        marks[self.board == 1] = 'X'
        marks[self.board == -1] = 'O'
        marks[self.forbidden_point] = '*'
        id_row = [str(i % 10) for i in range(self.width)]
        return '\n'.join(' '.join(row) for row in [*marks[::-1], id_row])
