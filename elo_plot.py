import json
from collections import defaultdict

import matplotlib.pyplot as plt

elos = defaultdict(list)
with open('elo.json', 'r', encoding='utf-8') as file:
    for player, elo in json.load(file).items():
        run, it = player.split(':')
        elos[run].append((int(it), elo))

for run, ratings in elos.items():
    ratings.sort()
    if ratings[0][0] != 0:
        ratings.insert(0, (0, 0.0))
    epochs, scores = zip(*ratings)
    plt.plot(epochs, scores, marker='o', label=run)

plt.xlabel('Epochs')
plt.ylabel('ELO Rating')
plt.title('ELO Rating Over Time')
plt.grid(True)
plt.legend()
plt.show()
