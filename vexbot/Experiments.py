import numpy as np
from card import Suit, Card, Deck, Hand, handtype_to_str
import random
from copy import deepcopy
from player import Player, RandomPlayer, RaisePlayer, VexBot
from game import MatchSimulator



trials = 25
winner_list = list()
for i in range(trials):
    players = [RaisePlayer(0), VexBot(1)]
    sim = MatchSimulator(players, n_games=100,initial_players_chips=[2000, 2000])
    winner,game, chips = sim.run()
    winner_list.append(winner)
