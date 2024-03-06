import time
import numpy as np
from copy import deepcopy
from player import RandomPlayer, PassivePlayer, MinimaxPlayer, AlphaBetaPlayer, MonteCarloPlayer
from heuristic import DualBallReachableHeuristic, BallReachableHeuristic, DualBallDistanceHeuristic
from game import BoardState, GameSimulator, Rules
from util import GameRunner
import pytest




# player_0 = RandomPlayer(0)
# player_1 = AlphaBetaPlayer(1, DualBallReachableHeuristic)

# game = GameRunner(player_0,player_1)
# game.run()
# game.interpret_results()

class TestAdversarial:

    # for each minimax test:
    # test increasing depth +1
    # test decreasing depth -1
    # test depth evaluation methods
    # test normal baseline
    # all against  random player
    @pytest.mark.parametrize("depth,term_type",[(4,1),(2,1),(3,0),(3,1)])
    def test_minimax_a(self,depth,term_type):
        player_0 = RandomPlayer(0)
        player_1 = MinimaxPlayer(1,DualBallReachableHeuristic)
        player_1.max_depth = depth
        player_1.term_type = term_type
        game = GameRunner(player_0,player_1,10,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()
    
    @pytest.mark.parametrize("depth,term_type",[(4,1),(2,1),(3,0),(3,1)])
    def test_minimax_b(self,depth,term_type):
        player_0 = MinimaxPlayer(0,DualBallReachableHeuristic)
        player_1 = RandomPlayer(1)
        player_0.max_depth = depth
        player_0.term_type = term_type
        game = GameRunner(player_0,player_1,10,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()

    @pytest.mark.parametrize("depth,term_type",[(4,1),(2,1),(3,0),(3,1)])
    def test_minimax_c(self,depth,term_type):
        player_0 = RandomPlayer(0)
        player_1 = MinimaxPlayer(1,DualBallReachableHeuristic)
        player_1.max_depth = depth
        player_1.term_type = term_type
        game = GameRunner(player_0,player_1,10)
        game.run()
        game.interpret_results()
    


    # for each alphabeta test:
    # test depth increase/decrease
    # test move ordering  vs non move ordering
    # test depth evaluation
    # test baseline
    @pytest.mark.parametrize("depth,term_type,move_order",[
       (2, 1, 1),(3, 0, 1),(3, 1, 0),(3, 1, 1),(4, 1, 1)
    ])
    def test_alpha_a(self, depth, term_type, move_order):
        player_0 = RandomPlayer(0)
        player_1 = AlphaBetaPlayer(1, DualBallReachableHeuristic)
        player_1.max_depth = depth
        player_1.term_type = term_type
        player_1.move_order = move_order
        game = GameRunner(player_0,player_1,10,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()
    
    @pytest.mark.parametrize("depth,term_type,move_order",[
       (2, 1, 1),(3, 0, 1),(3, 1, 0),(3, 1, 1),(4, 1, 1)
    ])
    def test_alpha_b(self, depth, term_type, move_order):
        player_0 = AlphaBetaPlayer(0, DualBallReachableHeuristic)
        player_1 = RandomPlayer(1)
        player_0.max_depth = depth
        player_0.term_type = term_type
        player_0.move_order = move_order
        game = GameRunner(player_0,player_1,10,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()
    
    @pytest.mark.parametrize("depth,term_type,move_order",[
       (2, 1, 1),(3, 0, 1),(3, 1, 0),(3, 1, 1),(4, 1, 1)
    ])
    def test_alpha_c(self, depth, term_type, move_order):
        player_0 = RandomPlayer(0)
        player_1 = AlphaBetaPlayer(1, DualBallReachableHeuristic)
        player_1.max_depth = depth
        player_1.term_type = term_type
        player_1.move_order = move_order
        game = GameRunner(player_0,player_1,10)
        game.run()
        game.interpret_results()

    
    
    # for each monte carlo test:
    # test exploration constant c: 1/2 sqrt(2)
    # test num iterations: +- 100 iterations 
    @pytest.mark.parametrize("c, iterations",[
        (2**0.5, 100), (2**0.5, 200), (2**0.5, 300),
        (2*2**0.5, 100), (2*2**0.5, 200), (2*2**0.5, 300),
        (0.5*2**0.5, 100), (0.5*2**0.5, 200), (0.5*2**0.5, 300)
    ])
    def test_monte_carlo_a(self,c,iterations):
        player_0 = RandomPlayer(0)
        player_1 = MonteCarloPlayer(1)
        player_1.c = c
        player_1.iterations = iterations
        game = GameRunner(player_0,player_1,10,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()

    @pytest.mark.parametrize("c, iterations",[
        (2**0.5, 100), (2**0.5, 200), (2**0.5, 300),
        (2*2**0.5, 100), (2*2**0.5, 200), (2*2**0.5, 300),
        (0.5*2**0.5, 100), (0.5*2**0.5, 200), (0.5*2**0.5, 300)
    ])
    def test_monte_carlo_b(self,c,iterations):
        player_0 = MonteCarloPlayer(0)
        player_1 = RandomPlayer(0)
        player_0.c = c
        player_0.iterations = iterations
        game = GameRunner(player_0,player_1,10,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()

    @pytest.mark.parametrize("c, iterations",[
        (2**0.5, 100), (2**0.5, 200), (2**0.5, 300),
        (2*2**0.5, 100), (2*2**0.5, 200), (2*2**0.5, 300),
        (0.5*2**0.5, 100), (0.5*2**0.5, 200), (0.5*2**0.5, 300)
    ])
    def test_monte_carlo_c(self,c,iterations):
        player_0 = RandomPlayer(0)
        player_1 = MonteCarloPlayer(1)
        player_1.c = c
        player_1.iterations = iterations
        game = GameRunner(player_0,player_1,10)
        game.run()
        game.interpret_results()