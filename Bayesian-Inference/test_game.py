import time
import numpy as np
from copy import deepcopy
from game import BoardState, GameSimulator, Rules, RandomPlayer, PassivePlayer, MinimaxPlayer, AlphaBetaPlayer, MonteCarloPlayer, DualBallReachableHeuristic, BallReachableHeuristic, DualBallDistanceHeuristic
import pytest


class Timer():

    def __init__(self):
        self.start = None
        self.end = None

    def startTime(self):
        self.start = time.perf_counter()
    
    def endTime(self):
        self.end = time.perf_counter()

    def time(self):
        if self.start is None or self.end is None:
            return 0.0
        else:
            return self.end - self.start 
    def reset(self):
        self.start = None
        self.end = None



class GameRunner():

    # initialize the players
    def __init__(self, player0, player1, trials=10, state=np.array([1,2,3,4,5,3,50,51,52,53,54,52])):
        self.player_0 = player0
        self.player_1 = player1
        self.trials = trials
        self.state = state

        # Used to track performance values
        # accumulated across multiple calls of run
        self.rounds = list()
        self.times = list()
        self.results = list()
        self.one_winner = 0
        self.zero_winner = 0

        # Used to determine if player1 meets the metrics (consistently wins)
        self.winrate = .80
        

    # Runs the game self.trials times
    def run(self):

        game_time = Timer()
        for i in range(self.trials):
            # make deep copy since MonteCarloPlayer object maintains a tree throughout policy calls
            # ensures that MonteCarloPlayer has an empty tree
            zero = deepcopy(self.player_0)
            one = deepcopy(self.player_1)
            players = [zero,one]

            game = GameSimulator(players)
            game.game_state.state = deepcopy(self.state)
            game.game_state.decoded_state = game.game_state.make_state()
            game_time.startTime()
            round_count, winner, err = game.run()
            game_time.endTime()
            
            self.times.append(game_time.time())
            self.rounds.append(round_count)
            game_time.reset()

            if winner == "WHITE":
                self.zero_winner += 1
                print("Player 0 wins")
                self.results.append(0)
            elif winner == "BLACK":
                self.one_winner += 1
                print("Player 1 wins")
                self.results.append(1)
        print("GameRunner done")
    
    # returns the raw data to work with
    def get_results(self):
        if not self.rounds or not self.times:
            raise RuntimeError("Need to run GameRunner.run() first")
        else:
            return self.rounds, self.times,self.results, self.one_winner, self.zero_winner
        

    # used for the matrix values for q3
    def get_metrics(self):
        if not self.rounds or not self.times:
            raise RuntimeError("Need to run GameRunner.run() first")
        else:
            total_games = self.one_winner + self.zero_winner
            total_round_count = sum(self.rounds)
            total_time = sum(self.times)

            zerowinrate = self.zero_winner/total_games
            onewinrate = self.one_winner/total_games
            time_per_round = total_time / total_round_count
            rounds_per_game = sum(self.rounds)/ len(self.rounds)


            return zerowinrate, onewinrate, time_per_round, rounds_per_game
    

    # Returns useful information about the runs
    def interpret_results(self):
        # Win rate
        total_games = self.one_winner + self.zero_winner

        print(f"Player 0 has won {self.zero_winner} games (win rate:{self.zero_winner/total_games})")
        print(f"Player 1 has won {self.one_winner} games (win rate:{self.one_winner/total_games})")
        print(self.results)
        # Average Time per round
        total_round_count = sum(self.rounds)
        total_time = sum(self.times)
        print(f"Average Time spent per round (both players): {total_time/total_round_count}")


        # Average Round per game
        print(f"Average # of rounds per game: {total_round_count / len(self.rounds)}")
        print(self.rounds)

        # Average Time per game
        print(f"Average time spent per game: {total_time / len(self.times)}")
        print(self.times)

        # Does Player one pass the threshold
        # if self.one_winner/total_games >= self.winrate:
        #     print(f"Player 1 has passed the threshold:  winrate = {self.one_winner/total_games} >= {self.winrate}")
        #     return True
        # else:
        #     print(f"Player 1 does not meet the win rate threshold:  winrate = {self.one_winner/total_games} < {self.winrate}")
        #     return False


class TestGame:

    # for each minimax test:
    # test increasing depth +1
    # test decreasing depth -1
    # test normal baseline
    # all against  random player
    @pytest.mark.parametrize("depth",[(4),(2),(3)])
    def test_minimax_a(self,depth):
        player_0 = RandomPlayer(0)
        player_1 = MinimaxPlayer(1,DualBallDistanceHeuristic)
        player_1.max_depth = depth
        game = GameRunner(player_0,player_1,5,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()
    
    @pytest.mark.parametrize("depth",[(4),(2),(3)])
    def test_minimax_b(self,depth):
        player_0 = MinimaxPlayer(0,DualBallDistanceHeuristic)
        player_1 = RandomPlayer(1)
        player_0.max_depth = depth
        game = GameRunner(player_0,player_1,5,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()

    @pytest.mark.parametrize("depth",[(4),(2),(3)])
    def test_minimax_c(self,depth):
        player_0 = RandomPlayer(0)
        player_1 = MinimaxPlayer(1,DualBallDistanceHeuristic)
        player_1.max_depth = depth
        game = GameRunner(player_0,player_1,5)
        game.run()
        game.interpret_results()
    


    # for each alphabeta test:
    # test depth increase/decrease
    # test move ordering  vs non move ordering
    # test baseline
    @pytest.mark.parametrize("depth,move_order",[
       (2, 1),(3, 1),(3, 0),(4,1)
    ])
    def test_alpha_a(self, depth, move_order):
        player_0 = RandomPlayer(0)
        player_1 = AlphaBetaPlayer(1, DualBallReachableHeuristic)
        player_1.max_depth = depth
        player_1.move_order = move_order
        game = GameRunner(player_0,player_1,10,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()
    
    @pytest.mark.parametrize("depth,move_order",[
       (2, 1),(3, 1),(3, 0),(4,1)
    ])
    def test_alpha_b(self, depth, move_order):
        player_0 = AlphaBetaPlayer(0, DualBallReachableHeuristic)
        player_1 = RandomPlayer(1)
        player_0.max_depth = depth
        player_0.move_order = move_order
        game = GameRunner(player_0,player_1,10,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()
    
    @pytest.mark.parametrize("depth,move_order",[
       (2, 1),(3, 1),(3, 0),(4,1)
    ])
    def test_alpha_c(self, depth, move_order):
        player_0 = RandomPlayer(0)
        player_1 = AlphaBetaPlayer(1, DualBallReachableHeuristic)
        player_1.max_depth = depth
        player_1.move_order = move_order
        game = GameRunner(player_0,player_1,10)
        game.run()
        game.interpret_results()

    
    
    # for each monte carlo test:
    # test exploration constant c:  2sqrt(2) , 4sqrt(2), 6sqrt(2)
    # test num iterations: +- 100 iterations: 150,200,250
    @pytest.mark.parametrize("c, iterations",[
        (2*2**0.5, 200), 
        (4*2**0.5, 150), (4*2**0.5, 200), (4*2**0.5, 250),
        (6*2**0.5, 200)
    ])
    def test_monte_carlo_a(self,c,iterations):
        player_0 = RandomPlayer(0)
        player_1 = MonteCarloPlayer(1)
        player_1.c = c
        player_1.iterations = iterations
        game = GameRunner(player_0,player_1,5,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()

    @pytest.mark.parametrize("c, iterations",[
        (2*2**0.5, 200), 
        (4*2**0.5, 150), (4*2**0.5, 200), (4*2**0.5, 250),
        (6*2**0.5, 200)
    ])
    def test_monte_carlo_b(self,c,iterations):
        player_0 = MonteCarloPlayer(0)
        player_1 = RandomPlayer(1)
        player_0.c = c
        player_0.iterations = iterations
        game = GameRunner(player_0,player_1,5,state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
        game.run()
        game.interpret_results()

    @pytest.mark.parametrize("c, iterations",[
        (2*2**0.5, 200), 
        (4*2**0.5, 150), (4*2**0.5, 200), (4*2**0.5, 250),
        (6*2**0.5, 200)
    ])
    def test_monte_carlo_c(self,c,iterations):
        player_0 = RandomPlayer(0)
        player_1 = MonteCarloPlayer(1)
        player_1.c = c
        player_1.iterations = iterations
        game = GameRunner(player_0,player_1,5)
        game.run()
        game.interpret_results()