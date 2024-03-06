
from game import BoardState, GameSimulator, Rules, RandomPlayer, MonteCarloPlayer, MinimaxPlayer, AlphaBetaPlayer, PassivePlayer, DualBallReachableHeuristic, BallReachableHeuristic, DualBallDistanceHeuristic
from test_game import GameRunner









player_1 = RandomPlayer(0)
player_2 = AlphaBetaPlayer(1)
player_2.max_depth = 4
game = GameRunner(player_1, player_2, 1, state=[1, 28, 17, 9, 10, 17, 39, 51, 52, 53, 2, 52])
game.run()
game.interpret_results()

# for c in [6]:
#     player_0 = MonteCarloPlayer(0)
#     player_0.c = c*2**0.5
#     player_0.iterations = 250
#     player_1 = RandomPlayer(1)
#     game = GameRunner(player_0,player_1,10)
#     game.run()
#     game.interpret_results()
    


# success_mat = [[0,0,0],[0,0,0],[0,0,0]]
# for i in range(3):
#     for j in range(3):
#         player_0 = None
#         player_1 = None
#         if i == 0:
#             player_0 = MinimaxPlayer(0, DualBallDistanceHeuristic)
#         elif i == 1:
#             player_0 = AlphaBetaPlayer(0, DualBallDistanceHeuristic)
#         else:
#             player_0 = MonteCarloPlayer(0)
        
#         if j == 0:
#             player_1 = MinimaxPlayer(1, DualBallDistanceHeuristic)
#         elif j == 1:
#             player_1 = AlphaBetaPlayer(1, DualBallDistanceHeuristic)
#         else:
#             player_1 = MonteCarloPlayer(1)
        
#         game = GameRunner(player_0,player_1,10)
#         game.run()
