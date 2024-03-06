
class Heuristic:
    # Base Class for Heuristic
    # Defaults to 0
    @staticmethod
    def estimate(state,player_idx):
        return 0


class DualBallDistanceHeuristic(Heuristic):
    
    # Gets the distance from the start to ball and 
    # subtract it by the distance from the goal to opponent ball
    # The heuristic values states where the ball is closer to the goal 
    # as well as states where the ball is further away
    
    # state: the Boardstate object of the current state to be evaluated
    # player_idx: the index of our player; the max player
    @staticmethod
    def estimate(state, player_idx):
        
        state_vec = state.state
        # get ball_dist, the distance from start to ball
        _, ball_pos_row = state.decode_single_pos(state_vec[player_idx*6+5]) 
        ball_start_row = player_idx*state.N_ROWS
        ball_dist = abs(ball_start_row - ball_pos_row) 

        # get opponent_dist, the distance from opponent start row to opponent ball
        _, opponent_ball_row = state.decode_single_pos(state_vec[(1-player_idx) * 6 + 5])
        opponent_ball_start_row = (1-player_idx)*state.N_ROWS
        opponent_dist = abs(opponent_ball_start_row - opponent_ball_row)
        
        # Normalized it such that h is [-1,1]
        h = (ball_dist - opponent_dist) / state.N_ROWS
        return h


class BallReachableHeuristic(Heuristic):

    # Helper method to get the reachable players (for passes)
    # Ignores opponent blocks that would impede the pass
    @staticmethod
    def get_reachable(board_state, player_idx):
        """
        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        """
        # explored set
        explored_players = set()
        # horizon
        reachable_players = list()

        # limit state to hold the relevant pieces depending on the player
        state = np.reshape(board_state.state, (2,-1))
        ball_pos =  state[player_idx][5] 
        # add initial ball position to get the loop started; will remove when returned
        reachable_players.append(ball_pos)
        while reachable_players:
            item = reachable_players.pop(0)    
            current_col, current_row = board_state.decode_single_pos(item)
            explored_players.add(item)
            for player_pos in state[player_idx][:5]: 
                if player_pos not in explored_players:
                    player_col, player_row = board_state.decode_single_pos(player_pos)

                    # check other players that are in the diagonals
                    if abs(player_col - current_col) == abs(player_row - current_row):
                        reachable_players.append(player_pos)
                    # check other players that are in the same column
                    elif abs(player_col - current_col) == 0:
                        reachable_players.append(player_pos)
                    # check other players that are on the same row   
                    elif abs(player_row - current_row) == 0:
                        reachable_players.append(player_pos)
                        
                

        # Keep ball_pos instead of removing it; still need to check it for the heuristic
        return explored_players


    # This Heuristic returns the distance from the furthest ball reachable player to the goal / 2
    # The lower bound on the number of turns to get a player in the goal
    # -1 is appended to simulate the turn needed to pass the goal\
    # Essentially, Better Heuristic returns the lower bound on turns to 
    # reach the goal <ignores opponent for blocked pieces>
    @staticmethod
    def estimate(state, player_idx):

        reachable_players = BallReachableHeuristic.get_reachable(state,player_idx)
        furthest_player_row = None
        distance = 0

        if player_idx:
            # top player
            # select smallest player (value is the lowest, thus closest to goal)
            _, furthest_player_row = state.decode_single_pos(min(reachable_players))
            distance = math.floor((state.N_ROWS - furthest_player_row) / 2) - 1
        else:
            # bot player
            # select the largest player
            _, furthest_player_row = state.decode_single_pos(max(reachable_players))
            distance = math.floor((furthest_player_row) / 2) - 1

        # distance in [-1,3]. normalize it to [-1,1]
        h = (distance - 1) / 2
        return h





class DualBallReachableHeuristic(BallReachableHeuristic):

    # Same measurement as BallReachable but considers both players
    # invokes BallReachableHeuristic on the state for both players
    # subtract the heuristic value of the player by the heuristic value of the opponent
    # Factors in both opponent and player position instead
    # of being limited to player position only

    @staticmethod
    def estimate(state,player_idx):
        player_val = BallReachableHeuristic.estimate(state,player_idx)
        opp_val = BallReachableHeuristic.estimate(state, 1-player_idx)
        # Subtract and normalize the difference
        h = (player_val - opp_val) / 2
        return h
