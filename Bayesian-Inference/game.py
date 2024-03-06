import numpy as np
from copy import deepcopy
import math
import random

from copy import deepcopy

class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        return [self.decode_single_pos(d) for d in self.state]

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive

        TODO: You need to implement this.
        """
        col, row = cr 
        return row * self.N_COLS + col


    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)

        TODO: You need to implement this.
        """
        row = n // self.N_COLS
        col = n % self.N_COLS
        return (col, row)

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.

        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a termainal board state, and return True or False.
        
        TODO: You need to implement this.
        """
        w_col, w_row = self.decode_single_pos(self.state[5])
        b_col, b_row = self.decode_single_pos(self.state[11])
        if b_row == 0 or w_row == self.N_ROWS -1:
            return self.is_valid() 

        return False

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constrainsts that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)
        
        TODO: You need to implement this.

        Constraints considered:
        constraint 1: all pieces + balls on game board (within row,col)
        constraint 2: player must be holding a ball; a ball must be on a player block
        constraint 3: only one player may win at a time
        constraint 4: one space can hold up to one player at a time
        """
        validball1 = False
        validball2 = False
        locationsSet = set()

        for i, v in enumerate(self.state):
            locationsSet.add(v)
            if v >= self.N_COLS * self.N_ROWS or v < 0:
                return False
            if i < 5 and v == self.state[5]:
                validball1 = True
            if i > 6 and i < 11 and v == self.state[11]:
                validball2 = True

        col1, row1 = self.decode_state[5]
        col2, row2 = self.decode_state[11]
        if row2 == 0 and row1 == self.N_ROWS -1:
            return False
        return validball1 and validball2 and len(locationsSet) == 10


class Rules:

    @staticmethod
    def single_piece_actions(board_state, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.
        
        TODO: You need to implement this.
        """
        nextStates = set()
        p_encoded = board_state.state[piece_idx]
        p_col, p_row = board_state.decode_single_pos(p_encoded)

        # check if piece has ball; if so return empty
        if p_encoded == board_state.state[5] or p_encoded == board_state.state[11]:
            return nextStates

        # check all possible moves; add legal moves only
        for i in [-1 ,1]:
            for j in [-2, 2]:
                
                if  0 <= p_col + i and p_col + i < board_state.N_COLS and \
                    0 <= p_row + j and p_row + j < board_state.N_ROWS \
                    and board_state.encode_single_pos((p_col + i, p_row + j)) not in board_state.state:
                    nextStates.add(board_state.encode_single_pos((p_col + i, p_row + j)))

                if  0 <= p_col + j and p_col + j < board_state.N_COLS and \
                    0 <= p_row + i and p_row + i < board_state.N_ROWS \
                    and board_state.encode_single_pos((p_col + j, p_row + i)) not in board_state.state:
                    nextStates.add(board_state.encode_single_pos((p_col + j, p_row + i)))
        return nextStates
    
    @staticmethod
    def single_ball_actions(board_state, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for plater_idx  in the board_state

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        
        TODO: You need to implement this.
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
                        directionr = np.sign(player_row - current_row)
                        directionc = np.sign(player_col - current_col)
                        blocked = False
                        # check if any opponent blocks the pass
                        for other_pos in state[1-player_idx][:5]:
                            other_col, other_row = board_state.decode_single_pos(other_pos)
                            if directionr == np.sign(other_row - current_row) and directionc == np.sign(other_col - current_col) and abs(current_row - other_row) < abs(current_row - player_row) and abs(other_col - current_col) == abs(other_row - current_row): 
                                blocked = True
                                break
                        if not blocked:
                            reachable_players.append(player_pos)
                    # check other players that are in the same column
                    elif abs(player_col - current_col) == 0:
                        direction = np.sign(player_row - current_row)
                        blocked = False
                        # check if any opponent blocks the pass
                        for other_pos in state[1-player_idx][:5]:
                            other_col, other_row = board_state.decode_single_pos(other_pos)
                            if (direction == np.sign(other_row - current_row) \
                                and abs(current_row - other_row) < abs(current_row - player_row)\
                                and abs(other_col - current_col) == 0): 
                                blocked = True
                                break
                        if not blocked:
                            reachable_players.append(player_pos)
                    # check other players that are on the same row   
                    elif abs(player_row - current_row) == 0:
                        direction = np.sign(player_col - current_col)
                        blocked = False
                        # checks if any opponent blocks the pass
                        for other_pos in state[1-player_idx][:5]:
                            other_col, other_row = board_state.decode_single_pos(other_pos)
                            if (direction == np.sign(other_col - current_col) \
                                and abs(current_col - other_col) < abs(current_col - player_col))\
                                and abs(other_row - current_row) == 0:
                                blocked = True
                                break
                        if not blocked:
                            reachable_players.append(player_pos)
                        
                

        # remove the initial ball_pos; not to be apart of potential ball action candidates
        explored_players.remove(ball_pos)
        return explored_players

        

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players,tries_per_round = 7):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players
        self.max_tries = tries_per_round

    def run(self):
        """
        Runs a game simulation
        """
        player_idx = 0
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2


            # get observation
            observation = self.sample_observation((player_idx + 1) % 2)

            tries = 0
            is_valid_action = False
            while (not is_valid_action) and (tries < self.max_tries):
                ## For the player who needs to move, provide them with the current game state
                ## and then ask them to choose an action according to their policy
                action, value = self.players[player_idx].policy( observation )

                try:
                    is_valid_action = self.validate_action(action,player_idx)
                except ValueError:
                    is_valid_action = False
                tries += 1
                print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value} Validity: {is_valid_action}")
                self.players[player_idx].process_feedback(observation, action, is_valid_action)
            if not is_valid_action:
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_sequence(self,N):
        """
        Runs a game simulation
        """
        ground_truth = list()
        obs_sequence = list()
        player_idx = 0
        is_valid_action = True
        while not self.game_state.is_termination_state() and self.current_round < N-2:
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2

            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            
            try:
                is_valid_action = self.validate_action(action,player_idx)
            except ValueError:
                is_valid_action = False

            #print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")
        
            ground_truth.append((tuple(self.game_state.state),action))
            # OBS0 = noisy observations of player 0 pieces
            # OBS1 = noisy observations of player 1 pieces
            observation_0 = self.sample_observation(0)
            observation_1 = self.sample_observation(1)
            decoded_obs = observation_0[0:6] + observation_1[6:12]
            encoded_obs = list()
            for col,row in decoded_obs:
                encoded_obs.append(row*7+col)
            obs_sequence.append(encoded_obs)

            ## If an invalid action is provided, then the other player will be declared the winner
            if not is_valid_action:
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ground_truth.append((tuple(self.game_state.state),None))
        observation_0 = self.sample_observation(0)
        observation_1 = self.sample_observation(1)
        decoded_obs = observation_0[0:6] + observation_1[6:12]
        encoded_obs = list()
        for col,row in decoded_obs:
            encoded_obs.append(row*7+col)
        obs_sequence.append(encoded_obs)
        ## return gt and observations
        return ground_truth, obs_sequence
    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.
            
        TODO: You need to implement this.
        """
        valid_actions = set()

        # state only contains the pieces relevant to player_idx
        state = np.reshape(self.game_state.state, (2,-1))
        ball_pos =  state[player_idx][5]
        player_blocks = state[player_idx]

        for i, block in enumerate(player_blocks):

            new_pos = list()

            if i < 5:
                # evaluate block actions
                new_pos = Rules.single_piece_actions(self.game_state, i+player_idx*6)
            else:
                # evaluate ball actions
                new_pos = Rules.single_ball_actions(self.game_state, player_idx)
    
            for pos in new_pos:
                valid_actions.add((i,pos))

        return valid_actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        
        TODO: You need to implement this.
        """

        valid_actions = self.generate_valid_actions(player_idx)
        relative_idx, encoded_pos = action
        col, row = self.game_state.decode_single_pos(encoded_pos)
        if action not in valid_actions:
            if action[0] == 5:
                # ball move errors
                # Check if ball is out of bounds
                if row < 0 or row >= self.game_state.N_ROWS or col < 0 or col >= self.game_state.N_COLS:
                    raise ValueError("Cannot move ball piece: Position out of bounds")
                else:
                    # check if there is a player on the location
                    if encoded_pos in self.game_state.state[:5]:
                        # if so, ball is unable to be passed to that location.
                        raise ValueError("Cannot move ball piece: Ball cannot reach location")
                    else:
                        # if not, there is an error: player must be at target location
                        raise ValueError("Cannot move ball piece: No player at position")
            else:
                # piece move
                # check if block is out of bounds
                if row < 0 or row >= self.game_state.N_ROWS or col < 0 or col >= self.game_state.N_COLS:
                    raise ValueError("Cannot move block piece: Position out of bounds")
                else:
                    # if so, it must mean that block cannot reach that location: obstructed or illegal move
                    raise ValueError("Cannot move block piece: Block cannot reach location")
                
        else:
            
            return True
        
    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)

    # Used to sample a noisy observation for pieces belong to the player of opposing_idx
    def sample_observation(self, opposing_idx):
        # Get a copy of the decoded state
        noisy_state = self.game_state.make_state()
        index =  6 * opposing_idx
        
        # our_state = noisy_state[6-index:6-index+5]
        our_state = self.game_state.make_state()

        # Track the relative index of the ball holder
        ball_guy_ind = -1

        # Handle player piece observations
        for i in range(index, index+5):
            # check the cardinal directions
            if noisy_state[i] == noisy_state[index+5]:
                ball_guy_ind = i

            # check the directions
            mid = noisy_state[i]
            left = (mid[0]-1, mid[1]) if (mid[0] > 0 and ((mid[0]-1, mid[1]) not in our_state)) else None
            right = (mid[0]+1, mid[1]) if (mid[0] < self.game_state.N_COLS -1 and ((mid[0]+1, mid[1]) not in our_state)) else None
            top  = (mid[0], mid[1]+1) if (mid[1] < self.game_state.N_ROWS - 1 and ((mid[0], mid[1]+1) not in our_state)) else None
            bot  = (mid[0], mid[1]-1) if (mid[1] > 0 and ((mid[0], mid[1]-1) not in our_state)) else None

            # Check the directions with non-none for any collisions with other player pieces
            
            

            # Update the noisy state to account for the noise
            rand = random.random()
            if rand < 0.4:
                if rand < 0.1:
                    if left:
                        noisy_state[i] = left  
                elif rand < 0.2:
                    if right:
                        noisy_state[i] = right
                elif rand < 0.3:
                    if top:
                        noisy_state[i] = top
                elif rand < 0.4:
                    if bot:
                        noisy_state[i] = bot
        
        # Update the ball to match the noisy estimate of the ball holder
        noisy_state[index+5] = noisy_state[ball_guy_ind]
        
        return noisy_state

        


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

    # Same measurement as BallReachable but considers both p layers
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


class Player:
    def __init__(self,policy_fnc):
        self.policy_fnc = policy_fnc
    
    def policy(self,decode_state):
        pass
    
    def process_feedback(self, observation, action, is_valid):
        pass
    

    # additional helper functions used by the adversarial search players
    # Converts a decode_state list to a board_state object
    def state_to_board(self, state):
        board_state = BoardState()
        board_state.state = [board_state.encode_single_pos(d) for d in state]
        board_state.decode_state = state
        return board_state

    # Get the available actions given a state and the current player
    def get_actions(self,state,player_idx):
        sim = GameSimulator(None)
        board_state = BoardState()
        board_state.state = [board_state.encode_single_pos(d) for d in state]
        board_state.decode_state = state
        sim.game_state = board_state
        actions = list(sim.generate_valid_actions(player_idx))
        return actions



class PassivePlayer(Player):
    # invokes super.__init.__ with None
    # get around using self.policy_fnc
    # Player classes created before the doc was updated :p
    def __init__(self, player_idx):
        super().__init__(None)
        self.player_idx = player_idx
        self.invalid_player_location = list()


    # adjust based on the feedback
    # automatically loses if it can't pass the ball
    def process_feedback(self, observation, action, is_valid):

        # If the pass is valid, flush the list for the next move
        if is_valid:
            self.invalid_player_location = list()
        else:
            # if invalid, remove the player from passing considerations
            # 
            _, player_location = action
            self.invalid_player_location.append(player_location)
                        
    
    # PassivePlayer: randomly samples from the list of available passing actions
    # returns the action that is selected (value is set to 0; it is not used)
    def policy(self,state):
    
        board_state =BoardState()
        board_state.state = [board_state.encode_single_pos(d) for d in state]
        board_state.decode_state = state

        all_actions = set()
        start_player_idx = self.player_idx * 6
        i = 0
        
        actions = list(Rules.single_ball_actions(board_state,self.player_idx))
        for action in actions:
            if action not in self.invalid_player_location:
                all_actions.add((5,action))
            
        
        if len(all_actions) > 0:
            action = random.choice(list(all_actions))
        elif len(self.invalid_player_location) < 5:

            # Sample from pieces that are blocked by the observation
            # The pieces that are reachable according to the observation are guaranteed to be
            # illegal here. Might as well try the other pieces.
            
            # randomly select a player that was considered invalid
            for i in range(6):
                ind = i + self.player_idx * 6
                if board_state.state[ind] not in self.invalid_player_location:
                    action = (5,board_state.state[ind]) 
                    break
        else:
            # no possible players to pass it on to;
            # just randomly select an invalid player; we are 100% sure that we can't move
            action = (5,random.choice(self.invalid_player_location))


        # value is arbitrary for the passive player,.
        value = 0
        return action, value
        
class RandomPlayer(Player):

    def __init__(self, player_idx):
        super().__init__(None)
        self.player_idx = player_idx

        self.invalid_player_receivers = list()
        self.invalid_move_locations = list()


    # adjust based on the feedback
    # keep track of invalid receivers and occupied blocks
    def process_feedback(self, observation, action, is_valid):
        
        # flush if the move is valid; moving onto the next turn
        if is_valid:
            self.invalid_player_receivers = list()
            self.invalid_move_locations = list()
        else:
            piece, loc = action            
            # if passing action, update receivers
            # else, update the moving actions 
            if piece == 5:
                self.invalid_player_receivers.append(loc)
            else:
                self.invalid_move_locations.append(loc)


    # RandomPlayer: randomly samples from the list of available actions
    # returns the action that is selected (value is set to 0; it is not used)
    def policy(self, state):
        
        # change enemy observations if there is only one piece adjacent to the observed invalid move
        for loc in self.invalid_move_locations:
            n_cols = 8
            n_rows = 7
            row = loc // n_cols
            col = loc % n_cols

            obs_loc = None
            # check the directions
            left = (col-1, row) if col > 0 else None
            right = (col+1, row) if col < n_cols-1 else None
            top  = (col, row+1) if row < n_rows-1 else None
            bot  = (col, row-1) if row > 0 else None

            count = 0
            for dir in [left,right,top,bot]:
                if dir:
                    # check if the observation has an enemy piece at this location
                    enc = dir[0]*n_cols + dir[1]
                    enemy_piece_idx = None
                    try:
                        enemy_piece_idx = state.index(enc,  (1-self.player_idx) * 6 ,(1-self.player_idx) * 6 + 6)
                    except ValueError:
                        pass
                    if enemy_piece_idx:
                        obs_loc = enemy_piece_idx
                        count += 1
            if count == 1:
                state[obs_loc] = loc
                
        
        actions = self.get_actions(state,self.player_idx)

        # remove pass options that failed in earlier move attempts
        for j in self.invalid_player_receivers:
            actions.remove((5,j))
        # Remove move actions in the case where no enemies were snapped
        for j in self.invalid_move_locations:
            for i in range(5):
                try:
                    actions.remove((i,j))
                except:
                    pass
                
        action = actions[(int)(random.random()*len(actions))]
        value = 0

        return action, value


class MinimaxPlayer(Player):

    # MinimaxPlayer uses depth limited minimax to select moves

    # max_depth limit is set to 3
    # Uses the DualBallReachableHeuristic as its default
    # term_type determines whether to use depth adjusted termination state evaluation
    def __init__(self,player_idx,heuristic= DualBallDistanceHeuristic):
        super().__init__(None)
        self.player_idx = player_idx
        self.max_depth = 3
        self.heuristic = heuristic
        self.term_type = 1
        self.n_best_actions = list()
        self.n_best_actions_state = None
        self.is_first_action = True

        # Used for process feedback
        self.invalid_player_receivers = list()
        self.invalid_move_locations = list()

    # Begins the minimax by invoking maxplayer(). 
    # The Minimax player is assumed to be the max player
    def policy(self,state):
        
        # Snap enemy observations if there is only one piece adjacent to the observed invalid move
        # Spaces in invalid_move_locations must have an enemy piece on it but the observation/state
        # does not show it. Attempt to improve the observation by changing the location of nearby enemy pieces 
        # to that location. If there are multiple enemy pieces adjacent, do not snap anything.
        for loc in self.invalid_move_locations:
            n_cols = 8
            n_rows = 7
            row = loc // n_cols
            col = loc % n_cols

            obs_loc = None
            # check the directions
            left = (col-1, row) if col > 0 else None
            right = (col+1, row) if col < n_cols-1 else None
            top  = (col, row+1) if row < n_rows-1 else None
            bot  = (col, row-1) if row > 0 else None

            count = 0
            for dir in [left,right,top,bot]:
                if dir:
                    # check if the observation has an enemy piece at this location
                    enc = dir[0]*n_cols + dir[1]
                    enemy_piece_idx = None
                    try:
                        enemy_piece_idx = state.index(enc,  (1-self.player_idx) * 6 ,(1-self.player_idx) * 6 + 6)
                    except ValueError:
                        pass
                    if enemy_piece_idx:
                        obs_loc = enemy_piece_idx
                        count += 1
            if count == 1:
                state[obs_loc] = loc

        # If the state/obs changed due to the above or from process_feedback, invoke Maxplayer to start minimax
        if state != self.n_best_actions_state or self.is_first_action:
            self.n_best_actions_state = state
            self.n_best_actions = list()
            res = self.maxplayer(self.state_to_board(state), 0)
            self.n_best_actions.sort()
            return res
        else:
           # If the state did not change due to illegal moves; choose the next best legal action frpm the last
           # run of minimax. This prevents recomputing the same minimax tree.
            chosen_action_value = None
            rand_move = self.n_best_actions[0]
            for action_value in self.n_best_actions:
                relative_idx, new_pos = action_value[0]
                if new_pos not in self.invalid_move_locations and new_pos not in self.invalid_player_receivers:
                    chosen_action_value = action_value
                    break
            
            if chosen_action_value is None:
                # Desparate case where all actions are illegal; Something went wrong
                # Just perform some action since all actions are illegal.
                # Should not occur in minimax since all actions are explored. This would imply that
                # no moves are available.
                return rand_move
            else:
                # Remove the chosen action and return it.
                self.n_best_actions.remove(chosen_action_value)
                return chosen_action_value
            

    # If the last action is invalid, update invalid player receivers for bad passes 
    # and invalid move locations for move actions
    # Observation is updated when policy is invoked.
    def process_feedback(self, observation, action, is_valid):
        # flush if the move is valid; moving onto the next turn
        if is_valid:
            self.is_first_action = True
            self.invalid_player_receivers = list()
            self.invalid_move_locations = list()
        else:
            self.is_first_action = False
            piece, loc = action            
            # if passing action, update receivers
            # else, update the moving actions 
            if piece == 5:
                self.invalid_player_receivers.append(loc)
                # n_best_actions.remove(action)
            else:
                self.invalid_move_locations.append(loc)
                # n_best_actions.remove(action)
    
    # Process the moves of the max player
    # MinimaxPlayer will cycle between maxplayer and minplayer until the run hits a termination state
    def maxplayer(self,state,depth):
        actions = self.get_actions(state.decode_state, self.player_idx)
        current_max = -np.inf
        current_action = None
        next_state = state

        # Check all possible actions
        for action in actions:
            relative_idx, new_pos = action
            old_pos = state.state[(self.player_idx) * 6 + relative_idx]
            next_state.update((self.player_idx) * 6 + relative_idx, new_pos)
            next_value = 0
            
            # Eliminate previously tried invalid actions from consideration by pruning the action branch at
            #   depth 0 during search
            if depth == 0:
                if new_pos in self.invalid_move_locations or new_pos in self.invalid_player_receivers:
                    continue
                
            # if the next_state is a termination state, max player has a winning move
            if next_state.is_termination_state():
                # 1: corresponds to a win for max player
                # 0.1:  used to differentiate between a heuristic value at max depth
                # self.max_depth - depth: grants larger values to closer winning states
                next_value = 1 + (0.1 + self.max_depth - depth)*self.term_type
            else:
                if depth >= self.max_depth:
                    # if at max depth, evaluate the heuristic instead of passing the run off to the min player
                    next_value = self.heuristic.estimate(next_state,self.player_idx)
                else:
                    # check the next level, the min player
                    next_value = self.minplayer(next_state,depth+1)[1]

            if next_value > current_max:
                current_max = next_value
                current_action = action

            # At the first round, keep track of the best actions available
            # Prevent rerunning the entire minimax algorithm if possible if the state/observation
            # remains unchanged after process feecback and the preprocessing done in policy
            if depth == 0:
                self.n_best_actions.append((action, next_value))

            next_state.update((self.player_idx) * 6 + relative_idx, old_pos)
        
        return (current_action, current_max)
        

    # Process the moves of the min player.  
    # MinimaxPlayer will cycle between maxplayer and minplayer until the run hits a termination state
    def minplayer(self,state,depth):
        actions = self.get_actions(state.decode_state, 1-self.player_idx)
        current_min = np.inf
        current_action = None
        next_state = state

        # check all possible actions
        for action in actions:
            relative_idx, new_pos = action     
            old_pos = state.state[(1-self.player_idx) * 6 + relative_idx]
            next_state.update((1-self.player_idx) * 6 + relative_idx, new_pos)
            next_value = 0
            # if the next_state is a termination state, min player has a winning move
            if next_state.is_termination_state():
                # 1: corresponds to a win for min player (loss for max player)
                # 0.1:  used to differentiate between a heuristic value at max depth
                # self.max_depth - depth: grants more values to closer winning states
                # -1 scalar: turns the entire value negative to correspond to min player victory
                next_value = -1*(1 + (0.1 + self.max_depth - depth)*self.term_type)
            else:
                if depth >= self.max_depth:
                    # if at max depth, evaluate the heuristic instead of passing the run off to the max player
                    next_value = self.heuristic.estimate(next_state, self.player_idx)
                else:
                    next_value = self.maxplayer(next_state,depth+1)[1]
            if next_value < current_min:
                current_min = next_value
                current_action = action
            next_state.update((1-self.player_idx) * 6 + relative_idx, old_pos)
        return (current_action, current_min)

class AlphaBetaPlayer(Player):

    # AlphaBetaPlayer implements a depth limited alpha-beta pruning tree

    # max depth limited is set to 5
    # uses the DualBallReachableHeuristic as its default heuristic
    def __init__(self,player_idx,heuristic=DualBallReachableHeuristic):
        # Player.__init__(self,player_idx)
        super().__init__(None)
        self.player_idx = player_idx
        self.max_depth = 3
        self.heuristic = heuristic  

        self.move_heuristic = DualBallReachableHeuristic
        self.move_order = True

        self.term_type = 1

        self.n_best_actions = list()
        self.n_best_actions_state = None
        self.is_first_action = True

        # Used for process feedback
        self.invalid_player_receivers = list()
        self.invalid_move_locations = list()    
    
    def policy(self,state):
        # Perform snapping; same as Minimax
        for loc in self.invalid_move_locations:
            # adjacent_opps = list()
            n_cols = 8
            n_rows = 7
            row = loc // n_cols
            col = loc % n_cols

            obs_loc = None
            # check the directions
            left = (col-1, row) if col > 0 else None
            right = (col+1, row) if col < n_cols-1 else None
            top  = (col, row+1) if row < n_rows-1 else None
            bot  = (col, row-1) if row > 0 else None

            count = 0
            for dir in [left,right,top,bot]:
                if dir:
                    # check if the observation has an enemy piece at this location
                    enc = dir[0]*n_cols + dir[1]
                    enemy_piece_idx = None
                    try:
                        enemy_piece_idx = state.index(enc,  (1-self.player_idx) * 6 ,(1-self.player_idx) * 6 + 6)
                    except ValueError:
                        pass
                    if enemy_piece_idx:
                        obs_loc = enemy_piece_idx
                        count += 1
            if count == 1:
                state[obs_loc] = loc

        
        if state != self.n_best_actions_state or self.is_first_action:
            self.n_best_actions_state = state
            self.n_best_actions = list()
            res = self.maxplayer(self.state_to_board(state), 0, -np.inf, np.inf)
            self.n_best_actions.sort()
            return res
        else:
           # If the state did not change due to illegal moves; choose the next best legal action
            chosen_action_value = None
            rand_move = self.n_best_actions[0]
            for action_value in self.n_best_actions:
                relative_idx, new_pos = action_value[0]
                if new_pos not in self.invalid_move_locations and new_pos not in self.invalid_player_receivers:
                    chosen_action_value = action_value
                    break
            
            if chosen_action_value is None:
                # Desparate case where all actions are illegal; Something went wrong
                # Should never happen; Given an infinite amount of retries, all actions will
                # eventually be exhausted
                return rand_move
            else:
                self.n_best_actions.remove(chosen_action_value)
                return chosen_action_value
            

       
    # Used to process invalid and valid actions; same for minimax and random player
    def process_feedback(self, observation, action, is_valid):
        
        # flush if the move is valid; moving onto the next turn
        if is_valid:
            self.is_first_action = True
            self.invalid_player_receivers = list()
            self.invalid_move_locations = list()
        else:
            self.is_first_action = False
            piece, loc = action            
            # if passing action, update receivers
            # else, update the moving actions 
            if piece == 5:
                self.invalid_player_receivers.append(loc)
                # n_best_actions.remove(action)
            else:
                self.invalid_move_locations.append(loc)
                # n_best_actions.remove(action)
    
    def move_ordering(self,action,state,idx): 
        r_idx, new_pos = action
        old_pos = state.state[(idx) * 6 + r_idx]
        state.update((idx) * 6 + r_idx, new_pos)
        value = self.move_heuristic.estimate(state,idx)
        state.update((idx) * 6 + r_idx, old_pos)
        return value

     


    # maxplayer corresponds to the AlphaBetaPlayer
    def maxplayer(self,state,depth,alpha,beta):
        actions = self.get_actions(state.decode_state, self.player_idx)
        current_max = -np.inf
        current_action = None
        next_state = state
        
        if self.move_order:
            actions.sort(key=lambda action: self.move_ordering(action, next_state,self.player_idx), reverse=True)

        for action in actions:
            relative_idx, new_pos = action
            old_pos = state.state[(self.player_idx) * 6 + relative_idx]
            next_state.update((self.player_idx) * 6 + relative_idx, new_pos)
            next_value = 0

            # If at the root of the tree, ignore moves that go to invalid move positions or
            # pass to invalid recievers
            if depth == 0:
                if new_pos in self.invalid_move_locations or new_pos in self.invalid_player_receivers:
                    continue

            if next_state.is_termination_state():
                # same as maxplayer in MiniMaxPlayer
                next_value = 1 + (0.1 + self.max_depth - depth)*self.term_type
            else:
                if depth >= self.max_depth:
                    next_value = self.heuristic.estimate(next_state,self.player_idx)
                else:
                    next_value = self.minplayer(next_state,depth+1, alpha, beta)[1]
            if next_value > current_max:
                current_max = next_value
                current_action = action
            if depth == 0:
                self.n_best_actions.append((action, next_value))
            if next_value >= beta:
                return (current_action, current_max)
            alpha = max(alpha,current_max)
            next_state.update((self.player_idx) * 6 + relative_idx, old_pos)
            
        return (current_action, current_max)
        

    # minplayer corresponds the opponent of the AlphaBetaPlayer
    def minplayer(self,state,depth,alpha,beta):
        actions = self.get_actions(state.decode_state, 1-self.player_idx)
        current_min = np.inf
        current_action = None
        next_state = state

        if self.move_order:
            actions.sort(key=lambda action: self.move_ordering(action, next_state,1-self.player_idx), reverse=True)

        for action in actions:
            relative_idx, new_pos = action     
            old_pos = state.state[(1-self.player_idx) * 6 + relative_idx]
            next_state.update((1-self.player_idx) * 6 + relative_idx, new_pos)
            next_value = 0
            if next_state.is_termination_state():
                # same as minplayer in MinimaxPlayer
                next_value = -1*(1 + (0.1 + self.max_depth - depth)*self.term_type)
            else:
                if depth >= self.max_depth:
                    next_value = self.heuristic.estimate(next_state, self.player_idx)
                else:
                    next_value = self.maxplayer(next_state,depth+1, alpha, beta)[1]
            if next_value < current_min:
                current_min = next_value
                current_action = action
            if next_value <= alpha:
                return (current_action, current_min)
            beta = min(beta,current_min)
            next_state.update((1-self.player_idx) * 6 + relative_idx, old_pos)
        return (current_action, current_min)
        
class MonteCarloPlayer(Player):

    # Node class: used to model the Search Tree
    class Node:
        # state: the state vector, containing the positions of pieces and ball
        # parent: the parent node if applicable
        # idx: the node's corresponding player (max or min)
        def __init__(self,state,parent,idx):
            self.state = state          # state: state vector
            self.value = 0              # value: the win ratio of the node so far 
            self.wins = 0               # wins: the number of times the max player has won from rollout
            self.games = 0              # games: the number of times the node has been explored           
            self.parent = parent        # parent: the parent node (None if root)
            self.children = list()      # children: contains the children nodes (indexes correspond to actions list)            
            self.actions = list()       # actions: contain the actions taken to the children
            self.idx = idx              # idx: the player_idx of the node


    def __init__(self, player_idx,heuristic=DualBallReachableHeuristic):
        super().__init__(None)
        self.player_idx = player_idx
        self.root = None                # root: tracks the root state
        self.iterations = 200           # iterations: the number of state selection -> rollout-> backprop performed  
        self.c = 4*np.sqrt(2)             # c: the exploration constant of UCT/UCB1 selection heuristic       
        self.max_rollout = 75
        self.rollout_est = heuristic

        # Used for process feedback
        self.invalid_player_receivers = list()
        self.invalid_move_locations = list() 
    
    
    # Same function as defined in RandomPlayer, MinimaxPlayer, and AlphaBetaPlayer
    def process_feedback(self, observation, action, is_valid):
        
        # flush if the move is valid; moving onto the next turn
        if is_valid:
            self.is_first_action = True
            self.invalid_player_receivers = list()
            self.invalid_move_locations = list()
        else:
            self.is_first_action = False
            piece, loc = action            
            # if passing action, update receivers
            # else, update the moving actions 
            if piece == 5:
                self.invalid_player_receivers.append(loc)
                # n_best_actions.remove(action)
            else:
                self.invalid_move_locations.append(loc)
                # n_best_actions.remove(action)
    
    def policy(self,state):
        # For each invalid move recorded for the current observation,
        # attempt to move an adjacent enemy piece to that location. 
        # If there are multiple enemy pieces, do not perform the snapping
        # Same implementation as seen in Minimax and Alphabeta players
        for loc in self.invalid_move_locations:
            n_cols = 8
            n_rows = 7
            row = loc // n_cols
            col = loc % n_cols

            obs_loc = None
            # check the directions
            left = (col-1, row) if col > 0 else None
            right = (col+1, row) if col < n_cols-1 else None
            top  = (col, row+1) if row < n_rows-1 else None
            bot  = (col, row-1) if row > 0 else None

            count = 0
            for dir in [left,right,top,bot]:
                if dir:
                    # check if the observation has an enemy piece at this location
                    enc = dir[0]*n_cols + dir[1]
                    enemy_piece_idx = None
                    try:
                        enemy_piece_idx = state.index(enc,  (1-self.player_idx) * 6 ,(1-self.player_idx) * 6 + 6)
                    except ValueError:
                        pass
                    if enemy_piece_idx:
                        obs_loc = enemy_piece_idx
                        count += 1
            if count == 1:
                state[obs_loc] = loc

        # Update self.root to node that corresponds to the new state
        # if the node exists on the current tree at self.root, point self.root to that node.
        # MonteCarloPlayer can take advantage of previous rollouts instead of starting from scratch
        tree_exists = False
        if self.root is None:
            # initial state/ first state of the game evaluated by the MonteCarloPlayer
            self.root = self.Node(deepcopy(state),None,self.player_idx)
        else:
            found = False

            # Check if the root state is the same as the arg; if so, this means we are re-selecting an action
            if self.root.parent.state == state:
                self.root = self.root.parent
                self.root.parent = None
                found = True
                tree_exists = True
            else:
                # Check if the children of self.root contains state; if not recreate tree at state
                for child in self.root.children:
                    if child.state == state:
                        self.root = child
                        self.root.parent = None
                        found = True
                        break
                    
            # Most of the time, this option is used.
            if not found:
                self.root = self.Node(deepcopy(state),None,self.player_idx)
        
        # self.root now corresponds to state

        # perform self.iterations iterations of expansion, rollout, and backprop
        # Expansion + rollout + backprop
        num_iter  = 0
        while num_iter < self.iterations and not tree_exists: 

            # state selection
            current = self.root
            action_for_expansion = None
            # Search for a leaf node
            while current.children:
                actions = self.get_actions(current.state, current.idx)

                # Select action and resulting node that produces the highest UCT value
                best_score = -np.inf
                selected_action = None
                selected_node = None
                for action in actions:
                    if current.children == self.root:
                        if action[1] in self.invalid_move_locations or action[1] in self.invalid_player_receivers:
                            continue

                    current_score = 0
                    temp_node = current # defaulted to current node in the case of unexplored actions
                    if action in current.actions:
                        # handles explored actions
                        node_idx = current.actions.index(action)
                        temp_node = current.children[node_idx]
                        maxmin = 0 if current.idx == self.player_idx else 1
                        current_score = (-1 ** maxmin) * temp_node.value + self.c * np.sqrt(np.log(current.games)/temp_node.games)
                    else:
                        # handles unexplored actions (node not in tree)
                        # unexplored actions are treated as having infinity as its UCT value
                        current_score = np.inf
                        
                    
                    if current_score > best_score:
                        best_score = current_score
                        selected_action = action
                        selected_node = temp_node
                

                

                # if the selected_node is equal to current, 
                # it means that selected action leads to the new node to be added.
                # Mark action for explansion as the selected action
                if selected_node == current:
                    action_for_expansion = selected_action
                    break

                current = selected_node

            # for leaf nodes; randomly choose actions as UCT values for all possible actions must be inf/equal
            if action_for_expansion is None:
                actions = self.get_actions(current.state, current.idx)
                action_for_expansion = random.choice(actions)
            
            # Create the new node/expanded state and add it to the tree
            selected_board_state = self.state_to_board(deepcopy(current.state))
            selected_board_state.update(6*current.idx+action_for_expansion[0], action_for_expansion[1])
            expanded_state = selected_board_state.decode_state
            expanded_node = self.Node(expanded_state,current, 1-current.idx)
            current.actions.append(action_for_expansion)
            current.children.append(expanded_node)
            current = expanded_node
            
            # current is new node without any children/explored actions; 
            # perform rollout on this node

            # Rollout: performs random rollout
            idx = current.idx
            cur_state = deepcopy(current.state)
            is_terminal = False
            won = False
            iter = 0
            # print("before rollout "+str(time.time_ns()))
            while not is_terminal and iter < self.max_rollout :
                # pick random move for each player, until termination                
                actions = self.get_actions(cur_state, idx)
                r_action = random.choice(actions)
                cur_board_state = self.state_to_board(cur_state)
                cur_board_state.update(6*idx+r_action[0], r_action[1])
                cur_state = cur_board_state.decode_state
                idx = 1 - idx
                if cur_board_state.is_termination_state():
                    is_terminal = True
                    if idx != self.player_idx:
                        won = True
                    else:
                        won = False
                iter += 1

                if iter == self.max_rollout:
                    temp = self.rollout_est.estimate(self.state_to_board(cur_state), self.player_idx)
                    if temp > 0:
                        won = True
                    else:
                        won = False
            # print("after rollout_ " +str(time.time_ns()))
            # backprop the win after rollout is done
            # update the win rate up the tree
            while current is not None:
                current.games += 1
                current.wins = current.wins + 1 if won else current.wins
                current.value = current.wins/ current.games
                current = current.parent
            
            num_iter += 1


        # extract the best action by evaluating actions and children at the root
        # Ensure that it excludes piece moves that move to locations in self.invalid_move_location and
        # ball moves that end up in players in self.invalid_player_receiv
        best_action = None
        best_value = -np.inf
        best_child = None
        for child, action in zip(self.root.children,self.root.actions):
            if child.value > best_value and action[1] not in self.invalid_player_receivers \
                and action[1] not in self.invalid_move_locations:
                
                best_action = action
                best_value = child.value
                best_child = child
        
        
        # Handle case where none of the explored actions are valid
        # Rare case: Solve by randomly selecting actions
        if best_action is None:
            all_actions = self.get_actions(self.root.state, self.player_idx)
            for action in all_actions:
                if action not in self.root.actions and action[1] not in self.invalid_player_receivers \
                    and action[1] not in self.invalid_move_locations:

                    best_action = action
                    best_value = 0
                    best_child = self.Node(None,self.root,1-self.root.idx)
                    


        # reorient the root to point to the best_child
        # While this is a min node, the next call of policy will reorient root back to
        # a max node.
        
        # self.root.parent = None
        self.root = best_child
        
        # assert(self.root.idx == 1-self.player_idx)
        return (best_action, best_value)
