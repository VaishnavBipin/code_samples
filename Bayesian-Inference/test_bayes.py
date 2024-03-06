from game import RandomPlayer, PassivePlayer, MinimaxPlayer, AlphaBetaPlayer, MonteCarloPlayer, DualBallReachableHeuristic, BallReachableHeuristic, DualBallDistanceHeuristic
from game import BoardState, GameSimulator, Rules
from util import GameRunner
import pytest
import random
import itertools
import numpy as np
from copy import deepcopy


# Generates Sequences of length up to N
# Simulates and records a game between two random players
# If the game terminates earlier than N, then the sequence length is less than N
def generate_sequence(N,init_state=None):
    zero = RandomPlayer(0)
    one = RandomPlayer(1)
    players = [zero,one]
    game = GameSimulator(players)
    if init_state is not None:
        game.game_state.state = init_state
        game.game_state.decoded_state = game.game_state.make_state()
    gt,obs = game.generate_sequence(N)
    return gt,obs


# Same as generate_sequence but ensures that the observation/ground truth
# sequence has a length of N exactly.
def generate_sequence_exact(N,init_state=None):

    zero = RandomPlayer(0)
    one = RandomPlayer(1)
    players = [zero,one]
    game = GameSimulator(players)
    if init_state is not None:
        game.game_state.state = [v for v in init_state]
        game.game_state.decoded_state = game.game_state.make_state()
    gt,obs = game.generate_sequence(N)

    while len(obs) != N:
        game = GameSimulator(players)
        if init_state is not None:
            game.game_state.state = [v for v in init_state]
            game.game_state.decoded_state = game.game_state.make_state()
        gt,obs = game.generate_sequence(N)
    return gt,obs


# gt,obs = generate_sequence(5)
# print(gt)
# print(obs)


# Gets the difference between the observation/inferred state
# Returns the number of pieces that are not the same.
def get_diff(inferred,state):
    # piece_idx = [0,1,2,3,4, 6,7,8,9,10]
    dist = 0
    for p_idx in range(12):
        if inferred[p_idx] != state[p_idx]:
            dist += 1
    return dist

# Helper function used to test if an decoded location is within bounds
def in_bounds(obs, N_ROWS=8, N_COLS=7):
    return (obs[0] >= 0) and (obs[0] < N_COLS) and (obs[1] < N_ROWS) and (obs[1] >= 0)   

# Helper function used to return the decoded positions reachable by enc, the encoded starting position
def get_valid_piece_moves(enc, N_COLS=7):
    col = enc % N_COLS
    row = enc // N_COLS
    
    # Generate possible states that enc can reach
    prev_states = [(col, row), (col+2, row+1), (col+2, row-1), (col-2, row+1), (col-2, row-1), (col+1, row+2), (col+1, row-2), (col-1, row+2), (col-1, row-2)]
    
    # Return reachable states that are inbound
    valid_piece_states = list()
    for state in prev_states:
        if in_bounds(state):
            enc = state[1]*N_COLS + state[0]
            valid_piece_states.append(enc)
    return valid_piece_states
            
        
# Initialize the translation matrix (fixed for a given board size)
# Each element (row i, col j) corresponds to the probability of a piece transitioning to encoded 
#    position i at time t given that it is at encoded position j at time t-1. 
#       i.e. T(i,j) = p(x_t = i | x_t-1 = j)
#    This structure is useful as it allows for a belief distribution vector of size (N_COLS * N_ROWS, 1)
#    to have its next state predicted via matrix multiplication (bel' = T * bel)
# The probabilities are determined by assuming pieces at any location moves uniformly over the set of 
#    possible actions that are possible.
def init_trans_mat(N_ROWS=8, N_COLS=7):
    trans_mat = np.zeros((N_ROWS*N_COLS, N_ROWS*N_COLS))
    # Row i corresponds to the position we are trying to get to
    for i in range(N_COLS * N_ROWS):
    
        # previously doing premature normalization
        for location in get_valid_piece_moves(i, N_COLS):
            moves = get_valid_piece_moves(location, N_COLS)
            w = 1 / len(moves) # Guaranteed to be >= 1; it is reachable from iS
            trans_mat[i,location] += w

    return trans_mat

# Generate an initial belief of uniform distribution.
# This belief is then refined by the inital observation to generate the refined uniform initial distribution.
#   e.g. Each location on the board is given an initial belief of 1/56 in an 8 x 7 board
def get_init_belief(N_ROWS=8, N_COLS=7):
    return np.full(N_COLS * N_ROWS, 1/(N_COLS * N_ROWS))

# Takes in a belief and updates the observation
# This is done piece by piece where bel and obs correspond to one piece.
def refine_belief_with_obs(bel, obs, N_ROWS=8, N_COLS=7):
    p_obs_given_x = np.zeros(N_COLS * N_ROWS)
    col = obs % N_COLS
    row = obs // N_COLS
    mid = (col, row)
    left = (mid[0]-1, mid[1]) if (mid[0] > 0) else None
    right = (mid[0]+1, mid[1]) if (mid[0] < N_COLS - 1) else None
    top  = (mid[0], mid[1]+1) if (mid[1] < N_ROWS - 1) else None
    bot  = (mid[0], mid[1]-1) if (mid[1] > 0) else None
    
    direction  = [left,right,top,bot]
    for dir in direction:
        if dir:
            enc = dir[0] + dir[1]*N_COLS
            p_obs_given_x[enc] = 0.1
        else:
            p_obs_given_x[obs] += 0.1
    p_obs_given_x[obs] += 0.6

    return p_obs_given_x * bel

# Normalizes the belief by dividing each entry by the total weight of the belief
def normalize_belief(bel, N_ROWS=8, N_COLS=7):
    total_w = np.sum(bel)

    # Shouldn't be possible if observations are rightly considered (i.e. all possible
    # ground truth states are allocated probabilities; e.g. this is not the case if 
    # Dirac delta is used during any observation based refining steps)
    if total_w == 0:
        # print("observation killed all belief probabilities")
        return get_init_belief(N_ROWS, N_COLS)

    return bel / total_w 

# Helper function used to return the index of the item
# If item appears in multiple entries, it will return the indices of those entries
def find_idxs_of_item(arr, item, offset):
    a = list(np.where(np.array(arr) == item)[0])
    for i in range(len(a)):
        a[i] += offset
    return a


# Used to determine who is the ball player in a given board observation
def select_ball_player(offset, board_obs, bel_list):
    # Find all indices/players that match the observed location of the ball
    # There may be multiple players that share the ball in the observation due to noise allowing overlap
    b_p_idxs = find_idxs_of_item(board_obs[0+offset:5+offset], board_obs[5+offset], offset)
    chosen_p = None
    max_prob = -np.inf
    # Choose the player that has the highest probability at being at the observed location of the ball
    for idx in b_p_idxs:
        prob_p_at_b1 = bel_list[idx][board_obs[5+offset]]
        if prob_p_at_b1 > max_prob:
            max_prob = prob_p_at_b1
            chosen_p = idx
    return chosen_p 

# Given a sequence of observations, determine the last state in that sequence
# Returns the final estimate
def infer_last_state_from_seq(obs_seq, select_dist_val='mode', init_dist='refined_uniform', N_ROWS=8, N_COLS=7):
   
    p_idxs = [0,1,2,3,4, 6,7,8,9,10]
    bel_list = [None] * 12
    trans_mat = init_trans_mat()

    if init_dist == 'refined_uniform':
        # Uniform belief
        for p in range(12):
            bel = get_init_belief()
            bel_list[p] = bel

        # Refine uniform with inital observation
        for p in p_idxs:

            board_obs = obs_seq[0]
            bel_list[p] = refine_belief_with_obs(bel_list[p], board_obs[p])
            bel_list[p] = normalize_belief(bel_list[p])

            bel_list[5] = bel_list[select_ball_player(0, board_obs, bel_list)]
            bel_list[11] = bel_list[select_ball_player(6, board_obs, bel_list)]

    elif init_dist == 'dirac_delta':
        # All observations are believed to be exact. Probability of 1 at the observed location
        for p in range(12):
            bel_list[p] = np.zeros(N_COLS * N_ROWS)
            bel_list[p][obs_seq[0][p]] = 1
    # Process the observations 
    for board_obs in obs_seq[1:]:
        for p in p_idxs:
            bel_list[p] = np.matmul(trans_mat, bel_list[p])
        
        
        for p in p_idxs:
            bel_list[p] = refine_belief_with_obs(bel_list[p], board_obs[p])
            bel_list[p] = normalize_belief(bel_list[p])

            bel_list[5] = bel_list[select_ball_player(0, board_obs, bel_list)]
            bel_list[11] = bel_list[select_ball_player(6, board_obs, bel_list)]

    # Estimate the final state
    final_estimate = None
    if select_dist_val == 'mode':
        final_estimate = [np.argmax(bel_list[p]) for p in range(12)]
    elif select_dist_val == 'sample':
        final_estimate = [None] * 12
        for p in range(12):
            rand = random.random()
            for enc,prob in enumerate(bel_list[p]):
                rand -= prob
                if rand < 0:
                    final_estimate[p] = enc
    
    return final_estimate

# Given a sequence of observation, predict the sequence of actions taken.
def infer_actions_from_seq(obs_seq, select_dist_val='mode', init_dist='refined_uniform', use_active_prob=0,N_ROWS=8,N_COLS=7):
   
    p_idxs = [0,1,2,3,4, 6,7,8,9,10]
    bel_list = [None] * 12
    actions = list()
    trans_mat = init_trans_mat()

    if init_dist == 'refined_uniform':
        # Uniform belief
        for p in range(12):
            bel = get_init_belief()
            bel_list[p] = bel

        # Refine uniform with inital observation
        for p in p_idxs:

            board_obs = obs_seq[0]
            bel_list[p] = refine_belief_with_obs(bel_list[p], board_obs[p])
            bel_list[p] = normalize_belief(bel_list[p])

            bel_list[5] = bel_list[select_ball_player(0, board_obs, bel_list)]
            bel_list[11] = bel_list[select_ball_player(6, board_obs, bel_list)]

    elif init_dist == 'dirac_delta':
        for p in range(12):
            bel_list[p] = np.zeros(N_COLS * N_ROWS)
            bel_list[p][obs_seq[0][p]] = 1


    for board_obs in obs_seq[1:]:

        bel_list_prev = deepcopy(bel_list)
        for p in p_idxs:
            bel_list[p] = np.matmul(trans_mat, bel_list[p])
        
        
        for p in p_idxs:
            bel_list[p] = refine_belief_with_obs(bel_list[p], board_obs[p])
            bel_list[p] = normalize_belief(bel_list[p])

            bel_list[5] = bel_list[select_ball_player(0, board_obs, bel_list)]
            bel_list[11] = bel_list[select_ball_player(6, board_obs, bel_list)]
        
        prev_estimate = None
        if select_dist_val == 'mode':
            prev_estimate = [np.argmax(bel_list_prev[p]) for p in range(12)]
        elif select_dist_val == 'sample':
            prev_estimate = [None] * 12
            for p in range(12):
                rand = random.random()
                for enc,prob in enumerate(bel_list_prev[p]):
                    rand -= prob
                    if rand < 0:
                        prev_estimate[p] = enc
                        
        # Compare the two beliefs to extract the action distribution
        # Compute all possible locations of the action (move/pass) and evaluate those locations for
        # the belief at the next time step. Select the action with the greatest probability of occurring
        best_valid_actions = list()
        for p_idx,(bel_1,bel_2) in enumerate(zip(bel_list_prev,bel_list)):
            best_valid_action = None
            max_action_prob = -np.inf
            
           
            for enc_1,prob_1 in enumerate(bel_1):
                next_states = []
                if p_idx == 5 or p_idx == 11:
                    inferred_pos = prev_estimate[p_idx] # temp
                    prev_estimate[p_idx] = enc_1
                    temp = BoardState()
                    temp.state = prev_estimate
                    temp.decode_state = temp.make_state()
                    next_states = Rules.single_ball_actions(temp, p_idx//6)
                    prev_estimate[p_idx] = inferred_pos
                    
                else:
                    inferred_pos = prev_estimate[p_idx] # temp
                    prev_estimate[p_idx] = enc_1
                    temp = BoardState()
                    temp.state = prev_estimate
                    temp.decode_state = temp.make_state()
                    next_states = Rules.single_piece_actions(temp, p_idx)
                    prev_estimate[p_idx] = inferred_pos
                    
                for enc_2 in next_states:
                    prob_2 = bel_2[enc_2]
                    if prob_1*prob_2 > max_action_prob:
                        best_valid_action = (p_idx%6, enc_2)
                        max_action_prob = prob_1*prob_2

            inactive_prob = np.dot(bel_1,bel_2)
            active_prob = 1 - inactive_prob
            if use_active_prob == 1:
                best_valid_actions.append((active_prob*max_action_prob, best_valid_action))
            else:
                best_valid_actions.append((max_action_prob, best_valid_action))

        best_valid_actions.sort(reverse=True)
        best_action = best_valid_actions[0]    
        
        actions.append(best_action[1])

    return actions
        
def generate_matrix(MM_params, AB_params, MC_params, n_trials):
    success_mat = np.array([[0,0,0],[0,0,0],[0,0,0]])
    rounds_mat = np.array([[0,0,0],[0,0,0],[0,0,0]])
    for i in range(3):
        for j in range(3):
            # print((i,j))
            # if i == j or i > j:
            #     continue
            # if not ((i == 2) or (j == 2)):
            #     continue
            player_0 = None
            player_1 = None
            trials = n_trials
            if i == 0:
                player_0 = MinimaxPlayer(0, DualBallReachableHeuristic)
                player_0.max_depth = MM_params
            elif i == 1:
                player_0 = AlphaBetaPlayer(0, DualBallReachableHeuristic)
                player_0.max_depth = AB_params[0]
                player_0.move_order = AB_params[1]
            else:
                player_0 = MonteCarloPlayer(0)
                player_0.c = MC_params[0]
                player_0.iterations = MC_params[1]
                
            if j == 0:
                player_1 = MinimaxPlayer(1, DualBallReachableHeuristic)
                player_1.max_depth = MM_params
            elif j == 1:
                player_1 = AlphaBetaPlayer(1, DualBallReachableHeuristic)
                player_1.max_depth = AB_params[0]
                player_1.move_order = AB_params[1]
            else:
                player_1 = MonteCarloPlayer(1)
                player_1.c = MC_params[0]
                player_1.iterations = MC_params[1]
            
            game = GameRunner(player_0,player_1,trials)
            game.run()
            results = game.get_results()
            # print(results)
            winrate_p0 = results[4]/(results[4]+results[3])
            success_mat[i,j] = winrate_p0

            rounds_mat[i,j] = sum(results[0])/n_trials
    return success_mat, rounds_mat
# Pre-generated ground truths and observations 
# done with generate_sequence and used for two tests:
# test_infer_last_state_from_seq
# test_infer_actions_from_seq

gt1 = [((1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52), (5, 2)), ((1, 2, 3, 4, 5, 2, 50, 51, 52, 53, 54, 52), (3, 38)), ((1, 2, 3, 4, 5, 2, 50, 51, 52, 38, 54, 52), (2, 16)), ((1, 2, 16, 4, 5, 2, 50, 51, 52, 38, 54, 52), (0, 37)), ((1, 2, 16, 4, 5, 2, 37, 51, 52, 38, 54, 52), None)]
obs1 = [[1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52], [1, 2, 3, 4, 6, 2, 49, 51, 52, 53, 54, 52], [1, 2, 3, 4, 6, 2, 50, 44, 52, 38, 55, 52], [0, 9, 16, 11, 5, 9, 49, 51, 52, 38, 54, 52], [0, 3, 16, 3, 5, 3, 37, 50, 52, 38, 47, 52]]

gt2 = [((29, 30, 12, 46, 25, 29, 27, 18, 47, 45, 23, 27), (1, 17)), ((29, 17, 12, 46, 25, 29, 27, 18, 47, 45, 23, 27), (2, 52)), ((29, 17, 12, 46, 25, 29, 27, 18, 52, 45, 23, 27), (4, 40)), ((29, 17, 12, 46, 40, 29, 27, 18, 52, 45, 23, 27), (5, 45)), ((29, 17, 12, 46, 40, 29, 27, 18, 52, 45, 23, 45), None)]
obs2 = [[28, 30, 12, 46, 26, 28, 27, 18, 40, 44, 16, 27], [29, 17, 5, 46, 32, 29, 26, 18, 48, 45, 23, 26], [22, 17, 12, 46, 25, 22, 27, 18, 52, 45, 23, 27], [36, 24, 12, 46, 41, 36, 27, 18, 52, 45, 23, 27], [22, 17, 12, 46, 40, 22, 27, 18, 51, 45, 23, 45]]

gt3 = [((23, 7, 30, 27, 5, 5, 8, 41, 39, 24, 17, 24), (2, 15)), ((23, 7, 15, 27, 5, 5, 8, 41, 39, 24, 17, 24), (1, 32)), ((23, 7, 15, 27, 5, 5, 8, 32, 39, 24, 17, 24), (3, 18)), ((23, 7, 15, 18, 5, 5, 8, 32, 39, 24, 17, 24), (1, 41)), ((23, 7, 15, 18, 5, 5, 8, 41, 39, 24, 17, 24), None)]
obs3 = [[23, 7, 30, 27, 6, 6, 1, 48, 40, 24, 17, 24], [22, 7, 22, 27, 5, 5, 8, 41, 38, 24, 17, 24], [23, 7, 15, 27, 5, 5, 9, 31, 39, 24, 17, 24], [23, 7, 15, 11, 5, 5, 8, 32, 39, 24, 17, 24], [16, 14, 15, 18, 12, 12, 8, 41, 40, 24, 16, 24]]

gt4 = [((1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52), (5, 4)), ((1, 2, 3, 4, 5, 4, 50, 51, 52, 53, 54, 52), (1, 46)), ((1, 2, 3, 4, 5, 4, 50, 46, 52, 53, 54, 52), (0, 10)), ((10, 2, 3, 4, 5, 4, 50, 46, 52, 53, 54, 52), (4, 39)), ((10, 2, 3, 4, 5, 4, 50, 46, 52, 53, 39, 52), None)]
obs4 = [[1, 2, 3, 4, 12, 3, 50, 51, 52, 53, 54, 52], [1, 9, 10, 11, 12, 11, 50, 51, 52, 53, 54, 52], [1, 2, 3, 4, 5, 4, 50, 46, 52, 53, 54, 52], [10, 2, 3, 4, 5, 4, 50, 47, 52, 53, 54, 52], [11, 2, 3, 4, 5, 4, 43, 46, 52, 53, 39, 52]]

    

class TestBayes:

# test_infer_last_state_from_seq
    @pytest.mark.parametrize("gt_seq, obs_seq",[
        (gt1,obs1),
        (gt2,obs2),
        (gt3,obs3)
    ])
    def test_infer_last_state_from_seq(self,gt_seq,obs_seq):
        
        final_estimate = infer_last_state_from_seq(obs_seq=obs_seq)
        final_gt = gt_seq[len(gt_seq)-1][0]
        final_obs = obs_seq[len(obs_seq)-1]
        # Our estimate should not be worse than the observation
        # assert(get_diff(final_estimate,final_gt) <= get_diff(final_obs,final_gt))


# test_infer_actions_from_seq
    @pytest.mark.parametrize("gt_seq, obs_seq",[
        (gt1,obs1),
        (gt2,obs2),
        (gt3,obs3)
    ])
    def test_infer_actions_from_seq(self,gt_seq,obs_seq):
        
        actions = infer_actions_from_seq(obs_seq=obs_seq)
        # Measure accuracy of the action estimation
        correct = 0
        for i in range(len(actions)):
            if actions[i] == gt_seq[i][1]:
                correct += 1
        correct /= len(actions)
        # assert(correct >= 0.5)
    
    # Evaluates average accuracy over sequences of varying lengths
    def test_1_obs_lengths_infer_state(self):
        avg_accuracy = list()
        obs_lens = list(itertools.chain(range(1,10,1),range(10,30,2),range(30,70,4),range(70,200,8)))
        n_samples = 100
        for obs_len in obs_lens:
            avg = 0
            for i in range(n_samples):
                gt_seq, obs_seq = generate_sequence_exact(obs_len)
                final_estimate = infer_last_state_from_seq(obs_seq=obs_seq)
                final_gt = gt_seq[len(gt_seq)-1][0]
                num_diff = get_diff(final_estimate,final_gt)
                avg += (12 - num_diff) / 12
            avg /= n_samples
            avg_accuracy.append(avg)
        
                
    # Evaluates average accuracy over sequences of varying lengths
    def test_1_obs_lengths_infer_actions(self):
        avg_accuracy = list()
        obs_lens = list(itertools.chain(range(2,10,1),range(10,30,2),range(30,70,4),range(70,200,8)))
        n_samples = 5
        for obs_len in obs_lens:
            avg = 0
            for i in range(n_samples):
                gt_seq, obs_seq = generate_sequence_exact(obs_len)
                actions = infer_actions_from_seq(obs_seq=obs_seq)
                correct = 0
                for i in range(len(actions)):
                    if actions[i] == gt_seq[i][1]:
                        correct += 1
                correct /= len(actions)
                avg += correct
            avg /= n_samples
            avg_accuracy.append(avg)
    
    # Measures the impact of the initial distribution on the performance of the estimation
    # Baseline uses refined uniform. This test will try dirac delta distribution for the initial
    def test_2_init_dist_infer_state(self):
        avg_accuracy = list()
        obs_lens = list(itertools.chain(range(1,10,1),range(10,30,2),range(30,70,4),range(70,200,8)))
        n_samples = 100
        for obs_len in obs_lens:
            avg = 0
            for i in range(n_samples):
                gt_seq, obs_seq = generate_sequence_exact(obs_len)
                final_estimate = infer_last_state_from_seq(obs_seq=obs_seq,init_dist="dirac_delta")
                final_gt = gt_seq[len(gt_seq)-1][0]
                num_diff = get_diff(final_estimate,final_gt)
                avg += (12 - num_diff) / 12
            avg /= n_samples
            avg_accuracy.append(avg)
        
    def test_2_init_dist_infer_actions(self):
        avg_accuracy = list()
        obs_lens = list(itertools.chain(range(2,10,1),range(10,30,2),range(30,70,4),range(70,200,8)))
        n_samples = 5
        for obs_len in obs_lens:
            avg = 0
            for i in range(n_samples):
                gt_seq, obs_seq = generate_sequence_exact(obs_len)
                actions = infer_actions_from_seq(obs_seq=obs_seq,init_dist="dirac_delta")
                correct = 0
                for i in range(len(actions)):
                    if actions[i] == gt_seq[i][1]:
                        correct += 1
                correct /= len(actions)
                avg += correct
            avg /= n_samples
            avg_accuracy.append(avg)
        
    # Measures the impact of the state/action selection process on the performance of the estimation
    # Baseline uses the mode to select the value. This test will sample states/actions
    def test_3_select_val_infer_state(self):
        avg_accuracy = list()
        obs_lens = list(itertools.chain(range(1,10,1),range(10,30,2),range(30,70,4),range(70,200,8)))
        n_samples = 100
        for obs_len in obs_lens:
            avg = 0
            for i in range(n_samples):
                gt_seq, obs_seq = generate_sequence_exact(obs_len)
                final_estimate = infer_last_state_from_seq(obs_seq=obs_seq,select_dist_val="sample")
                final_gt = gt_seq[len(gt_seq)-1][0]
                num_diff = get_diff(final_estimate,final_gt)
                avg += (12 - num_diff) / 12
            avg /= n_samples
            avg_accuracy.append(avg)

    def test_3_select_val_infer_actions(self):
        avg_accuracy = list()
        obs_lens = list(itertools.chain(range(2,10,1),range(10,30,2),range(30,70,4),range(70,200,8)))
        n_samples = 5
        for obs_len in obs_lens:
            avg = 0
            for i in range(n_samples):
                gt_seq, obs_seq = generate_sequence_exact(obs_len)
                actions = infer_actions_from_seq(obs_seq=obs_seq,select_dist_val="sample")
                correct = 0
                for i in range(len(actions)):
                    if actions[i] == gt_seq[i][1]:
                        correct += 1
                correct /= len(actions)
                avg += correct
            avg /= n_samples
            avg_accuracy.append(avg)
    
    # Test the impact of inactive transition probability on selecting actions
    # baseline does not do this
    def test_4_active_prob_infer_actions(self):
        avg_accuracy = list()
        obs_lens = list(itertools.chain(range(2,10,1),range(10,30,2),range(30,70,4),range(70,200,8)))
        n_samples = 5
        for obs_len in obs_lens:
            avg = 0
            for i in range(n_samples):
                gt_seq, obs_seq = generate_sequence_exact(obs_len)
                actions = infer_actions_from_seq(obs_seq=obs_seq,use_active_prob=1)
                correct = 0
                for i in range(len(actions)):
                    if actions[i] == gt_seq[i][1]:
                        correct += 1
                correct /= len(actions)
                avg += correct
            avg /= n_samples
            avg_accuracy.append(avg)

    
    # Compare the performances of each of the agents against each other
    # Used to compute the values used in Q2 of the report
    def test_generate_matrix(self):
        success_mat, rounds_mat = generate_matrix((3), (3,1), (4*2**0.5, 200), 1) # takes 15 minutes
