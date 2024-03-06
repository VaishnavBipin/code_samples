from game import RandomPlayer, PassivePlayer, MinimaxPlayer, AlphaBetaPlayer, MonteCarloPlayer
from heuristic import DualBallReachableHeuristic, BallReachableHeuristic, DualBallDistanceHeuristic
from game import BoardState, GameSimulator, Rules
from util import GameRunner
from copy import deepcopy
import numpy as np



def in_bounds(obs, N_ROWS=8, N_COLS=7):
    return (obs[0] >= 0) and (obs[0] < N_COLS) and (obs[1] < N_ROWS) and (obs[1] >= 0)   

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


def get_init_belief(N_ROWS=8, N_COLS=7):
    return np.full(N_COLS * N_ROWS, 1/(N_COLS * N_ROWS))

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

def normalize_belief(bel, N_ROWS=8, N_COLS=7):
    total_w = np.sum(bel)

    # shouldn't be possible
    if total_w == 0:
        print("observation killed all belief probabilities")
        return get_init_belief(N_ROWS, N_COLS)

    return bel / total_w 

def find_idxs_of_item(arr, item, offset):
    a = list(np.where(np.array(arr) == item)[0])
    for i in range(len(a)):
        a[i] += offset
    return a

def select_ball_player(offset, board_obs, bel_list):
    b_p_idxs = find_idxs_of_item(board_obs[0+offset:5+offset], board_obs[5+offset], offset)
    chosen_p = None
    max_prob = -np.inf
    for idx in b_p_idxs:
        prob_p_at_b1 = bel_list[idx][board_obs[5+offset]]
        if prob_p_at_b1 > max_prob:
            max_prob = prob_p_at_b1
            chosen_p = idx
    return chosen_p 

N_ROWS = 8
N_COLS = 7

#gt_seq = [((1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52), (5, 2)), ((1, 2, 3, 4, 5, 2, 50, 51, 52, 53, 54, 52), (3, 38)), ((1, 2, 3, 4, 5, 2, 50, 51, 52, 38, 54, 52), (2, 16)), ((1, 2, 16, 4, 5, 2, 50, 51, 52, 38, 54, 52), (0, 37)), ((1, 2, 16, 4, 5, 2, 37, 51, 52, 38, 54, 52), None)]
#obs_seq = [[1, 2, 3, 4, 5, 3, 50, 51, 52, 53, 54, 52], [1, 2, 3, 4, 6, 2, 49, 51, 52, 53, 54, 52], [1, 2, 3, 4, 6, 2, 50, 44, 52, 38, 55, 52], [0, 9, 3, 11, 5, 9, 49, 51, 52, 38, 54, 52], [0, 3, 16, 3, 5, 3, 37, 50, 52, 38, 47, 52]]
# (final obs): (0, 3, 16, 3, 5, 3, 37, 50, 52, 38, 47, 52)
# final gt:    (1, 2, 16, 4, 5, 2, 37, 51, 52, 38, 54, 52)
# predicted:   (0,   2,   16,  4,   5,   _,   37,  51,  52,  38,  54,  _)
# probs:        0.82 0.76 1.0  0.76 0.92 0.02 0.83 0.71 1.0  0.99 0.63 0.02

#gt_seq = [((29, 30, 12, 46, 25, 29, 27, 18, 47, 45, 23, 27), (1, 17)), ((29, 17, 12, 46, 25, 29, 27, 18, 47, 45, 23, 27), (2, 52)), ((29, 17, 12, 46, 25, 29, 27, 18, 52, 45, 23, 27), (4, 40)), ((29, 17, 12, 46, 40, 29, 27, 18, 52, 45, 23, 27), (5, 45)), ((29, 17, 12, 46, 40, 29, 27, 18, 52, 45, 23, 45), None)]
#obs_seq = [[28, 30, 12, 46, 26, 28, 27, 18, 40, 44, 16, 27], [29, 17, 5, 46, 32, 29, 26, 18, 48, 45, 23, 26], [22, 17, 12, 46, 25, 22, 27, 18, 52, 45, 23, 27], [36, 24, 12, 46, 41, 36, 27, 18, 52, 45, 23, 27], [22, 17, 12, 46, 40, 22, 27, 18, 51, 45, 23, 45]]

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

gt_seq,obs_seq = generate_sequence(200)

p_idxs = [0,1,2,3,4, 6,7,8,9,10]

bel_list = [None] * 12
actions = list()
trans_mat = init_trans_mat()

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


for board_obs in obs_seq[1:]:
    
    bel_list_prev = deepcopy(bel_list)
    for p in p_idxs:
        bel_list[p] = np.matmul(trans_mat, bel_list[p])
    
    
    for p in p_idxs:
        bel_list[p] = refine_belief_with_obs(bel_list[p], board_obs[p])
        bel_list[p] = normalize_belief(bel_list[p])

        bel_list[5] = bel_list[select_ball_player(0, board_obs, bel_list)]
        bel_list[11] = bel_list[select_ball_player(6, board_obs, bel_list)]
    
    prev_estimate = [np.argmax(bel_list_prev[p]) for p in range(12)]
    # Compare the two beliefs to extract the action distribution
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
        # best_valid_actions.append((active_prob*max_action_prob, best_valid_action))
        best_valid_actions.append((max_action_prob, best_valid_action))

    best_valid_actions.sort(reverse=True)
    best_action = best_valid_actions[0]    
    
    actions.append(best_action[1])


correct = 0
for i in range(len(actions)):
    if actions[i] == gt_seq[i][1]:
        correct += 1
correct /= len(actions)
    
print(actions)
print([gt[1] for gt in gt_seq])
print(correct)
print(len(obs_seq))