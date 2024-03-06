import numpy as np
from game import BoardState, GameSimulator, Rules
import queue


class Problem:

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        The form of initial state is:
        ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg="bfs"):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.
        """

        # defaults to bfs if no arg is given.
        if alg.lower() == 'a_star':
            self.search_alg_fnc = self.a_star
        else:
            self.search_alg_fnc = self.bfs

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
        self.sim.game_state.decode_state = self.sim.game_state.make_state()

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))

    ## TODO: Implement your search algorithm(s) here as methods of the GameStateProblem.
    ##       You are free to specify parameters that your method may require.
    ##       However, you must ensure that your method returns a list of (state, action) pairs, where
    ##       the first state and action in the list correspond to the initial state and action taken from
    ##       the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    ##       (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    ## NOTE: The format of state is a tuple: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers
    ##       (mirroring the contents of BoardState.state), and player_idx is 0 or 1, indicating the player that is
    ##       moving in this state.
    ##       The format of action is a tuple: (relative_idx, position), where relative_idx the relative index into encoded_state
    ##       with respect to player_idx, and position is the encoded position where the piece should move to with this action.
    ## NOTE: self.get_actions will obtain the current actions available in current game state.
    ## NOTE: self.execute acts like the transition function.
    ## NOTE: Remember to set self.search_alg_fnc in set_search_alg above.
    ## 


    # BFS: Breadth-First Search implementation
    def bfs(self):
        horizon = list()
        visited = set()
        parent = dict()
        path = list()

        visited.add(self.initial_state)
        horizon.append(self.initial_state)
        parent[self.initial_state] = None
        current_state = None
        while horizon:
            current_state = horizon.pop(0)

            if self.is_goal(current_state):
                path.append((current_state, None))
                break


            actions = self.get_actions(current_state)
            for action in actions:
                next_state = self.execute(current_state, action)
                if next_state not in visited:
                    parent[next_state] = (current_state, action)
                    visited.add(next_state)
                    horizon.append(next_state)

        # extract path
        while parent[current_state] is not None:
            path.insert(0, parent[current_state])
            current_state = parent[current_state][0]
        return path
    


    # Heuristic: # of pieces not in their goal positions
    def heuristic(self, state):
        # cost can never exceed this; max cost is all pieces not in place or entire state list
        min_cost = len(state)+1
        for goal_state in self.goal_state_set:
            goal, player = goal_state
            curr_cost = 0
            for current, target in zip(state,goal):
                if current != target:
                    curr_cost += 1
            if curr_cost < min_cost:
                min_cost = curr_cost
        
        return min_cost
            
    # a_start implements a* search with the heuristic defined in self.heuristic
    def a_star(self):
        horizon = queue.PriorityQueue()
        visited = set()
        parent = dict()
        path = list()

        visited.add(self.initial_state)
        horizon.put((self.heuristic(self.initial_state[0]),0,self.initial_state))
        parent[self.initial_state] = None
        current_state = None
        while horizon:
            current_h, current_path_cost, current_state = horizon.get()

            if self.is_goal(current_state):
                path.append((current_state, None))
                break

            actions = self.get_actions(current_state)
            for action in actions:
                next_state = self.execute(current_state, action)
                if next_state not in visited:
                    parent[next_state] = (current_state, action)
                    visited.add(next_state)
                    horizon.put((self.heuristic(next_state[0])+current_path_cost+1,current_path_cost+1,next_state))

        # extract path
        while parent[current_state] is not None:
            path.insert(0, parent[current_state])
            current_state = parent[current_state][0]
        return path


