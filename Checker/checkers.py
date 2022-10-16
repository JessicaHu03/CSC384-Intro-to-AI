from copy import deepcopy
from distutils.ccompiler import new_compiler
import heapq
import sys


class State:
    """A state is a table of all the tiles with given initial locations.
    """

    def __init__(self, dim=8, player="Max", data=None, value=None, parent=None):
        self.dim = dim
        self.player = player
        self.data = data
        self.value = value
        self.parent = parent
        if self.parent:
            self.cost = parent.cost + 1
        else:
            self.cost = 0
        self.min_pieces = min_pieces(self)
        self.max_pieces = max_pieces(self)
        if self.player == 'Max':
            self.utility = -(formal_utility(self))
        else:
            self.utility = formal_utility(self)

    # for node ordering
    def __gt__(self, state):
        return self.utility > state.utility

    def __lt__(self, state):
        return self.utility < state.utility

    def get_data(self):
        """ return curr data"""
        return self.data

    def get_parent(self):
        """ return the parent of curr state"""
        return self.parent

    def get_player(self):
        """ return the curr player """
        return self.player

    def set_player(self, player):
        self.player = player

    def set_value(self, value):
        self.value = value


def min_pieces(state):  # checked
    num = 0
    for i in range(8):
        for j in range(8):
            if state.data[i][j] == 'b' or state.data[i][j] == 'B':
                num += 1
    return num


def max_pieces(state):  # checked
    num = 0
    for i in range(8):
        for j in range(8):
            if state.data[i][j] == 'r' or state.data[i][j] == 'R':
                num += 1
    return num


def num_king(state):  # checked
    r, b = 0, 0
    for i in range(state.dim):
        for j in range(state.dim):
            if state.data[i][j] == 'B':
                b += 1
            elif state.data[i][j] == 'R':
                r += 1
    return [r, b]


######################### utility function ########################

def simple_utility(state):  # checked
    king = num_king(state)
    num_red_king, num_black_king = king[0], king[1]
    red = 2 * num_red_king + (state.max_pieces - num_red_king)
    black = 2 * num_black_king + (state.min_pieces - num_black_king)
    return red - black


def formal_utility(state):
    king = num_king(state)
    num_red_king, num_black_king = king[0], king[1]
    distance = total_distance(state)
    red_distance, black_distance = distance[0], distance[1]
    homes = home(state)
    red_home, black_home = homes[0], homes[1]

    red = 10 * num_red_king + \
        (state.max_pieces - num_red_king) + red_distance + 10 * red_home
    black = 10 * num_black_king + \
        (state.min_pieces - num_black_king) + black_distance + 10 * black_home
    result = red - black
    return result


def total_distance(state):
    """ distance from king row (attack) """
    red_dis = 0
    black_dis = 0
    for i in range(8):
        for j in range(8):
            if state.data[i][j] == 'r':
                red_dis += (8 - i)
            elif state.data[i][j] == 'b':
                black_dis += i
    return [red_dis, black_dis]


def home(state):
    """ num of pieces in home row (defend) """
    red = 0
    black = 0
    i = 0
    while i < 8:
        if state.data[0][i+1] == 'b':
            black += 1
        if state.data[7][i] == 'r':
            red += 1
        i += 2
    return [red, black]

############################## next states #####################


def terminal(state):  # checked
    """ 
    return if the game is over in curr state
    """
    if state.min_pieces == 0 or state.max_pieces == 0:
        # "no piece"
        return True
    if len(find_moves(state)) == 0:
        # "no move"
        return True
    return False


def check_max_moves(data, old_pos, new_pos):  # checked
    old_y, old_x = old_pos
    new_y, new_x = new_pos
    if new_y < 0 or new_y > 7:
        return False
    if new_x < 0 or new_x > 7:
        return False
    if data[old_y][old_x] == ".":
        return False
    if data[new_y][new_x] != ".":
        return False
    if data[old_y][old_x] == "b" or data[old_y][old_x] == "B":
        return False
    if data[new_y][new_x] == ".":
        return True


def check_min_moves(data, old_pos, new_pos):  # checked
    old_y, old_x = old_pos
    new_y, new_x = new_pos
    if new_y < 0 or new_y > 7:
        return False
    if new_x < 0 or new_x > 7:
        return False
    if data[old_y][old_x] == ".":
        return False
    if data[new_y][new_x] != ".":
        return False
    if data[old_y][old_x] == "r" or data[old_y][old_x] == "R":
        return False
    if data[new_y][new_x] == ".":
        return True


def check_max_jumps(data, old_pos, mid_pos, new_pos):  # checked
    old_y, old_x = old_pos
    mid_y, mid_x = mid_pos
    new_y, new_x = new_pos
    if new_y < 0 or new_y > 7:
        return False
    if new_x < 0 or new_x > 7:
        return False
    if data[new_y][new_x] != ".":
        return False
    if data[mid_y][mid_x] == ".":
        return False
    if data[mid_y][mid_x] == "r" or data[mid_y][mid_x] == "R":
        return False
    if data[old_y][old_x] == ".":
        return False
    if data[old_y][old_x] == "b" or data[old_y][old_x] == "B":
        return False
    return True


def check_min_jumps(data, old_pos, mid_pos, new_pos):  # checked
    old_y, old_x = old_pos
    mid_y, mid_x = mid_pos
    new_y, new_x = new_pos
    if new_y < 0 or new_y > 7:
        return False
    if new_x < 0 or new_x > 7:
        return False
    if data[new_y][new_x] != ".":
        return False
    if data[mid_y][mid_x] == ".":
        return False
    if data[mid_y][mid_x] == "b" or data[mid_y][mid_x] == "B":
        return False
    if data[old_y][old_x] == ".":
        return False
    if data[old_y][old_x] == "r" or data[old_y][old_x] == "R":
        return False
    return True


def multi_jump(state, player, move_dir, jump_dir, curr_pos):  # checked
    """ 
    if there's available jumping, jumps
    return the max available jump ending state
    """
    init_data = state.data
    jumps = []
    y, x = curr_pos
    for k in range(len(jump_dir)):
        if player == 'Max':
            new_player = 'Min'
            new_pos = (y + jump_dir[k][0], x + jump_dir[k][1])
            if check_max_jumps(init_data, (y, x), (y + move_dir[k][0], x + move_dir[k][1]), new_pos):
                # checked that new_pos can be jumped
                new_data = deepcopy(init_data)
                move(new_data, curr_pos, new_pos)
                new_data = multi_jump_helper(
                    new_data, player, move_dir, jump_dir, new_pos)
                new_state = State(
                    parent=state, data=new_data, player=new_player)
                jumps.append(new_state)
        else:
            new_player = 'Max'
            new_pos = (y + jump_dir[k][0], x + jump_dir[k][1])
            if check_min_jumps(init_data, (y, x), (y + move_dir[k][0], x + move_dir[k][1]), new_pos):
                new_data = deepcopy(init_data)
                move(new_data, curr_pos, new_pos)
                new_data = multi_jump_helper(
                    new_data, player, move_dir, jump_dir, new_pos)
                new_state = State(
                    parent=state, data=new_data, player=new_player)
                jumps.append(new_state)
    return jumps


def multi_jump_helper(init_data, player, move_dir, jump_dir, curr_pos):  # checker
    """ return the final position after multi-jumping from i, j """
    # print("curr pos:", curr_pos)
    y, x = curr_pos
    data = deepcopy(init_data)
    # stop multi-jump and return when there's no next available jump
    if player == 'Max':
        if all([check_max_jumps(data, (y, x), (y + move_dir[l][0], x + move_dir[l][1]), (y + jump_dir[l][0], x + jump_dir[l][1])) == False for l in range(len(jump_dir))]):
            return data

        for k in range(len(jump_dir)):
            new_pos = (y + jump_dir[k][0], x + jump_dir[k][1])
            # print("new pos: ", new_pos)
            if check_max_jumps(data, (y, x), (y + move_dir[k][0], x + move_dir[k][1]), new_pos):
                king_letter = 'R'
                king_row = 0
                move(data, (y, x), new_pos, king_letter, king_row)
                return multi_jump_helper(data, player, move_dir, jump_dir, new_pos)
    else:
        if all([check_min_jumps(data, (y, x), (y + move_dir[l][0], x + move_dir[l][1]), (y + jump_dir[l][0], x + jump_dir[l][1])) == False for l in range(len(jump_dir))]):
            return data

        for k in range(len(jump_dir)):
            new_pos = (y + jump_dir[k][0], x + jump_dir[k][1])
            if check_min_jumps(data, (y, x), (y + move_dir[k][0], x + move_dir[k][1]), new_pos):
                king_letter = 'B'
                king_row = 7
                move(data, (y, x), new_pos, king_letter, king_row)
                return multi_jump_helper(data, player, move_dir, jump_dir, new_pos)


def find_states(state):  # checked
    """ return a list of children states"""
    moves_pos = []

    moves_states = []
    jumps_states = []

    if state.player == 'Max':
        new_player = "Min"
        king_row = 0
        king_letter = "R"
        for i in range(8):
            for j in range(8):
                # men
                if state.data[i][j] == "r":
                    # go left up and right up
                    move_dir = [[-1, -1], [-1, +1]]
                    for k in range(len(move_dir)):
                        if check_max_moves(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1])):
                            new_data = deepcopy(state.data)
                            move(
                                new_data, (i, j), (i + move_dir[k][0], j + move_dir[k][1]), king_letter, king_row)
                            new_state = State(
                                parent=state, data=new_data, player=new_player)
                            moves_states.append(new_state)

                    # multi-jump
                    jump_dir = [[-2, -2], [-2, +2]]
                    mj = multi_jump(state,
                                    state.player, move_dir, jump_dir, (i, j))
                    if len(mj) != 0:
                        for j in mj:
                            jumps_states.append(j)

                # king
                elif state.data[i][j] == "R":
                    move_dir = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
                    for k in range(len(move_dir)):
                        if check_max_moves(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1])):
                            new_data = deepcopy(state.data)
                            move(
                                new_data, (i, j), (i + move_dir[k][0], j + move_dir[k][1]), king_letter, king_row)
                            new_state = State(
                                parent=state, data=new_data, player=new_player)
                            moves_states.append(new_state)

                    # multi-jump
                    jump_dir = [[-2, -2], [-2, +2], [2, -2], [2, 2]]
                    mj = multi_jump(state,
                                    state.player, move_dir, jump_dir, (i, j))
                    if len(mj) != 0:
                        for j in mj:
                            jumps_states.append(j)

    elif state.player == 'Min':
        new_player = "Max"
        king_row = 7
        king_letter = "B"
        for i in range(8):
            for j in range(8):
                # men
                if state.data[i][j] == "b":
                    move_dir = [[1, -1], [1, +1]]
                    for k in range(len(move_dir)):
                        if check_min_moves(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1])):
                            new_data = deepcopy(state.data)
                            move(
                                new_data, (i, j), (i + move_dir[k][0], j + move_dir[k][1]), king_letter, king_row)
                            new_state = State(
                                parent=state, data=new_data, player=new_player)
                            moves_states.append(new_state)

                    # multi-jump
                    jump_dir = [[2, -2], [2, +2]]
                    mj = multi_jump(state,
                                    state.player, move_dir, jump_dir, (i, j))
                    if len(mj) != 0:
                        for j in mj:
                            jumps_states.append(j)

                # king
                elif state.data[i][j] == "B":
                    move_dir = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
                    for k in range(len(move_dir)):
                        if check_min_moves(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1])):
                            new_data = deepcopy(state.data)
                            move(
                                new_data, (i, j), (i + move_dir[k][0], j + move_dir[k][1]), king_letter, king_row)
                            new_state = State(
                                parent=state, data=new_data, player=new_player)
                            moves_states.append(new_state)

                    # multi-jump
                    jump_dir = [[-2, -2], [-2, +2], [2, -2], [2, 2]]
                    mj = multi_jump(state,
                                    state.player, move_dir, jump_dir, (i, j))
                    if len(mj) != 0:
                        for j in mj:
                            jumps_states.append(j)

    if len(jumps_states) != 0:
        return jumps_states
    else:
        return moves_states


def find_moves(state):  # checked
    """ return a list of moves and jumps available for Max or Min"""
    moves = []
    jumps = []
    if state.player == 'Max':
        for i in range(8):
            for j in range(8):
                # men
                if state.data[i][j] == "r":
                    # go left up and right up
                    move_dir = [[-1, -1], [-1, +1]]
                    for k in range(len(move_dir)):
                        if check_max_moves(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1])):
                            moves.append(
                                [i, j, i + move_dir[k][0], j + move_dir[k][1]])

                    jump_dir = [[-2, -2], [-2, +2]]
                    for k in range(len(jump_dir)):
                        new_pos = (i + jump_dir[k][0], j + jump_dir[k][1])
                        if (check_max_jumps(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1]), new_pos)):
                            jumps.append(new_pos)

                # king
                elif state.data[i][j] == "R":
                    move_dir = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
                    for k in range(len(move_dir)):
                        if check_max_moves(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1])):
                            moves.append(
                                [i, j, i + move_dir[k][0], j + move_dir[k][1]])

                    # multi-jump
                    jump_dir = [[-2, -2], [-2, +2], [2, -2], [2, 2]]
                    for k in range(len(jump_dir)):
                        new_pos = (i + jump_dir[k][0], j + jump_dir[k][1])
                        if (check_max_jumps(state.data, (i, j), (
                                i + move_dir[k][0], j + move_dir[k][1]), new_pos)):
                            jumps.append(new_pos)

    elif state.player == 'Min':
        for i in range(8):
            for j in range(8):
                # men
                if state.data[i][j] == "b":
                    move_dir = [[1, -1], [1, +1]]
                    for k in range(len(move_dir)):
                        if check_min_moves(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1])):
                            moves.append(
                                [i, j, i + move_dir[k][0], j + move_dir[k][1]])

                    # multi-jump
                    jump_dir = [[2, -2], [2, +2]]
                    for k in range(len(jump_dir)):
                        new_pos = (i + jump_dir[k][0], j + jump_dir[k][1])
                        if (check_min_jumps(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1]), new_pos)):
                            jumps.append(new_pos)

                # king
                elif state.data[i][j] == "B":
                    move_dir = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
                    for k in range(len(move_dir)):
                        if check_min_moves(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1])):
                            moves.append(
                                [i, j, i + move_dir[k][0], j + move_dir[k][1]])

                    # multi-jump
                    jump_dir = [[-2, -2], [-2, +2], [2, -2], [2, 2]]
                    for k in range(len(jump_dir)):
                        new_pos = (i + jump_dir[k][0], j + jump_dir[k][1])
                        if (check_min_jumps(state.data, (i, j), (
                                i + move_dir[k][0], j + move_dir[k][1]), new_pos)):
                            jumps.append(new_pos)

    if len(jumps) != 0:
        return jumps
    else:
        return moves


def move(data, old_pos, new_pos, king_letter='R', king_row=0):  # checked
    old_y, old_x = old_pos
    new_y, new_x = new_pos

    diff_y = old_y - new_y
    diff_x = old_x - new_x

    # for single jumps
    if diff_y == -2 and diff_x == 2:
        data[old_y + 1][old_x - 1] = "."

    elif diff_y == 2 and diff_x == 2:
        data[old_y - 1][old_x - 1] = "."

    elif diff_y == 2 and diff_x == -2:
        data[old_y - 1][old_x + 1] = "."

    elif diff_y == -2 and diff_x == -2:
        data[old_y + 1][old_x + 1] = "."

    if new_y == king_row:
        letter = king_letter
    else:
        letter = data[old_y][old_x]
    data[old_y][old_x] = "."
    data[new_y][new_x] = letter


#################### Minimax with AlphaBeta Pruning ##############

def Minimax(state):
    """ return the best move & max value by using minimax with alpha beta pruning"""
    depth = 8

    def AlphaBeta(state, alpha, beta, depth):
        best_move = None
        if terminal(state) or depth == 0:
            return state, formal_utility(state)
        if state.player == 'Max':
            value = float('-inf')
        if state.player == 'Min':
            value = float('inf')
        frontier = find_states(state)
        while frontier:
            nxt_state = heapq.heappop(frontier)
            nxt_move, nxt_val = AlphaBeta(nxt_state, alpha, beta, depth-1)
            if state.player == 'Max':
                if value < nxt_val:
                    value, best_move = nxt_val, nxt_move
                if value >= beta:
                    return best_move, value
                alpha = max(alpha, value)
            if state.player == 'Min':
                if value > nxt_val:
                    value, best_move = nxt_val, nxt_move
                if value <= alpha:
                    return best_move, value
                beta = min(beta, value)
        return best_move, value

    best_move, value = AlphaBeta(state, float('-inf'), float('inf'), depth)
    # print("wanted value: ", value)
    return trace_back(best_move, state)


def trace_back(state, init_state):
    while state.parent != init_state:
        state = state.parent
    return state


def to_string(data):
    string = " ".join([' '.join([str(c) for c in lst]) for lst in data])
    return string

######################### files ###########################


def output_format(state):
    result = ""
    for i in range(8):
        for j in range(8):
            result += str(state.data[i][j])
        result += "\n"
    return result


def read_file(filename):
    """ read the state from input file"""
    file = open(filename, "r")

    new_data = [["." for _ in range(8)] for _ in range(8)]

    for i in range(8):
        row = file.readline()  # len = 8
        for j in range(8):
            new_data[i][j] = str(row[j])

    res = State(data=new_data)
    file.close()
    return res


def output_file(filename, state):
    """ print dfs output into file """

    file = open(filename, "w")

    next_state = Minimax(state)

    file.write(output_format(next_state))

    file.close()


if __name__ == '__main__':

    init = read_file(sys.argv[1])
    output_file(sys.argv[2], init)
