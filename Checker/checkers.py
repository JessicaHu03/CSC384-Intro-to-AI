from copy import deepcopy
import copy
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
            self.utility = -(simple_utility(self))
        else:
            self.utility = simple_utility(self)

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


def min_pieces(state):
    num = 0
    for i in range(8):
        for j in range(8):
            if state.data[i][j] == 'b' or state.data[i][j] == 'B':
                num += 1
    return num


def max_pieces(state):
    num = 0
    for i in range(8):
        for j in range(8):
            if state.data[i][j] == 'r' or state.data[i][j] == 'R':
                num += 1
    return num


def num_king(state):
    r, b = 0, 0
    for i in range(state.dim):
        for j in range(state.dim):
            if state.data[i][j] == 'B':
                b += 1
            elif state.data[i][j] == 'R':
                r += 1
    return [r, b]

######################### utility function ########################


def simple_utility(state):
    king = num_king(state)
    num_red_king, num_black_king = king[0], king[1]
    red = 2 * num_red_king + (state.max_pieces - num_red_king)
    black = 2 * num_black_king + (state.min_pieces - num_black_king)
    return red - black


def formal_utility(state):
    res = 0
    max = 0
    min = 0
    for i in range(8):
        for j in range(8):
            # TODO
            pass

############################## next states #####################


def terminal(state):
    """ 
    return if the game is over in curr state
    """
    if state.min_pieces == 0 or state.max_pieces == 0:
        # "no piece"
        return True
    if len(find_moves(state)) == 0:
        # "not move"
        return True
    return False


def next_states(state):
    """ 
    return all the possible next states 
    """
    children = []
    moves = []
    new_player = None
    king_letter = ""
    king_row = 0

    if state.player == "Max":
        new_player = "Min"
        king_row = 0
        king_letter = "R"
    elif state.player == "Min":
        new_player = "Max"
        king_row = 7
        king_letter = "B"

    moves = find_moves(state)
    for i in range(len(moves)):
        old_y = moves[i][0]
        old_x = moves[i][1]
        new_y = moves[i][2]
        new_x = moves[i][3]

        old_pos = (old_y, old_x)
        new_pos = (new_y, new_x)

        new_data = deepcopy(state.data)
        move(new_data, old_pos, new_pos, king_letter, king_row)
        new_state = State(parent=state, data=new_data, player=new_player)

        children.append(new_state)
    return children


def check_max_moves(data, old_pos, new_pos):
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


def check_min_moves(data, old_pos, new_pos):
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


def check_jumps(data, old_pos, mid_pos, new_pos):
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


def multi_jump(init_data, player, move_dir, jump_dir, curr_pos):
    jumps = []
    y, x = curr_pos
    for k in range(len(jump_dir)):
        if check_jumps(init_data, (y, x), (y + move_dir[k][0], x + move_dir[k][1]), (y + jump_dir[k][0], x + jump_dir[k][1])):
            new_pos = multi_jump_helper(
                init_data, player, move_dir, jump_dir, curr_pos)
            if new_pos and new_pos != curr_pos:
                old_i, old_j = curr_pos
                new_i, new_j = new_pos
                jumps.append([old_i, old_j, new_i, new_j])
    return jumps


def multi_jump_helper(init_data, player, move_dir, jump_dir, curr_pos):
    """ return the final position after multi-jumping from i, j """
    y, x = curr_pos
    data = deepcopy(init_data)
    # stop multi-jump and return curr pos when there's no next available jump
    if all([check_jumps(data, (y, x), (y + move_dir[l][0], x + move_dir[l][1]), (y + jump_dir[l][0], x + jump_dir[l][1])) == False for l in range(len(jump_dir))]):
        return curr_pos

    for k in range(len(jump_dir)):
        new_pos = (y + jump_dir[k][0], x + jump_dir[k][1])
        if check_jumps(data, (y, x), (y + move_dir[k][0], x + move_dir[k][1]), new_pos):
            king_letter = 'R' if player == 'Max' else 'B'
            king_row = 0 if player == 'Max' else 7
            player = 'Min' if player == 'Max' else 'Max'
            move(data, (y, x), new_pos, king_letter, king_row)
            return multi_jump_helper(data, player, move_dir, jump_dir, new_pos)


def find_moves(state):
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

                    # multi-jump
                    jump_dir = [[-2, -2], [-2, +2]]
                    jumps = multi_jump(
                        state.data, state.player, move_dir, jump_dir, (i, j))

                # king
                elif state.data[i][j] == "R":
                    move_dir = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
                    for k in range(len(move_dir)):
                        if check_max_moves(state.data, (i, j), (i + move_dir[k][0], j + move_dir[k][1])):
                            moves.append(
                                [i, j, i + move_dir[k][0], j + move_dir[k][1]])

                    # multi-jump
                    jump_dir = [[-2, -2], [-2, +2], [2, -2], [2, 2]]
                    jumps = multi_jump(
                        state.data, state.player, move_dir, jump_dir, (i, j))

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
                        new_pos = multi_jump(
                            state.data, state.player, move_dir, jump_dir, (i, j))
                        if new_pos and new_pos != (i, j):
                            new_i, new_j = new_pos
                            jumps.append([i, j, new_i, new_j])

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
                        new_pos = multi_jump(
                            state.data, state.player, move_dir, jump_dir, (i, j))
                        if new_pos and new_pos != (i, j):
                            new_i, new_j = new_pos
                            jumps.append([i, j, new_i, new_j])

    if len(jumps) != 0:
        return jumps
    else:
        return moves


def move(data, old_pos, new_pos, king_letter, king_row):
    old_y, old_x = old_pos
    new_y, new_x = new_pos

    diff_y = old_y - new_y
    diff_x = old_x - new_x

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
    explored = dict()

    def Max_val(state, alpha, beta, depth):
        """ Max nodes """
        # print("MAX")
        # print("curr state: \n", output_format(state))
        if terminal(state) or depth == 0:
            # print("end here, terminal state: ",
            #       terminal(state), " depth: ", depth)
            return [state, simple_utility(state)]  # TODO
        value = float('-inf')
        best_move = state
        frontier = next_states(state)
        while frontier:
            curr = heapq.heappop(frontier)
            # print("child")
            # print(output_format(curr))
            explore = to_string(curr.data)
            if explore in explored and curr.player == explored[explore]:
                continue
            else:
                pair = Min_val(curr, alpha, beta, depth-1)
                move, val = pair[0], pair[1]
                # we want children of min node to order min to max
                curr.set_value(val)
                explored[explore] = curr.player
                if value < val:
                    best_move, value = move, val
                if value >= beta:
                    # print("the other children is pruned")
                    return [best_move, value]
                alpha = max(alpha, value)
        return [best_move, value]

    def Min_val(state, alpha, beta, depth):
        """ Min nodes """
        # print("MIN")
        # print("curr state: \n", output_format(state))
        if terminal(state) or depth == 0:
            # print("end here, terminal state: ",
            #      terminal(state), " depth: ", depth)
            return [state, simple_utility(state)]
        value = float('inf')
        best_move = state
        frontier = next_states(state)
        while frontier:
            curr = heapq.heappop(frontier)
            # print("child")
            # print(output_format(curr))
            explore = to_string(curr.data)
            if explore in explored and curr.player == explored[explore]:
                continue
            else:
                pair = Max_val(curr, alpha, beta, depth-1)
                move, val = pair[0], pair[1]
                curr.set_value(val)
                explored[explore] = curr.player
                if value > val:
                    best_move, value = move, val
                if value <= alpha:
                    # print("other children is pruned")
                    return [best_move, value]
                beta = min(beta, value)
        return [best_move, value]

    pair = Max_val(state, float('-inf'), float('inf'), depth)
    best_move, val = pair[0], pair[1]
    print("wanted value: ", val)
    return trace_back(best_move, state)


def trace_back(state, init_state):
    while state.parent != init_state:
        print(output_format(state))
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
    init = read_file('input1.txt')
    next = Minimax(init)
    print(output_format(next))

    # init = read_file(sys.argv[1])
    # output_file(sys.argv[2], init)
