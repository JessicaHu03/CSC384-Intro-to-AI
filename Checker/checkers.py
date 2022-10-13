from copy import deepcopy
import copy
import heapq
import sys
import numpy as np
from collections import defaultdict, deque
from queue import PriorityQueue
import os


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

    def __gt__(self, state):
        return self.value > state.value

    def __lt__(self, state):
        return self.value < state.value

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
    # print("terminal!")
    # print("max pieces = ", state.max_pieces)
    # print("min pieces = ", state.min_pieces)
    # print("curr state: ", output_format(state))

    if state.min_pieces == 0 or state.max_pieces == 0:
        # print("no piece")
        return True
    if len(find_moves(state)) == 0:
        # print("not move")
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


def multi_jump(data, player, move_dir, jump_dir, curr_pos):
    """ return the final position after multi-jumping from i, j """
    y, x = curr_pos
    # stop multi-jump when there's no next available jump
    if not any([check_jumps(data, (y, x), (y + move_dir[l][0], x + move_dir[l][1]), (y + jump_dir[l][0], x + jump_dir[l][1])) for l in range(len(jump_dir))]):
        # return curr pos
        return curr_pos

    for k in range(len(jump_dir)):
        new_pos = (y + jump_dir[k][0], x + jump_dir[k][1])
        if check_jumps(data, (y, x), (y + move_dir[k][0], x + move_dir[k][1]), new_pos):
            king_letter = 'R' if player == 'Max' else 'B'
            king_row = 0 if player == 'Max' else 7
            player = 'Min' if player == 'Max' else 'Max'
            move(data, (y, x), new_pos, king_letter, king_row)
            multi_jump(data, player, move_dir, jump_dir, new_pos)


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
                    for k in range(len(jump_dir)):
                        new_pos = multi_jump(
                            state.data, state.player, move_dir, jump_dir, (i, j))
                        if new_pos and new_pos != (i, j):
                            new_i, new_j = new_pos
                            jumps.append([i, j, new_i, new_j])

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
                        new_pos = multi_jump(
                            state.data, state.player, move_dir, jump_dir, (i, j))
                        if new_pos and new_pos != (i, j):
                            new_i, new_j = new_pos
                            jumps.append([i, j, new_i, new_j])

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
                                [i, j, i + move_dir[i][0], j + move_dir[i][1]])

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


#################### Minimax with AlphaBeta Pruning ###############

def Minimax(state):
    """ return the best move & max value by using minimax with alpha beta pruning"""
    depth = 6
    explored = dict()

    def Max_val(state, alpha, beta, depth):
        """ Max nodes """
        print("curr state: \n", output_format(state))
        if terminal(state) or depth == 0:
            print("end here, terminal state: ",
                  terminal(state), " depth: ", depth)
            return simple_utility(state)
        val = float('-inf')
        print("MAX turn")
        print("curr state: \n", output_format(state))
        frontier = next_states(state)
        while frontier:
            curr = heapq.heappop(frontier)
            print("child")
            print(output_format(curr))
            if is_in(curr, explored):
                continue
            else:
                val = max(val, Min_val(curr, alpha, beta, depth-1))
                # print("val = ", val)
                # we want children of min node to order min to max
                curr.set_value(-val)
                explored[curr] = val
                if val >= beta:
                    return val
                alpha = max(alpha, val)
        return val

    def Min_val(state, alpha, beta, depth):
        """ Min nodes """
        if terminal(state) or depth == 0:
            print("end here, terminal state: ",
                  terminal(state), " depth: ", depth)
            return simple_utility(state)
        val = float('inf')
        print("Min turn")
        print("curr state: \n", output_format(state))
        frontier = next_states(state)
        while frontier:
            # TODO value has not been assigned yet, heappop will not pop the min value node
            curr = heapq.heappop(frontier)
            print("child")
            print(output_format(curr))
            if is_in(curr, explored):
                continue
            else:
                val = min(val, Max_val(curr, alpha, beta, depth-1))
                # we want children of min node to order min to max
                curr.set_value(val)
                explored[curr] = val
                if val <= alpha:
                    return val
                beta = min(beta, val)
        return val

    val = Max_val(state, float('-inf'), float('inf'), depth)
    # print("explored: ")
    # for state in explored:
    #     print("state: ")
    #     print(output_format(state))
    #     print("value = ", explored[state])
    # print("explored end")
    return find_state(explored, val)


def is_in(given_state, explored):
    """ 
    find if the state is in explored
    """
    for state in explored:
        if state.player == given_state.player and state.data == given_state.data and state.cost == given_state.cost and state.value == given_state.value and state.cost == given_state.cost:
            return True
    return False


def find_state(explored, val):
    # print("children")
    # print("wanted val = ", val)
    for state in explored:
        # print(output_format(state))
        # print("with value = ", state.value)
        if state.value == val:
            return state
    # shouldn't get here
    print("WRONG")
    return None

######################### files ###########################


def output_format(state):
    result = ""
    for i in range(8):
        for j in range(8):
            result += str(state.data[i][j])
        result += "\n"
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
    init = read_file('input0.txt')
    next = Minimax(init)
    print(output_format(next))
