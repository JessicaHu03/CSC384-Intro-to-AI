from copy import deepcopy
import heapq
import sys
import numpy as np
from collections import defaultdict, deque
from queue import PriorityQueue


class State:
    """A state is a table of all the tiles with given initial locations.
    """

    def __init__(self, width=4, height=5, data=None, parent=None):
        """read the input file and convert into state table"""
        self.width = width    # num of cols
        self.height = height  # num of rows
        self.data = data
        self.parent = parent
        if self.parent:
            self.cost = parent.cost + 1
        else:
            self.cost = 0

    def __gt__(self, state):
        return func_f(self) > func_f(state)

    def __lt__(self, state):
        return func_f(self) < func_f(state)

    def get_data(self):
        """ return curr data"""
        return self.data

    def get_parent(self):
        """ return the parent of curr state"""
        return self.parent

    def get_cost(self):
        """ Get the g function value """
        return self.cost


def get_blank_pos(state):  # checked
    """ return the pos of the two blank piece (y, x) """
    res = []
    for i in range(state.height):
        for j in range(state.width):
            if state.data[i][j] == 0:
                res.append([0, (i, j)])
    return res


def get_data_pos(state):  # checked
    """ return a table of positions where each value as a key and a list of corresponding positions as value"""
    data_pos = defaultdict(list)
    for i in range(state.height):
        for j in range(state.width):
            data_pos[state.data[i][j]].append((i, j))
    return data_pos


def get_pos(state, piece):  # checked
    """ return the curr pos (upper left corner) of this piece"""
    if piece[0] == 7:
        return piece[1]
    positions = get_data_pos(state)
    (min_y, min_x) = piece[1]
    for position in positions[piece[0]]:
        y, x = position
        if y <= min_y and x <= min_x:
            min_y = y
            min_x = x
    return min_y, min_x


def get_cao_cao_pos(state):  # checked
    """ return the upper left corner pos of cao cao (y, x)"""
    for i in range(state.height):
        for j in range(state.width):
            if state.data[i][j] == 1:
                return (i, j)


def pieces_around(state, blank1, blank2=None):  # checked
    """ return the value of the pieces and corresponding pos that is around blank spaces """
    res = []
    blank1_y, blank1_x = blank1
    if blank2:
        blank2_y, blank2_x = blank2
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # up, down, right, left
    for dx, dy in directions:
        new_pos_x = blank1_x + dx
        new_pos_y = blank1_y + dy
        if 0 <= new_pos_x <= 3 and 0 <= new_pos_y <= 4:
            value = state.data[new_pos_y][new_pos_x]
            res.append([value, (new_pos_y, new_pos_x)])
        if blank2:
            new_pos_x = blank2_x + dx
            new_pos_y = blank2_y + dy
            if 0 <= new_pos_x <= 3 and 0 <= new_pos_y <= 4:
                value = state.data[new_pos_y][new_pos_x]
                if [value, (new_pos_y, new_pos_x)] not in res:
                    res.append([value, (new_pos_y, new_pos_x)])
    return res

# def is_adjacent(pos1, pos2):  # checked
#     """ return if pos1 is adjacent to pos2"""
#     pos1_y, pos1_x = pos1
#     pos2_y, pos2_x = pos2
#     if pos1_x == pos2_x + 1 or pos1_x == pos2_x - 1:
#         return True
#     if pos1_y == pos2_y + 1 or pos1_y == pos2_y - 1:
#         return True
#     return False


def move_up(state, piece):  # checked
    """ move this piece up"""
    (y, x) = piece[1]
    new_state_data = deepcopy(state.data)
    new_state = State(data=new_state_data, parent=state)
    pos_table = get_data_pos(state)
    if len(pos_table[piece[0]]) == 4:  # 2x2 or 1x1
        if state.data[y][x] == 1:  # 2x2
            y, x = get_pos(state, piece)
            new_state.data[y][x], new_state.data[y -
                                                 1][x] = new_state.data[y-1][x], new_state.data[y][x]
            new_state.data[y][x+1], new_state.data[y-1][x +
                                                        1] = new_state.data[y-1][x+1], new_state.data[y][x+1]
            new_state.data[y+1][x], new_state.data[y][x] = new_state.data[y][x], new_state.data[y+1][x]
            new_state.data[y+1][x+1], new_state.data[y][x +
                                                        1] = new_state.data[y][x+1], new_state.data[y+1][x+1]
        elif state.data[y][x] == 7:  # 1x1
            new_state.data[y][x], new_state.data[y -
                                                 1][x] = new_state.data[y-1][x], new_state.data[y][x]
    elif len(pos_table[piece[0]]) == 2:  # 1x2 or 2x1
        (y1, x1) = pos_table[piece[0]][0]
        (y2, x2) = pos_table[piece[0]][1]
        if x1 == x2 and y1 != y2:  # 2x1
            y, x = get_pos(state, piece)
            new_state.data[y][x], new_state.data[y -
                                                 1][x] = new_state.data[y-1][x], new_state.data[y][x]
            new_state.data[y+1][x], new_state.data[y][x] = new_state.data[y][x], new_state.data[y+1][x]
        elif y1 == y2 and x1 != x2:  # 1x2
            y, x = get_pos(state, piece)
            new_state.data[y][x], new_state.data[y -
                                                 1][x] = new_state.data[y-1][x], new_state.data[y][x]
            new_state.data[y][x+1], new_state.data[y-1][x +
                                                        1] = new_state.data[y-1][x+1], new_state.data[y][x+1]
    return new_state


def move_down(state, piece):  # checked
    """ move piece down"""
    (y, x) = piece[1]
    new_state_data = deepcopy(state.data)
    new_state = State(data=new_state_data, parent=state)
    pos_table = get_data_pos(state)
    if len(pos_table[piece[0]]) == 4:  # 2x2 or 1x1
        if state.data[y][x] == 1:  # 2x2
            y, x = get_pos(state, piece)
            new_state.data[y+1][x], new_state.data[y +
                                                   2][x] = new_state.data[y+2][x], new_state.data[y+1][x]
            new_state.data[y+1][x+1], new_state.data[y+2][x +
                                                          1] = new_state.data[y+2][x+1], new_state.data[y+1][x+1]
            new_state.data[y][x], new_state.data[y +
                                                 1][x] = new_state.data[y+1][x], new_state.data[y][x]
            new_state.data[y][x+1], new_state.data[y+1][x +
                                                        1] = new_state.data[y+1][x+1], new_state.data[y][x+1]
        elif state.data[y][x] == 7:  # 1x1
            new_state.data[y][x], new_state.data[y +
                                                 1][x] = new_state.data[y+1][x], new_state.data[y][x]
    elif len(pos_table[piece[0]]) == 2:  # 1x2 or 2x1
        (y1, x1) = pos_table[piece[0]][0]
        (y2, x2) = pos_table[piece[0]][1]
        if x1 == x2 and y1 != y2:  # 2x1
            y, x = get_pos(state, piece)
            new_state.data[y+1][x], new_state.data[y +
                                                   2][x] = new_state.data[y+2][x], new_state.data[y+1][x]
            new_state.data[y][x], new_state.data[y +
                                                 1][x] = new_state.data[y+1][x], new_state.data[y][x]
        elif y1 == y2 and x1 != x2:  # 1x2
            y, x = get_pos(state, piece)
            new_state.data[y][x], new_state.data[y +
                                                 1][x] = new_state.data[y+1][x], new_state.data[y][x]
            new_state.data[y][x+1], new_state.data[y+1][x +
                                                        1] = new_state.data[y+1][x+1], new_state.data[y][x+1]

    return new_state


def move_left(state, piece):  # checked
    """ move piece left"""
    (y, x) = piece[1]
    new_state_data = deepcopy(state.data)
    new_state = State(data=new_state_data, parent=state)
    pos_table = get_data_pos(state)
    if len(pos_table[piece[0]]) == 4:  # 2x2 or 1x1
        if state.data[y][x] == 1:  # 2x2
            y, x = get_pos(state, piece)
            new_state.data[y][x], new_state.data[y][x -
                                                    1] = new_state.data[y][x-1], new_state.data[y][x]
            new_state.data[y][x +
                              1], new_state.data[y][x] = new_state.data[y][x], new_state.data[y][x+1]
            new_state.data[y+1][x], new_state.data[y+1][x -
                                                        1] = new_state.data[y+1][x-1], new_state.data[y+1][x]
            new_state.data[y+1][x+1], new_state.data[y +
                                                     1][x] = new_state.data[y+1][x], new_state.data[y+1][x+1]

        elif state.data[y][x] == 7:  # 1x1
            new_state.data[y][x], new_state.data[y][x -
                                                    1] = new_state.data[y][x-1], new_state.data[y][x]

    elif len(pos_table[piece[0]]) == 2:  # 1x2 or 2x1
        (y1, x1) = pos_table[piece[0]][0]
        (y2, x2) = pos_table[piece[0]][1]
        if x1 == x2 and y1 != y2:  # 2x1
            y, x = get_pos(state, piece)
            new_state.data[y][x], new_state.data[y][x -
                                                    1] = new_state.data[y][x-1], new_state.data[y][x]
            new_state.data[y+1][x], new_state.data[y+1][x -
                                                        1] = new_state.data[y+1][x-1], new_state.data[y+1][x]
        elif y1 == y2 and x1 != x2:  # 1x2
            y, x = get_pos(state, piece)
            new_state.data[y][x], new_state.data[y][x -
                                                    1] = new_state.data[y][x-1], new_state.data[y][x]
            new_state.data[y][x +
                              1], new_state.data[y][x] = new_state.data[y][x], new_state.data[y][x+1]
    return new_state


def move_right(state, piece):  # checked
    """ move pieces right if possible """
    (y, x) = piece[1]
    new_state_data = deepcopy(state.data)
    new_state = State(data=new_state_data, parent=state)
    pos_table = get_data_pos(state)
    if len(pos_table[piece[0]]) == 4:  # 2x2 or 1x1
        if state.data[y][x] == 1:  # 2x2
            y, x = get_pos(state, piece)
            new_state.data[y][x+1], new_state.data[y][x +
                                                      2] = new_state.data[y][x+2], new_state.data[y][x+1]
            new_state.data[y][x], new_state.data[y][x +
                                                    1] = new_state.data[y][x+1], new_state.data[y][x]
            new_state.data[y+1][x+1], new_state.data[y+1][x +
                                                          2] = new_state.data[y+1][x+2], new_state.data[y+1][x+1]
            new_state.data[y+1][x], new_state.data[y+1][x +
                                                        1] = new_state.data[y+1][x+1], new_state.data[y+1][x]
        elif state.data[y][x] == 7:  # 1x1
            new_state.data[y][x], new_state.data[y][x +
                                                    1] = new_state.data[y][x+1], new_state.data[y][x]
    elif len(pos_table[piece[0]]) == 2:  # 1x2 or 2x1
        (y1, x1) = pos_table[piece[0]][0]
        (y2, x2) = pos_table[piece[0]][1]
        if x1 == x2 and y1 != y2:  # 2x1
            y, x = get_pos(state, piece)
            new_state.data[y][x], new_state.data[y][x +
                                                    1] = new_state.data[y][x+1], new_state.data[y][x]
            new_state.data[y+1][x], new_state.data[y+1][x +
                                                        1] = new_state.data[y+1][x+1], new_state.data[y+1][x]
        elif y1 == y2 and x1 != x2:  # 1x2
            y, x = get_pos(state, piece)
            new_state.data[y][x+1], new_state.data[y][x +
                                                      2] = new_state.data[y][x+2], new_state.data[y][x+1]
            new_state.data[y][x], new_state.data[y][x +
                                                    1] = new_state.data[y][x+1], new_state.data[y][x]
    return new_state


def func_g(state):
    """ return g(n): the path cost so far"""
    return state.get_cost()


def func_h(state):
    """ return h(n): Manhanttan heuristic
    the vertical + horizatonal distance from curr cao cao pos to goal pos (3, 1) """
    cao_y, cao_x = get_cao_cao_pos(state)
    goal_y, goal_x = 3, 1
    h = abs(goal_x - cao_x) + abs(goal_y - cao_y)
    return h


def func_f(state):
    """ return f(n) = g(n) + h(n) the total cost """
    return func_g(state) + func_h(state)


def is_final(state):  # checked
    """ if the curr state is the final state"""
    data = state.get_data()
    if (data[3][1] == 1 and data[4][1] == 1
            and data[3][2] == 1 and data[4][2] == 1):
        return True
    else:
        return False


def next_states(state):
    """ return all the possible next states """
    children = []
    blanks = get_blank_pos(state)
    blank1, blank2 = blanks[0], blanks[1]
    pieces = pieces_around(state, blank1[1], blank2[1])
    pos_table = get_data_pos(state)

    for piece in pieces:
        if piece[0] == 0:
            continue
        y, x = get_pos(state, piece)

        # if any of the four sides is empty to move, move to that dir, add the new state to children
        def is_empty(dir):
            """ return if dir is empty to move"""
            if dir == 1:  # left
                if piece[0] == 1:  # 2x2
                    return (x > 0 and y < 4 and state.data[y][x-1] == 0 and state.data[y+1][x-1] == 0)
                elif piece[0] == 7:  # 1x1
                    return (x > 0 and state.data[y][x-1] == 0)
                else:
                    if piece[0] != 0:
                        if x < 3 and state.data[y][x] == state.data[y][x+1]:  # 1x2
                            return (x > 0 and state.data[y][x-1] == 0)
                        else:  # 2x1
                            return (x > 0 and y < 4 and state.data[y][x-1] == 0 and state.data[y+1][x-1] == 0)
            if dir == 2:  # right
                if piece[0] == 1:  # 2x2
                    return (x < 2 and y < 4 and state.data[y][x+2] == 0 and state.data[y+1][x+2] == 0)
                elif piece[0] == 7:  # 1x1
                    return (x < 3 and state.data[y][x+1] == 0)
                else:
                    if piece[0] != 0:
                        if x < 3 and state.data[y][x] == state.data[y][x+1]:  # 1x2
                            return (x < 2 and state.data[y][x+2] == 0)
                        else:  # 2x1
                            return (x < 3 and y < 4 and state.data[y][x+1] == 0 and state.data[y+1][x+1] == 0)
            if dir == 3:  # up
                if piece[0] == 1:  # 2x2
                    return (x < 3 and y > 0 and state.data[y-1][x] == 0 and state.data[y-1][x+1] == 0)
                elif piece[0] == 7:  # 1x1
                    return (y > 0 and state.data[y-1][x] == 0)
                else:
                    if piece[0] != 0:
                        if x < 3 and state.data[y][x] == state.data[y][x+1]:  # 1x2
                            return (x < 3 and y > 0 and state.data[y-1][x] == 0 and state.data[y-1][x+1] == 0)
                        else:  # 2x1
                            return (y > 0 and state.data[y-1][x] == 0)
            if dir == 4:  # down
                if piece[0] == 1:  # 2x2
                    return (x < 3 and y < 3 and state.data[y+2][x] == 0 and state.data[y+2][x+1] == 0)
                elif piece[0] == 7:  # 1x1
                    return (y < 4 and state.data[y+1][x] == 0)
                else:
                    if piece[0] != 0:
                        if x < 3 and state.data[y][x] == state.data[y][x+1]:  # 1x2
                            return (x < 3 and y < 4 and state.data[y+1][x] == 0 and state.data[y+1][x+1] == 0)
                        else:  # 2x1
                            return (y < 3 and state.data[y+2][x] == 0)

        if is_empty(1):
            children.append(move_left(state, piece))
        if is_empty(2):
            children.append(move_right(state, piece))
        if is_empty(3):
            children.append(move_up(state, piece))
        if is_empty(4):
            children.append(move_down(state, piece))

    return children


def a_star(state):
    """ A* """
    explored = set()
    frontier = [state]

    moves = 0
    while frontier:
        curr = heapq.heappop(frontier)
        explore = " ".join([' '.join([str(c) for c in lst])
                           for lst in curr.get_data()])
        if explore in explored:
            # skip to prune this path
            continue
        else:
            explored.add(explore)
            states = next_states(curr)
            moves += 1

            for next_state in states:
                # print("parent state: ", output_format(next_state.parent))
                if is_final(next_state):
                    return next_state, moves
                else:
                    frontier.append(next_state)
    return None, -1


def dfs(state):
    """ dfs """
    explored = set()
    frontier = deque()
    frontier.append(state)

    while frontier:
        curr = frontier.pop()
        explore = " ".join([' '.join([str(c) for c in lst])
                           for lst in curr.get_data()])
        if explore not in explored:
            # print(output_format(curr), "not explored")
            explored.add(explore)

            states = next_states(curr)

            for next_state in states:
                if is_final(next_state):
                    return next_state, next_state.cost
                frontier.append(next_state)

    return None, -1


def read_file(filename):
    """ read the board from input file"""
    file = open(filename, "r")

    res = State()
    res.data = [[0 for _ in range(4)] for _ in range(5)]

    for i in range(5):
        row = file.readline()  # len = 4
        for j in range(4):
            res.data[i][j] = int(row[j])

    file.close()
    return res


def real_output_format(state):
    """ Convert curr state to the real format (1234) that print on output file """
    res = [[0 for _ in range(4)] for _ in range(5)]
    for i in range(5):
        for j in range(4):
            if state.data[i][j] == 1:
                res[i][j] = 1
            elif state.data[i][j] == 7:
                res[i][j] = 4
            elif state.data[i][j] == 0:
                res[i][j] = 0
            else:
                if j < 3 and state.data[i][j] == state.data[i][j + 1]:  # 1x2
                    res[i][j] = 2
                    res[i][j+1] = 2
                elif i < 4 and state.data[i][j] == state.data[i+1][j]:  # 2x1
                    res[i][j] = 3
                    res[i+1][j] = 3

    result = ""
    for i in range(5):
        for j in range(4):
            result += str(res[i][j])
        result += "\n"
    result += "\n"
    return result


def output_dfs_file(filename, state):
    """ print dfs output into file """

    file = open(filename, "w")

    file.write(f"Cost of the solution: {state.cost}\n")
    # file.write("\n")

    solution = []
    cost = state.cost
    while state:
        solution.append(state)
        state = state.parent
    for i in range(cost + 1):
        file.write(real_output_format(solution[cost - i]))

    file.close()


def output_astar_file(filename, state):
    """ print astar output into file """

    file = open(filename, "w")

    file.write(f"Cost of the solution: {state.cost}\n")
    # file.write("\n")

    solution = []
    cost = state.cost
    while state:
        solution.append(state)
        state = state.parent
    for i in range(cost + 1):
        file.write(real_output_format(solution[cost - i]))

    file.close()


if __name__ == '__main__':
    file = open(sys.argv[1])
    init = read_file(sys.argv[1])
    final, cost = dfs(init)
    output_dfs_file(sys.argv[2], final)
    final, cost = a_star(init)
    output_astar_file(sys.argv[3], final)

    # init = read_file("hrd_input3.txt")
    # final, cost = a_star(init)
    # output_astar_file("hrd_output3.txt", final)
