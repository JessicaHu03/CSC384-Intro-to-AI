from collections import defaultdict
from copy import deepcopy
from ctypes.wintypes import DWORD
import heapq
from http.client import OK
import sys

ships = {"S": 0, "D": 0, "C": 0, "B": 0}


class State:
    """A state is a table of all the tiles with given initial locations. 
    """

    def __init__(self, dim, domain=None, data=None, value=None, parent=None):
        self.dim = dim
        self.domain = domain
        self.data = data
        self.value = value
        self.parent = parent
        if self.parent:
            self.cost = parent.cost + 1
        else:
            self.cost = 0

    # for node ordering

    def __gt__(self, state):
        return self.cost > state.cost

    def __lt__(self, state):
        return self.cost < state.cost

    def get_domain(self):
        return self.domain

    def get_data(self):
        """ return curr data"""
        return self.data

    def get_parent(self):
        """ return the parent of curr state"""
        return self.parent

    def set_value(self, value):
        self.value = value


def preprocessing(state):  # checked
    update_data(state)
    # print("after update_data: \n", print_data(state))
    update_const(state)
    # print("after update_const: \n", print_data(state))
    domains(state)


def update_data(state):  # checked
    """ 
    update the grid
    add M to L, R, T, B
    surround ships with W
    """
    data = state.data
    dim = state.dim
    for i in range(dim):
        for j in range(dim):

            # skip this square to boost efficiency
            if data[i][j] == '0':
                continue
            # if single submarine, surround it with water
            if data[i][j] == 'S':
                if i != 0:
                    data[i-1][j] = 'W'
                    if j != 0:
                        data[i-1][j-1] = 'W'
                    if j != dim-1:
                        data[i-1][j+1] = 'W'
                if j != 0:
                    data[i][j-1] = 'W'
                if j != dim-1:
                    data[i][j+1] = 'W'
                if i != dim-1:
                    data[i+1][j] = 'W'
                    if j != 0:
                        data[i+1][j-1] = 'W'
                    if j != dim-1:
                        data[i+1][j+1] = 'W'
            # if left end, add M to its right and filled needed squares with W
            elif data[i][j] == 'L':
                # add W's
                if i != 0:
                    data[i-1][j] = 'W'
                    if j != 0:
                        data[i-1][j-1] = 'W'
                    if j != dim-1:
                        data[i-1][j+1] = 'W'
                if j != 0:
                    data[i][j-1] = 'W'
                if i != dim-1:
                    data[i+1][j] = 'W'
                    if j != 0:
                        data[i+1][j-1] = 'W'
                    if j != dim-1:
                        data[i+1][j+1] = 'W'

            # if right end, add M to its left and filled needed squares with W
            elif data[i][j] == 'R':
                if i != 0:
                    data[i-1][j] = 'W'
                    if j != 0:
                        data[i-1][j-1] = 'W'
                    if j != dim-1:
                        data[i-1][j+1] = 'W'
                if j != dim-1:
                    data[i][j+1] = 'W'
                if i != dim-1:
                    data[i+1][j] = 'W'
                    if j != 0:
                        data[i+1][j-1] = 'W'
                    if j != dim-1:
                        data[i+1][j+1] = 'W'

            # if top end, add M to its bottom and filled needed squares with W
            elif data[i][j] == 'T':
                if i != 0:
                    data[i-1][j] = 'W'
                    if j != 0:
                        data[i-1][j-1] = 'W'
                    if j != dim-1:
                        data[i-1][j+1] = 'W'
                if j != 0:
                    data[i][j-1] = 'W'
                if j != dim-1:
                    data[i][j+1] = 'W'
                if i != dim-1:
                    # # either M or B, depend on ship & col constraint
                    # def num_filled(data, type):
                    #     for i in range(len(data[0])):
                    #         for j in range(len(data[0])):
                    #             if
                    #         if data[i][j]
                    # if ships['Des'] > 1 and (col_const[i]-num_filled()) == 0:

                    # data[i+1][j] = 'M'
                    if j != 0:
                        data[i+1][j-1] = 'W'
                    if j != dim-1:
                        data[i+1][j+1] = 'W'

            # if bottom end, add M to its top and filled needed squares with W
            elif data[i][j] == 'B':
                if i != 0:
                    if j != 0:
                        data[i-1][j-1] = 'W'
                    if j != dim-1:
                        data[i-1][j+1] = 'W'
                if j != 0:
                    data[i][j-1] = 'W'
                if j != dim-1:
                    data[i][j+1] = 'W'
                if i != dim-1:
                    data[i+1][j] = 'W'
                    if j != 0:
                        data[i+1][j-1] = 'W'
                    if j != dim-1:
                        data[i+1][j+1] = 'W'

# def update_domain(state):
#     """ update the domain matrix based on data"""
#     data = state.data
#     domain = state.domain

#     for i in range(state.dim):
#         for j in range(state.dim):
#             if data[i][j] != '0':
#                 domain[i][j] = data[i][j]
#             else:
#                 continue


def update_const(state):  # checked
    """
    update the board based on row, col, ships const 
    """
    data = state.data
    dim = state.dim

    if check_ships(state.data):
        # ships number matched, filled the rest of the board with W
        # print("ships checked")
        for i in range(state.dim):
            for j in range(state.dim):
                if data[i][j] == '0':
                    data[i][j] = 'W'
        return

    for i in range(len(row_const)):
        if check_row_const(data, i):
            # row_num matches, need to filled out the rest squares on this row with W
            # print("row number " + str(i) + " is satisfied")
            for j in range(dim):
                if data[i][j] == '0':
                    data[i][j] = 'W'
        if check_col_const(data, i):
            # col_num matches, need to filled out the rest squares on this row with W
            # print("col number " + str(i) + " is satified")
            for j in range(dim):
                if data[j][i] == '0':
                    data[j][i] = 'W'

# def next_state(state):
#     data = state.data
#     dim = state.dim
#     for i in range(dim):
#         for j in range(dim):
#             if data[i][j] == 'L':
#                 # there must be a M at
#                 new_data = deepcopy(data)
#                 new_data[i][j+2] = 'B'
#                 if check_row_const(new_data, i):


def check_row_const(data, row_num):
    """ check if the current board falsify the row constraint"""
    row = data[row_num]
    total = len(row)
    for i in range(len(row)):
        if row[i] == '0' or row[i] == 'W':
            total -= 1
    if total == row_const[row_num]:
        return True
    else:
        return False


def check_col_const(data, col_num):
    """ check if the current board falsify the col constraint"""
    col = [data[i][col_num] for i in range(len(data))]
    total = len(col)
    for i in range(len(col)):
        if col[i] == '0' or col[i] == 'W':
            total -= 1
    if total == col_const[col_num]:
        return True
    else:
        return False


def check_ships(data):
    """ check if the current board falsify the ships constraint"""
    sub_num = 0
    des_num = 0
    crui_num = 0
    bat_num = 0
    length = len(data[0])
    for i in range(length):
        for j in range(length):
            if data[i][j] == '0' or data[i][j] == 'W':
                continue
            if data[i][j] == 'S':  # submarine 1x1
                sub_num += 1
            elif data[i][j] == 'T':  # vertical destroyer 1x2 or cruiser 1x3 or battleship 1x4
                if data[i+1][j] == 'M':  # cruiser 1x3 or battleship 1x4
                    if data[i+2][j] == 'M':  # battleship 1x4
                        bat_num += 1
                    elif data[i+2][j] == 'B':  # cruiser 1x3
                        crui_num += 1
                elif data[i+1][j] == 'B':  # destroyer 1x2
                    des_num += 1
            elif data[i][j] == 'L':  # horizontal destroyer 1x2 or cruiser 1x3 or battleship 1x4
                if data[i][j+1] == 'M':  # cruiser 1x3 or battleship 1x4
                    if data[i][j+2] == 'M':  # battleship 1x4
                        bat_num += 1
                    elif data[i][j+2] == 'R':  # cruiser 1x3
                        crui_num += 1
                elif data[i][j+1] == 'R':  # destroyer 1x2
                    des_num += 1

    if sub_num == ships["S"] and des_num == ships["D"] and crui_num == ships["C"] and bat_num == ships["B"]:
        # the ships number match
        return True
    else:
        return False


def domains(state):
    """ 
    add all possible move (domain) for each ship
    should have checked that there is bat ship num >= 1 before calling this function"""
    data = state.data
    domain = state.domain
    for i in range():
        for j in range():
            if data[i][j] == '0' and data[i+1][j] == '0' and data[i+2][j] == '0' and data[i+3][j] == '0':
                # there is space to place a vertical battleship
                domain["B"].append([[i, j], [i+1, j], [i+2, j], [i+3, j]])
            if data[i][j] == '0' and data[i][j+1] == '0' and data[i][j+2] == '0' and data[i][j+3] == '0':
                # there is space to place a horizontal battleship
                domain["B"].append([[i, j], [i, j+1], [i, j+2], [i, j+3]])
            if data[i][j] == '0' and data[i+1][j] == '0' and data[i+2][j] == '0':
                # there is space to place a vertical cruiser
                domain["C"].append([[i, j], [i+1, j], [i+2, j]])
            if data[i][j] == '0' and data[i][j+1] == '0' and data[i][j+2] == '0':
                # there is space to place a horizontal cruiser
                domain["C"].append([[i, j], [i, j+1], [i, j+2]])
            if data[i][j] == '0' and data[i+1][j] == '0':
                # there is space to place a vertical destroyer
                domain["D"].append([[i, j], [i+1, j]])
            if data[i][j] == '0' and data[i][j+1] == '0':
                # there is space to place a horizontal destroyer
                domain["D"].append([[i, j], [i, j+1]])
            if data[i][j] == '0':
                # there is space to place a submarine
                domain["S"].append([[i, j]])


def check_orientation(positions):
    """ determine the orientation of given positions """
    old_y, old_x = positions[0][0], positions[0][1]
    new_y, new_x = positions[1][0], positions[1][1]
    if new_y > old_y and new_x == old_x:  # vertical
        return 1
    elif new_y == old_y and new_x > old_x:  # horizontal
        return 0


def insert_into_board(data, var, pos):
    """ set the given variable into given position """
    ori = check_orientation(pos)
    # pos = all of the squares need to be filled up with 'T', 'B', 'L', 'R', 'M' or 'S'
    if var == 'B':  # a battleship, 1x4
        if ori == 0:  # horizontal
            y, x = pos[0][0], pos[0][1]
            data[y][x] = 'L'
            y, x = pos[1][0], pos[1][1]
            data[y][x] = 'M'
            y, x = pos[2][0], pos[2][1]
            data[y][x] = 'M'
            y, x = pos[3][0], pos[3][1]
            data[y][x] = 'R'
        else:  # vertical
            y, x = pos[0][0], pos[0][1]
            data[y][x] = 'T'
            y, x = pos[1][0], pos[1][1]
            data[y][x] = 'M'
            y, x = pos[2][0], pos[2][1]
            data[y][x] = 'M'
            y, x = pos[3][0], pos[3][1]
            data[y][x] = 'B'
    elif var == 'C':  # cruiser, 1x3
        if ori == 0:  # horizontal
            y, x = pos[0][0], pos[0][1]
            data[y][x] = 'L'
            y, x = pos[1][0], pos[1][1]
            data[y][x] = 'M'
            y, x = pos[2][0], pos[2][1]
            data[y][x] = 'R'
        else:  # vertical
            y, x = pos[0][0], pos[0][1]
            data[y][x] = 'T'
            y, x = pos[1][0], pos[1][1]
            data[y][x] = 'M'
            y, x = pos[2][0], pos[2][1]
            data[y][x] = 'B'
    elif var == 'D':  # destroyer, 1x2
        if ori == 0:  # horizontal
            y, x = pos[0][0], pos[0][1]
            data[y][x] = 'L'
            y, x = pos[1][0], pos[1][1]
            data[y][x] = 'R'
        else:  # vertical
            y, x = pos[0][0], pos[0][1]
            data[y][x] = 'T'
            y, x = pos[1][0], pos[1][1]
            data[y][x] = 'B'
    elif var == 'S':  # submarine, 1x1
        y, x = pos[0][0], pos[0][1]
        data[y][x] = 'S'


def find_constraints(data, var, pos):
    """ find the constraints that is related to var """
    res = []
    ori = check_orientation(pos)  # 0 = horizontal, 1 = vertical

    # row_const

    # col_const
    # ships
    res.append()
    # surround by 0 or W

    def safe_region(data, var, pos):
        res = []
        dim = len(data[0])
        if var == 'B':
            y0, x0 = pos[0][0], pos[0][1]
            y1, x1 = pos[1][0], pos[1][1]
            y2, x2 = pos[2][0], pos[2][1]
            y3, x3 = pos[3][0], pos[3][1]
            if ori == 0:  # horizontal
                if y0 != 0:
                    if x0 != 0:
                        res.append(data[y0-1][x0-1] ==
                                   'W' or data[y0-1][x0-1] == '0')
                    if x0 != dim-1:
                        data[y0-1][x0+1] = 'W'
                if x0 != 0:
                    data[y0][x0-1] = 'W'
                if x0 != dim-1:
                    data[y0][x0+1] = 'W'
                if y0 != dim-1:
                    data[y0+1][x0] = 'W'
                    if x0 != 0:
                        data[y0+1][x0-1] = 'W'
                    if x0 != dim-1:
                        data[y0+1][x0+1] = 'W'

    res.append()

##################################### BackTrack, FC, GAC algorithms ########################


def all_assigned(state):  # checked
    data = state.data
    for i in range(state.dim):
        for j in range(state.dim):
            if data[i][j] == '0':
                return False
    return True


def MRV(state):
    """" return the name of the unassigned variable with the least CurDom"""
    min = float('inf')
    min_var = ""
    for x in state.domain:
        if len(state.domain[x]) < min:  # if its len(domain) < min
            min = len(state.domain[x])
            min_var = str(x)         # update the least domain
    return min_var


def FCCheck(state, const, ship_type):
    domain = state.domain
    # const is a constraint with all its variables already assigned, except for variable x
    for pos in domain[ship_type]:
        new_data = deepcopy(state.data)
        insert_into_board(new_data, ship_type, pos)
        # if making x = val with previous assignments to variables in scope C falsifies C
        if not const(new_data):
            # remove d from CurDom(x)
            domain[ship_type].remove(pos)
    if len(domain[ship_type]) == 0:
        return 'DWO'
    return 'OK'


def FC(state, level):
    """ forward checking """
    if all_assigned(state):
        print(print_data(state))
        return

    # pick an unassigned variable that has the least domain
    var = MRV(state)
    assigned[var] = True
    preprocessing(state)
    data = state.data

    for val in state.domain[var]:
        DWO = False

        stored_domain = deepcopy(state.domain)
        new_data = deepcopy(state.data)
        insert_into_board(new_data, var, val)  # new board with the ship placed
        new_domain = deepcopy(state.domain)
        new_domain[var].remove(val)
        child = State(parent=state, data=new_data,
                      domain=new_domain, dim=state.dim)

        # all constraints related to this var
        consts = find_constraints(new_data, var, val)

        while consts:
            const = consts.pop()
            # C has only one unassigned variable X in its scope
            if FCCheck(child, const, domain) == 'DWO':
                DWO = True
                break
        if not DWO:  # all constraints consistent
            FC(state, level+1)
        state.domain = stored_domain
    assigned[var_y][var_x] = False  # undo since we have tried all of V's value
    return


def GAC():
    pass


######################################## read, output file ###############################
def print_data(state):  # check
    result = ""
    data = state.data
    length = len(data[0])
    for i in range(length):
        for j in range(length):
            result += str(data[i][j])
        result += "\n"
    return result


def print_domain(state):  # check
    result = ""
    domain = state.domain
    length = len(domain[0])
    for i in range(length):
        for j in range(length):
            result += str(domain[i][j])
        result += "\n"
    return result


def to_list(string):
    res = [int(string[i]) for i in range(len(string)-1)]
    return res


def read_file(filename):
    """ read the state from input file"""
    file = open(filename, "r")

    row_line = file.readline()
    global row_const
    row_const = to_list(row_line)
    # print(row_const)

    col_line = file.readline()
    global col_const
    col_const = to_list(col_line)
    # print(col_const)

    ships_line = file.readline()
    ship_const = to_list(ships_line)
    if len(ship_const) == 1:
        ships["S"] = ship_const[0]
    if len(ship_const) == 2:
        ships["S"] = ship_const[0]
        ships["D"] = ship_const[1]
    if len(ship_const) == 3:
        ships["S"] = ship_const[0]
        ships["D"] = ship_const[1]
        ships["C"] = ship_const[2]
    if len(ship_const) == 4:
        ships["S"] = ship_const[0]
        ships["D"] = ship_const[1]
        ships["C"] = ship_const[2]
        ships["B"] = ship_const[3]
    # print(ships)

    length = len(row_line)-1
    # print(length)

    new_data = [["." for _ in range(length)] for _ in range(length)]
    for i in range(length):
        row = file.readline()
        for j in range(length):
            new_data[i][j] = str(row[j])

    domain = [[dom for _ in range(length)] for _ in range(length)]

    # domain (list of possible locations) of each ship
    domain = defaultdict(list)
    domain["S"] = []
    domain["D"] = []
    domain["C"] = []
    domain["B"] = []

    global assigned
    assigned = [[False for _ in range(length)] for _ in range(length)]

    # update the initial state of domain and assigned
    for i in range(length):
        for j in range(length):
            if new_data[i][j] != '0':
                domain[i][j] = [new_data[i][j]]
                assigned[i][j] = True

    res = State(domain=domain, dim=length, data=new_data)

    file.close()
    return res


def output_file(filename, state):
    """ print dfs output into file """

    file = open(filename, "w")

    # next_state = Minimax(state)

    file.write(print_data(next_state))

    file.close()


if __name__ == '__main__':

    # init = read_file(sys.argv[1])
    # output_file(sys.argv[2], init)

    init = read_file("test_grid.txt")
    print("init: \n", print_data(init))

    # preprocessing(init)
    print(MRV(init))
    # print("final: \n", print_data(init))

    # FC(init, 0)
    # print(print_data(init))
