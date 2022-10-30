from copy import deepcopy
from ctypes.wintypes import DWORD
import heapq
from http.client import OK
import sys

ships = {"Sub": 0, "Des": 0, "Crui": 0, "Bat": 0}
dom = ['S', 'W', 'L', 'R', 'T', 'B', 'M']


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


def perprocessing(state):  # checked
    # if there's 0 in row or col constraints, filled with W
    data = state.data
    for i in range(state.dim):
        if row_const[i] == 0:
            for j in range(state.dim):
                data[i][j] = 'W'
        if col_const[i] == 0:
            for j in range(state.dim):
                data[j][i] = 'W'
    # update board with known squares
    update_data(state)


def update_data(state):  # checked
    """ 
    update the grid & constraints
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
                        data[i+1][j+1] == 'W'
            # if left end, add M to its right and filled needed squares with W
            elif data[i][j] == 'L':
                if i != 0:
                    data[i-1][j] = 'W'
                    if j != 0:
                        data[i-1][j-1] = 'W'
                    if j != dim-1:
                        data[i-1][j+1] = 'W'
                if j != 0:
                    data[i][j-1] = 'W'
                if j != dim-1:
                    data[i][j+1] = 'M'
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
                if j != 0:
                    data[i][j-1] = 'M'
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
                    data[i+1][j] = 'M'
                    if j != 0:
                        data[i+1][j-1] = 'W'
                    if j != dim-1:
                        data[i+1][j+1] = 'W'

            # if bottom end, add M to its top and filled needed squares with W
            elif data[i][j] == 'B':
                if i != 0:
                    data[i-1][j] = 'M'
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


def update_constraints(state):
    """
    update the board 
    """
    data = state.data
    dim = state.dim
    pass


def check_row_const(data, row_num, const):
    """ check if the current board falsify the row constraint"""
    row = data[row_num]
    total = len(row)
    for i in range(len(row)):
        if row[i] == '0' or row[i] == 'W':
            total -= 1
    if total <= const:
        return True
    else:
        return False


def check_col_const(data, col_num, const):
    """ check if the current board falsify the col constraint"""
    col = [data[i][col_num] for i in range(len(data))]
    total = len(col)
    for i in range(len(col)):
        if col[i] == '0' or col[i] == 'W':
            total -= 1
    if total <= const:
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
    if sub_num <= ships["Sub"] and des_num <= ships["Des"] and crui_num <= ships["Crui"] and bat_num <= ships["Bat"]:
        return True
    else:
        return False


##################################### BackTrack, FC, GAC algorithms ########################

def BackTrack():
    pass


def all_assigned(state):  # checked
    data = state.data
    for i in range(state.dim):
        for j in range(state.dim):
            if data[i][j] == '0':
                return False
    return True


def MRV(state):
    """" return the position of the unassigned variable with the least CurDom"""
    x, y = 0, 0
    min = float('inf')
    for i in range(state.dim):
        for j in range(state.dim):
            if state.data[i][j] == '0':  # for an unassigned variable
                if len(state.domain[i][j]) < min:  # if its len(domain) < min
                    y, x = i, j         # update the least domian x, y
    return [y, x]


def FCCheck(state, const, pos):
    y, x = pos[0], pos[1]
    domain = state.domain
    # const is a constraint with all its variables already assigned, except for variable x
    for val in state.domain[y][x]:
        new_data = deepcopy(state.domain)
        new_data[y][x] = val
        # if making x = val with previous assignments to variables in scope C falsifies C
        if (const == row_const and not check_row_const(new_data, y, const[y])) or (const == col_const and not check_col_const(new_data, x, const[x])) or (const == ships and not check_ships(state.data)):
            # remove d from CurDom(x)
            domain[y][x].remove(val)
    if len(domain[y][x]) == 0:
        return 'DWO'
    return 'OK'


def FC(state, level):
    """ forward checking """
    if all_assigned(state):
        print(output_format(state))
        return

    # pick an unassigned variable that has the least domain
    var = MRV(state)
    var_y, var_x = var[0], var[1]
    assigned[var_y][var_x] = True
    consts = [col_const, row_const, ships]
    for val in state.domain[var_y][var_x]:
        DWO = False
        stored_domain = deepcopy(state.domain)
        # new_data = deepcopy(state.data)
        state.data[var_y][var_x] = val
        while consts:
            const = heapq.heappop(consts)
            # C has only one unassigned variable X in its scope
            if FCCheck(const, var) == 'DWO':
                DWO = True
                break
        if not DWO:  # all constraints consistent
            FC(level+1)
        state.domain = stored_domain
    assigned = False  # undo since we have tried all of V's value
    return


def GAC():
    pass


######################################## read, output file ###############################
def output_format(state):  # check
    result = ""
    data = state.data
    length = len(data[0])
    for i in range(length):
        for j in range(length):
            result += str(state.data[i][j])
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
        ships["Sub"] = ship_const[0]
    if len(ship_const) == 2:
        ships["Sub"] = ship_const[0]
        ships["Des"] = ship_const[1]
    if len(ship_const) == 3:
        ships["Sub"] = ship_const[0]
        ships["Des"] = ship_const[1]
        ships["Crui"] = ship_const[2]
    if len(ship_const) == 4:
        ships["Sub"] = ship_const[0]
        ships["Des"] = ship_const[1]
        ships["Crui"] = ship_const[2]
        ships["Bat"] = ship_const[3]
    # print(ships)

    length = len(row_line)-1
    # print(length)

    new_data = [["." for _ in range(length)] for _ in range(length)]
    for i in range(length):
        row = file.readline()
        for j in range(length):
            new_data[i][j] = str(row[j])

    domain = [[dom for _ in range(length)] for _ in range(length)]

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

    file.write(output_format(next_state))

    file.close()


if __name__ == '__main__':

    # init = read_file(sys.argv[1])
    # output_file(sys.argv[2], init)

    init = read_file("test_grid.txt")
    print(output_format(init))
    FC(init, 0)
    print(output_format(init))
