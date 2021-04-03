"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):

    counter = 0

    for i in range(3):
        for j in range(3):
            if board[i][j] is not None:
                counter += 1

    if not terminal(board):
        if counter % 2 == 0:
            return X
        else:
            return O
    else:
        return X
    raise NotImplementedError


def actions(board):

    result = set()

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] is EMPTY:
                result.add(tuple([i, j]))

    return result
    raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    possible_move = actions(board)
    assigned_player = player(board)

    if len(possible_move) != 0:

        if action not in possible_move:
            raise Exception("This is an invalid move")
        else:
            boardII = copy.deepcopy(board)
            boardII[action[0]][action[1]] = assigned_player
            return boardII

    else:
        boardII = board.copy()
        return boardII

    raise NotImplementedError


def winner(board):

    final_winner = None

    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2]:
            if board[i][0] is not None:
                final_winner = board[i][0]
                break
        elif board[0][i] == board[1][i] == board[2][i]:
            if board[0][i] is not None:
                final_winner = board[0][i]
                break

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        final_winner = board[0][0]

    elif board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        final_winner = board[0][2]

    return final_winner

    raise NotImplementedError


def terminal(board):

    #create counter to check if the board is filled
    counter = 0

    for i in board:
        for j in i:
            if j is not None:
                counter += 1

    if winner(board) is None:
        if counter >= 9:
            return True
        else:
            return False
    else:
        return True

    raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0

    raise NotImplementedError


def MAX_VALUE(board):
    if terminal(board):
        return utility(board)

    v = -math.inf
    for action in actions(board):
        v = max(v, MIN_VALUE(result(board, action)))

    return v


def MIN_VALUE(board):

    if terminal(board):
        return utility(board)

    v = math.inf
    for action in actions(board):
        v = min(v, MAX_VALUE(result(board, action)))

    return v


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """

    actionsTuple = list(actions(board))

    v = []

    if player(board) == X:
        for action in actionsTuple:
            v.append(MIN_VALUE(result(board, action)))

        return actionsTuple[v.index(max(v))]

    if player(board) == O:
        for action in actionsTuple:
            v.append(MAX_VALUE(result(board, action)))

        return actionsTuple[v.index(min(v))]

    raise NotImplementedError
