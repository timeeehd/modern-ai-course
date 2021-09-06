### Code here!
import random
from copy import deepcopy
from typing import Sequence
from abc import ABC, abstractmethod
from collections import defaultdict
import math

NONE = '.'
MAX = 'X'
MIN = 'O'
COLS = 7
ROWS = 6
N_WIN = 4


class ArrayState:
    def __init__(self, board, heights, n_moves):
        self.board = board
        self.heights = heights
        self.n_moves = n_moves

    @staticmethod
    def init():
        board = [[NONE] * ROWS for _ in range(COLS)]
        return ArrayState(board, [0] * COLS, 0)


def result(state: ArrayState, action: int) -> ArrayState:
    """Insert in the given column."""
    assert 0 <= action < COLS, "action must be a column number"

    if state.heights[action] >= ROWS:
        raise Exception('Column is full')

    player = MAX if state.n_moves % 2 == 0 else MIN

    board = deepcopy(state.board)
    board[action][ROWS - state.heights[action] - 1] = player

    heights = deepcopy(state.heights)
    heights[action] += 1

    return ArrayState(board, heights, state.n_moves + 1)


def undo(state: ArrayState, action: int) -> ArrayState:
    """Insert in the given column."""
    assert 0 <= action < COLS, "action must be a column number"

    if state.heights[action] >= ROWS:
        raise Exception('Column is full')

    player = MAX if state.n_moves % 2 == 0 else MIN

    board = deepcopy(state.board)
    board[action][ROWS - state.heights[action] - 1] = '.'

    heights = deepcopy(state.heights)
    heights[action] -= 1

    return ArrayState(board, heights, state.n_moves + 1)


def actions(state: ArrayState) -> Sequence[int]:
    return [i for i in range(COLS) if state.heights[i] < ROWS]


def utility(state: ArrayState) -> float:
    """Get the winner on the current board."""

    board = state.board

    def diagonalsPos():
        """Get positive diagonals, going from bottom-left to top-right."""
        for di in ([(j, i - j) for j in range(COLS)] for i in range(COLS + ROWS - 1)):
            yield [board[i][j] for i, j in di if i >= 0 and j >= 0 and i < COLS and j < ROWS]

    def diagonalsNeg():
        """Get negative diagonals, going from top-left to bottom-right."""
        for di in ([(j, i - COLS + j + 1) for j in range(COLS)] for i in range(COLS + ROWS - 1)):
            yield [board[i][j] for i, j in di if i >= 0 and j >= 0 and i < COLS and j < ROWS]

    lines = board + \
            list(zip(*board)) + \
            list(diagonalsNeg()) + \
            list(diagonalsPos())

    max_win = MAX * N_WIN
    min_win = MIN * N_WIN
    for line in lines:
        str_line = "".join(line)
        if max_win in str_line:
            return 1
        elif min_win in str_line:
            return -1
    return 0


def terminal_test(state: ArrayState) -> bool:
    return state.n_moves >= COLS * ROWS or utility(state) != 0


def printBoard(state: ArrayState):
    board = state.board
    """Print the board."""
    print('  '.join(map(str, range(COLS))))
    for y in range(ROWS):
        print('  '.join(str(board[x][y]) for x in range(COLS)))
    print()


class MCTS:
    "Monte Carlo tree searcher."

    def __init__(self, exploration_weight=1, simulations=100):
        self.exploration_weight = exploration_weight
        self.tree = {}
        self.visit_count = {}
        self.wins = {}
        self.total_visit = 0
        self.simulations = simulations

    def choose(self, state: ArrayState) -> int:
        "Choose  a move in the game and execute it"
        self.tree[str(state.board)] = 'root'
        self.visit_count[str(state.board)] = 0
        self.wins[str(state.board)] = 0
        for i in range(self.simulations):
            self.do_rollout(state)
            self.total_visit += 1
        return self._uct_select(state)

    def do_rollout(self, state: ArrayState):
        "Train for one iteration."
        node = self._select(state)
        result = self._simulate(node)
        self._backpropagate(node, result)
        pass

    def _select(self, state: ArrayState):
        "Find an unexplored descendent of the `state`"
        s = deepcopy(state)
        while not terminal_test(s):
            for a in actions(s):
                board = deepcopy(s)
                board = result(board, a)
                if str(board.board) not in self.tree:
                    return self._expand(board, a)
            a = self._uct_select(s)
            s = result(s, a)
        return s

    def _expand(self, state: ArrayState, action: int):
        "Expand the `state` with all states reachable from it"
        self.tree[str(state.board)] = action
        self.visit_count[str(state.board)] = 0
        self.wins[str(state.board)] = 0
        return state

    def _simulate(self, state: ArrayState):
        "Returns the reward for a random simulation (to completion) of the `state`"
        s = deepcopy(state)
        while not terminal_test(s):
            a = random.choice(actions(s))
            s = result(s, a)
        return utility(s)

    def _backpropagate(self, state, reward):
        "Send the reward back up to the ancestors of the leaf"
        parent = deepcopy(state)
        action = self.tree[str(parent.board)]
        while action != 'root':
            self.visit_count[str(parent.board)] += 1
            self.wins[str(parent.board)] += reward
            parent = undo(parent, self.tree[str(parent.board)])
            action = self.tree[str(parent.board)]
        self.visit_count[str(parent.board)] += 1
        self.wins[str(parent.board)] += reward
        pass

    def _uct_select(self, state: ArrayState):
        "Select a child of state, balancing exploration & exploitation"
        best_value = -100000
        best_action = None
        for a in actions(state):
            board = deepcopy(state)
            board = result(board, a)
            UCT_value = (self.wins[str(board.board)] / self.visit_count[str(board.board)]) + self.exploration_weight * \
                        math.sqrt((2 * math.log(self.total_visit)) / self.visit_count[str(board.board)])
            if UCT_value > best_value:
                best_action = a
                best_value = UCT_value
        return best_action


if __name__ == '__main__':
    s = ArrayState.init()
    turn = 0
    while not terminal_test(s):
        if (turn % 2) == 0:
            agent = MCTS(1)
            a = agent.choose(s)
        # a = random.choice(actions(s))
        else:
            action = input("Which colomn do you wanna to put your chip in? ")
            a = int(action)
        s = result(s, a)
        printBoard(s)
        turn += 1
        print("Turn: " + str(turn))
    print(utility(s))
