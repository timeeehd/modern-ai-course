from agents import *
from pacman import *


class ZeroIntelligent(BaseAgent):
    class State:
        def __init__(self):
            self.actions = ["GoRight", "GoLeft", "GoForward", "GoBack"]

    def choose_action(self, state):
        action = random.choice(state.actions)
        print("Performing action:", action)
        return action


class Intelligent(BaseAgent):
    class State:
        def __init__(self):
            self.bump = False
            self.previous_action = ""
            self.actions = ["GoRight", "GoLeft", "GoForward", "GoBack"]

        def __repr__(self):
            if self.bump:
                return self.previous_action + " resulted in a bump"
            else:
                return self.previous_action

    def update_state_with_percept(self, percept, state):
        if percept[1] == "bump":
            state.bump = True
        else:
            state.bump = False

        test = GameState
        return state

    def choose_action(self, state):
        actions = state.actions
        if state.bump:
            actions.remove(state.previous_action)
        return random.choice(actions)

    def update_state_with_action(self, action, state):
        state.previous_action = action
        # Print the representation (i.e. __repr__) of the state
        print(state)
        return state


class Intelligent2(BaseAgent):
    class State:
        def __init__(self,state):
            self.bump = False
            self.previous_action = ""
            self.actions = ["GoRight", "GoLeft", "GoForward", "GoBack"]

        def __repr__(self):
            if self.bump:
                return self.previous_action + " resulted in a bump"
            else:
                return self.previous_action

    # def registerInitialState(self, state):
    #     """AgentState is stored in state"""
    #     self._dir = 0
    #     self._dirsMap = {(1,0):'East',(0,-1):'South',(-1,0):'West',(0,1):'North'}
    #     self._dirs = [(1,0),(0,-1),(-1,0),(0,1)]
    #     self._percept = ('clear', None)
    #     self._actions = ['GoRight','GoLeft','GoForward','GoBack']
    #     self._state = state

    def update_state_with_percept(self, percept, state):
        if percept[1] == "bump":
            state.bump = True
        else:
            state.bump = False

        test = GameState
        return state

    def choose_action(self, state):
        actions = state.actions
        if state.bump:
            actions.remove(state.previous_action)
        return random.choice(actions)

    def update_state_with_action(self, action, state):
        state.previous_action = action
        # Print the representation (i.e. __repr__) of the state
        print(state)
        return state

class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

# Run an agent in the room layout "layouts/custom.lay"
args = readCommand(["--pacman", Intelligent2,
                    "--layout", "mediumEmpty"])
runGames(**args)
