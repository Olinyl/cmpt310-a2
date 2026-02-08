# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        foodList: list = newFood.asList()
        
        # Food distance: the closer the food, the higher the score
        if foodList:
            minFoodDistance = min([manhattanDistance(newPos, food) for food in foodList])
            score += 1.0 / minFoodDistance
        # Ghost distance: the farther the ghost, the higher the score
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            ghostDistance = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer > 0:
                if ghostDistance > 0:
                    score += 1.0 / ghostDistance
            else:
                if ghostDistance == 0:
                    score -= 500 
                elif ghostDistance < 2:
                    score -= 100.0 / ghostDistance
                elif ghostDistance < 4:
                    score -= 10.0 / ghostDistance
        if action == Directions.STOP:
            score -= 5  

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(agentIndex, depth, gameState):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            
            legalActions = gameState.getLegalActions(agentIndex)

            isPacman = (agentIndex == 0)

            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth - 1 if nextAgentIndex == 0 else depth

            if isPacman:
                bestValue = float('-inf')
                bestAction = None
                for action in legalActions:
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    value, _ = minimax(nextAgentIndex, nextDepth, successorGameState)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            else:
                bestValue = float('inf')
                bestAction = None
                for action in legalActions:
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    value, _ = minimax(nextAgentIndex, nextDepth, successorGameState)
                    if value < bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            
        _, action = minimax(0, self.depth, gameState)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            
            legalActions = gameState.getLegalActions(agentIndex)
            isPacman = (agentIndex == 0)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth - 1 if nextAgentIndex == 0 else depth

            if isPacman:
                bestValue = float('-inf')
                bestAction = None
                for action in legalActions:
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    value, _ = alphabeta(nextAgentIndex, nextDepth, successorGameState, alpha, beta)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                    alpha = max(alpha, bestValue)
                    if beta < alpha:
                        break  
                return bestValue, bestAction
            else:
                bestValue = float('inf')
                bestAction = None
                for action in legalActions:
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    value, _ = alphabeta(nextAgentIndex, nextDepth, successorGameState, alpha, beta)
                    if value < bestValue:
                        bestValue = value
                        bestAction = action
                    beta = min(beta, bestValue)
                    if beta < alpha:
                        break  
                return bestValue, bestAction
        
        _, action = alphabeta(0, self.depth, gameState, float('-inf'), float('inf'))
        return action
                

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(agentIndex, depth, gameState):
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), None
            
            legalActions = gameState.getLegalActions(agentIndex)
            isPacman = (agentIndex == 0)
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth - 1 if nextAgentIndex == 0 else depth

            if isPacman:
                bestValue = float('-inf')
                bestAction = None
                for action in legalActions:
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    value, _ = expectimax(nextAgentIndex, nextDepth, successorGameState)
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            else:
                totalValue = 0
                for action in legalActions:
                    successorGameState = gameState.generateSuccessor(agentIndex, action)
                    value, _ = expectimax(nextAgentIndex, nextDepth, successorGameState)
                    totalValue += value
                expectedValue = totalValue / len(legalActions) if legalActions else 0
                return expectedValue, None
        
        _, action = expectimax(0, self.depth, gameState)
        return action

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()

    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    pacmanPos = currentGameState.getPacmanPosition()
    foodList: list = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    if foodList:
        minFoodDistance = min(manhattanDistance(pacmanPos, f) for f in foodList)
        score -= 1.5 * minFoodDistance

    score -= 4 * len(foodList)

    score -= 20 * len(capsules)

    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDistance = manhattanDistance(pacmanPos, ghostPos)
        if ghostState.scaredTimer > 0:
            if ghostDistance < ghostState.scaredTimer:
                score += 200.0 / (ghostDistance + 1)
        else:
            if ghostDistance < 2:
                score -= 200.0 / (ghostDistance + 1)
            elif ghostDistance < 4:
                score -= 10.0 / (ghostDistance + 1)
    return score

# Abbreviation
better = betterEvaluationFunction
