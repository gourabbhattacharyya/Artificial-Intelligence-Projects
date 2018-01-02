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
import random, util, sys

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    def evaluationFunction(self, currentGameState, action):
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

        #foodGrid = newFood.asList()
        newFood = currentGameState.getFood()
        foodGrid = newFood.asList()
        newPosList = list(newPos)
        dist = -1 * sys.maxint

        if action == 'Stop':
            return dist
        else:
            for ghostState in newGhostStates:
                if ghostState.scaredTimer == 0:
                    tempNewPos = tuple(newPosList)
                    if ghostState.getPosition() == tempNewPos:
                        return dist


            for food in foodGrid:
                maxDist = util.manhattanDistance(food, newPosList)
                #print "Max Dist", maxDist
                maxDist = -1 * maxDist
                if maxDist > dist:
                    dist = maxDist

        return dist

def scoreEvaluationFunction(currentGameState):
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

    def MaxValue(self, gameState, depth):

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"

        else:
            validMoves = gameState.getLegalActions()
            maxScoreList = []
            moveOfScores = {}
            bestMove = ""

            for move in validMoves:
                nextState = gameState.generateSuccessor(self.index, move)
                newScore, newMove = self.MinValue(nextState, 1, depth)

                moveOfScores[(newScore, newMove)] = move
                maxScoreList.append((newScore, newMove))

            maxScore = max(maxScoreList)

            for score in maxScoreList:
                if cmp(score, maxScore) == 0:
                    bestMove = moveOfScores[score]
                    break

            return maxScore, bestMove


    def MinValue(self, gameState, agent, depth):

        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), "STOP"

        else:
            validMoves = gameState.getLegalActions(agent)
            minScoreList = []
            moveOfScores = {}
            bestMove = ""

            for move in validMoves:
                nextState = gameState.generateSuccessor(agent, move)

                if (agent != gameState.getNumAgents() - 1):
                    newScore, newMove = self.MinValue(nextState, agent + 1, depth)

                else:
                    newScore, newMove = self.MaxValue(nextState, (depth - 1))

                moveOfScores[(newScore, newMove)] = move
                minScoreList.append((newScore, newMove))

            minScore = min(minScoreList)

            for score in minScoreList:
                if cmp(score, minScore) == 0:
                    bestMove = moveOfScores[score]
                    break

            return minScore, bestMove



    def getAction(self, gameState):
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
        """
        "*** YOUR CODE HERE ***"


        #gameScore, bestMove = self.MaxValue(gameState, self.depth)
        #print "BestMove", bestMove, gameScore
        #print "Score and BestMove", self.MaxValue(gameState, self.depth)[0], self.MaxValue(gameState, self.depth)[1]
        return self.MaxValue(gameState, self.depth)[1]

        #util.raiseNotDefined()



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        alpha = -1 * sys.maxint     #define alpha to min possible value
        beta = sys.maxint           #define beta to min possible value

        gameScore, bestMove = self.AlphaBetaPruning(gameState, 0, 0, alpha, beta)   #call wrapper function

        #print "Best Move", bestMove
        return bestMove

        # util.raiseNotDefined()


    def AlphaBetaPruning(self, gameState, agent, depth, alpha, beta):       #Wrapper function

        if agent >= gameState.getNumAgents():       #if ghost exhaust
            agent = 0
            depth = depth + 1


        if depth == self.depth or gameState.isWin() or gameState.isLose():      #test for terminal node
            return self.evaluationFunction(gameState), Directions.STOP

        elif agent == 0:                                                        #For Pacman call MaxValueAlphaBeta
            return self.MaxValueAlphaBeta(gameState, agent, depth, alpha, beta)

        else:                                                                   #For Ghost call MinValueAlphaBeta
            return self.MinValueAlphaBeta(gameState, agent, depth, alpha, beta)



    def MaxValueAlphaBeta(self, gameState, agent, depth, alpha, beta):              #Max value for Pacman

        validMoves = gameState.getLegalActions(agent)       #get all the valid moves
        bestMove = ""
        maxScore = -1 * sys.maxint                          #define maxscore

        if len(validMoves) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        else:
            for move in validMoves:
                nextState = gameState.generateSuccessor(agent, move)                #get the successor state
                newScore = self.AlphaBetaPruning(nextState, (agent + 1), depth, alpha, beta)[0]     #get the score for the successor state

                if newScore > maxScore:
                    maxScore = newScore
                    bestMove = move

                if newScore > beta:                   #return if alpha is greater than beta
                    return newScore, move

                if newScore > alpha:
                    alpha = newScore

            return maxScore, bestMove                 #return score and action


    def MinValueAlphaBeta(self, gameState, agent, depth, alpha, beta):                #Min value for Pacman
        validMoves = gameState.getLegalActions(agent)           #get all the valid moves
        minScore = sys.maxint                                   #define minscore
        bestMove = ""

        if len(validMoves) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        else:
            for move in validMoves:
                nextState = gameState.generateSuccessor(agent, move)            #get the score for the successor state

                newScore = self.AlphaBetaPruning(nextState, (agent + 1), depth, alpha, beta)[0]         #get the score for the successor state

                if newScore < minScore:
                    minScore = newScore
                    bestMove = move

                if newScore < alpha:                    #return if alpha is greater than beta
                    return newScore, move

                if newScore < beta:
                    beta = newScore

            return minScore, bestMove                   #return score and action




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        gameScore, bestMove = self.getExpectimax(gameState, 0, 0)       #call the wrapper function

        #print "Best Move", bestMove
        return bestMove

        # util.raiseNotDefined()


    def getExpectimax(self, gameState, agent, depth):           #define wrapper function

        if agent >= gameState.getNumAgents():               #check if ghost states exhaust
            agent = 0
            depth = depth + 1


        if depth == self.depth or gameState.isWin() or gameState.isLose():      #test for terminal state
            return self.evaluationFunction(gameState), Directions.STOP

        elif agent == 0:                                #call pacman for max value of expectimax
            return self.MaxValueExpectimax(gameState, agent, depth)

        else:                                           #call pacman for min value of expectimax
            return self.MinValueExpectimax(gameState, agent, depth)


    def MaxValueExpectimax(self, gameState, agent, depth):          #call max value function

        validMoves = gameState.getLegalActions(agent)               #get all the valid moves
        bestMove = ""
        maxScore = -1 * sys.maxint

        if len(validMoves) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        else:

            for move in validMoves:
                nextState = gameState.generateSuccessor(agent, move)            #generate the successor state
                newScore = self.getExpectimax(nextState, (agent + 1), depth)[0]         #get score for successor state

                if newScore > maxScore:
                    maxScore = newScore
                    bestMove = move

            return maxScore, bestMove                       #return score and action



    def MinValueExpectimax(self, gameState, agent, depth):

        validMoves = gameState.getLegalActions(agent)
        minScore = 0
        bestMove = ""

        if len(validMoves) == 0:
            return self.evaluationFunction(gameState), Directions.STOP

        else:

            for move in validMoves:
                nextState = gameState.generateSuccessor(agent, move)                    #generate the successor state
                newScore = self.getExpectimax(nextState, (agent + 1), depth)[0]         #get score for successor state

                minScore = minScore + (newScore * (1.0/len(validMoves)))                #get the total minvalue
                bestMove = move

            return minScore, bestMove                       #return score and action



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


