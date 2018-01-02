# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"

    sourceNode = (problem.getStartState(), None, 0)
    nodesVisited = []
    backTrackDict = {problem.getStartState() : None}
    fringeList = util.Stack()
    fringeList.push(sourceNode)
    action = []
    closed = []

    while fringeList.isEmpty() != True:

            currentNode = fringeList.pop()
            nodesVisited.append(currentNode[0])

            if problem.isGoalState(currentNode[0]):
                goalState = currentNode
                break

            successorNodes = problem.getSuccessors(currentNode[0])
            for next in successorNodes:
                if next[0] not in nodesVisited:
                    backTrackDict[next[0]] = currentNode

                    """if problem.isGoalState(next[0]):
                        goalState = next
                        goalStateFound = True
                        break"""

                    fringeList.push(next)

    while backTrackDict[goalState[0]] != None:
        if goalState[1] != None:
            action.append(goalState[1])
        goalState = backTrackDict[goalState[0]]

    return list(reversed(action))

    #util.raiseNotDefined()



def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    sourceNode = (problem.getStartState(), None, 0)
    nodesVisited = [problem.getStartState()]
    closed = []
    backTrackDict = {problem.getStartState(): None}
    fringeList = util.Queue()
    fringeList.push(sourceNode)
    action = []

    while fringeList.isEmpty() != True:
            currentNode = fringeList.pop()

            if problem.isGoalState(currentNode[0]):
                goalState = currentNode
                break


            if currentNode[0] not in closed:
                successorNodes = problem.getSuccessors(currentNode[0])
                closed.append(currentNode[0])

                for next in successorNodes:
                    if next[0] not in nodesVisited:
                        backTrackDict[next[0]] = currentNode

                        """if problem.isGoalState(next[0]):
                            goalState = next
                            goalStateFound = True
                            break"""

                        fringeList.push(next)
                        nodesVisited.append(next[0])



    while backTrackDict[goalState[0]] != None:
        if goalState[1] != None:
            action.append(goalState[1])
        goalState = backTrackDict[goalState[0]]

    return list(reversed(action))
    #util.raiseNotDefined()





def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    sourceNode = (problem.getStartState(), None, 0)
    nodesVisited = []
    backTrackDict = {problem.getStartState(): None}
    fringePList = util.PriorityQueue()
    fringePList.push(sourceNode, sourceNode[2])
    nodeCost = {problem.getStartState() : sourceNode[2]}
    action = []


    while fringePList.isEmpty() != True:
            currentNode = fringePList.pop()
            nodesVisited.append(currentNode[0])

            if problem.isGoalState(currentNode[0]):
                goalState = currentNode
                break

            successorNodes = problem.getSuccessors(currentNode[0])
            for next in successorNodes:
                if next[0] not in nodesVisited:

                    totalCost = nodeCost[currentNode[0]] + next[2]

                    if next[0] in nodeCost:
                        if nodeCost[next[0]] < totalCost:
                            continue    #search for optimal path
                        else:
                            fringePList.update(next, totalCost)
                    else:
                        fringePList.push(next, totalCost)

                    backTrackDict[next[0]] = currentNode
                    nodeCost[next[0]] = totalCost

                    """if problem.isGoalState(next[0]):
                        goalState = next
                        goalStateFound = True
                        break"""

    while backTrackDict[goalState[0]] != None:
        if goalState[1] != None:
            action.append(goalState[1])
        goalState = backTrackDict[goalState[0]]

    return list(reversed(action))

    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    sourceNode = (problem.getStartState(), None, 0)
    backTrackDict = {problem.getStartState(): None}

    fVal = {}
    fVal[sourceNode] = heuristic(problem.getStartState(), problem)

    fringeAList = util.PriorityQueue()
    fringeAList.push(sourceNode, fVal[sourceNode])

    nodesVisited = []
    nodeCost = {problem.getStartState(): sourceNode[2]}
    closed = []
    action = []


    while fringeAList.isEmpty() != True:
            currentNode = fringeAList.pop()
            nodesVisited.append(currentNode[0])

            if problem.isGoalState(currentNode[0]):
                goalState = currentNode
                break

            successorNodes = problem.getSuccessors(currentNode[0])
            for next in successorNodes:
                if next[0] not in nodesVisited:

                    totalCost = nodeCost[currentNode[0]] + next[2]
                    fVal[next] = totalCost + heuristic(next[0], problem)

                    if next[0] in nodeCost:
                        if nodeCost[next[0]] < totalCost:
                            continue
                        else:
                            fringeAList.update(next, fVal[next])
                    else:
                        fringeAList.push(next, fVal[next])

                    backTrackDict[next[0]] = currentNode
                    nodeCost[next[0]] = totalCost


    while backTrackDict[goalState[0]] != None:
        if goalState[1] != None:
            action.append(goalState[1])
        goalState = backTrackDict[goalState[0]]

    return list(reversed(action))



    #util.raiseNotDefined()

def checkIteminQ(fringList, item):
    items = []
    if fringList.isEmpty():
        return items
    else:
        currentItem = fringList.pop()
        items.append(currentItem[0])
        return items






# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch