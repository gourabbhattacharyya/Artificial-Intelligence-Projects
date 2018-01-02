import time
import Queue
import copy
import random



def backtracking_search(csp):       #This is generic backtrack search
    return backtrack(csp.assignment, csp)


def backtrack(assignment, csp):        #This will take assignment and CSP object as input and do a backtrack through all the nodes

    if csp.complete(assignment): #if the assignment is complete and valid, return the result
        return assignment

    var = csp.selectUnassignedVariables()
    domainVals = csp.findDomainValues(var, assignment)

    for value in domainVals:
        csp.counter += 1                                # increase the counter for each consistency check

        if csp.consistent(var, value, assignment):      #if value is consistent
            assignment[var] = value
            domain = []

            for idx in range(csp.n):
                tempList = []
                for idx in range(csp.d):
                    tempList.append(1)
                domain.append(tempList)



            for i in range(len(csp.domain)):
                for j in range(len(csp.domain[i])):
                    domain[i][j] = csp.domain[i][j]


            inferences = csp.inference(var, value)

            if inferences is not False:                 #add inferences to assignment
                for inference in inferences:
                    assignment[inference[0]] = inference[1]

                #print "Count, Assignment", csp.counter, assignment
                result = backtrack(assignment, csp)

                if result:
                    return result


            assignment[var] = None                      #remove {var = value} and inferences from assignment


            for i in range(len(domain)):                #restore the domain values
                for j in range(len(domain[i])):
                    csp.domain[i][j] = domain[i][j]



            if inferences is not False:
                for inference in inferences:
                    assignment[inference[0]] = None
    return False



def ac_3(csp, queue):          #This is used for Constraint Propagation

    while (queue.empty() == False):

        (item1, item2) = queue.get()                        # if the variable x_i is assigned, we can skip the tuple with certainty

        if csp.assignment[item1] is not None:
            continue


        if reviseNodes(csp, item1, item2):

            if csp.domain[item1].count(1) == 0:
                return False


            neighbors = set()

            for idx in range(len(csp.domain)):
                if item1 == idx:
                    continue

                neighbors.add(idx)


            if item2 in neighbors:
                neighbors.remove(item2)

            for item in neighbors:
                queue.put((item, item1))

    return set() # for consistency of the general backtrack algorithm


def reviseNodes(csp, item1, item2):     #if an variable is present in a domain the returns true

    revised = False
    for idx1 in range(len(csp.domain[item1])):

        if csp.domain[item1][idx1] == 0:
            continue

        satisfy = False

        assignment = copy.deepcopy(csp.assignment)          #construct the right assignment
        assignment[item1] = idx1

        if csp.assignment[item2] is not None:
            if csp.consistent(item2, csp.assignment[item2], assignment):
                satisfy = True

        else:
            for idx2 in range(len(csp.domain[item2])):
                if csp.consistent(item2, idx2, assignment):
                    satisfy = True


        if not satisfy:
            csp.domain[item1][idx1] = 0
            revised = True

    return revised




class CSPBackTrack:         #This is used for backtrack functionality

    n = 0       #number of variables
    d = 0       #domain of variables
    counter = 0
    consistentDict = dict()


    def __init__(self, ruler):
        self.n = ruler.M
        self.d = ruler.L + 1
        self.domain = []
        self.assignment = dict()

        for order in range(self.n):
            tempList = []
            for item in range(self.d):
                tempList.append(1)
            self.domain.append(tempList)


        for i in range(self.n):
            self.assignment[i] = None



    def complete(self, assignment):     #check for assignment completeness

        for i in range(self.n):
            if assignment[i] == None:
                return False

        return True



    def selectUnassignedVariables(self):       #Find the index of first any unassigned variable

        for i in range(self.n):
            if self.assignment[i] is None:
                return i



    def findDomainValues(self, item, assignment):       #find possible values in the domain

        possibleValues = []

        if item is not None:
            for i in range(len(self.domain[item])):
                if self.domain[item][i] != 0:
                    possibleValues.append(i)

        return possibleValues



    def consistent(self, item, value, sourceAssignment):        #check is the new assignment is consistent

        assignment = {}
        distance = []

        for i in range(0, len(sourceAssignment)):
            if i == item:
                assignment[i] = value
            else:
                assignment[i] = sourceAssignment[i]


        if tuple(assignment.items()) in self.consistentDict:
            return self.consistentDict[tuple(assignment.items())]


        for i in range(1, len(assignment)):
            if assignment[i] is None or assignment[i-1] is None:
                continue

            if not (assignment[i] > assignment[i-1]):
                self.consistentDict[tuple(assignment.items())] = False
                return False



        for i in range(self.n):
            for j in range(self.n - 1 - i):
                b = assignment[self.n - 1 - i]
                a = assignment[(self.n - 1 - i) - (j + 1)]
                if b is None or a is None:
                    continue

                distance.append(b-a)

        distance.sort()

        temp = None

        for i in range(len(distance)):
            if distance[i] == temp:
                self.consistentDict[tuple(assignment.items())] = False
                return False
            temp = distance[i]


        self.consistentDict[tuple(assignment.items())] = True

        return True


    def inference(self, item, value):       #find out the inference for the new assignment

        newVal = set()
        return newVal






class CSPBTForwardCheck(CSPBackTrack):          #This is used for Backtrack with Forward Check

    def __init__(self, ruler):
        self.n = ruler.M
        self.d = ruler.L + 1
        self.domain = []
        self.assignment = dict()

        for order in range(self.n):
            tempList = []
            for item in range(self.d):
                tempList.append(1)
            self.domain.append(tempList)


        # node consistency
        for i in range(self.n):
            temp_list = []
            if i == 0:
                for x in range(self.d):
                    temp_list.append(0)
                temp_list[0] = 1
                self.domain[i] = temp_list

            elif i == self.n - 1:
                for x in range(self.d):
                    temp_list.append(0)
                temp_list[self.d - 1] = 1
                self.domain[i] = temp_list
            else:
                for x in range(self.d):
                    temp_list.append(1)

                for j in range(i + 1, self.n):
                    temp_list[self.d - self.n + j] = 0
                self.domain[i] = temp_list


        # initialize the assignment
        for i in range(self.n):
            self.assignment[i] = None



    def inference(self, item, value):

        inferenced = set()

        for i in range(len(self.domain)):

            if i == item or self.assignment[i] is not None:
                continue


            for j in range(len(self.domain[i])):
                if self.domain[i][j] == 0:
                    continue

                if not (self.consistent(i, j, self.assignment)):
                    self.domain[i][j] = 0


            if self.domain[i].count(1) == 0:
                return False
            elif self.domain[i].count(1) == 1:
                inferenced.add((i, self.domain[i].index(1)))

        return inferenced





class CSPBTConsProp(CSPBTForwardCheck):         #This is used for Backtrack with Constraint Propagation

    def inference(self, item, value):

        queue = Queue.Queue()

        for i in range(len(self.domain)):

            if i != item and self.assignment[i] is None:
                queue.put((i, item))


        return ac_3(self, queue)      #call for contraint propagation









#Define Golomb Ruler class and set order M and length L for the CSP
class Golomb_Ruler:
    M = 0  # Order
    L = 0  # Length

    def __init__(self, M, L):
        self.M = M
        self.L = L





def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2 - time1) * 1000.0))
        print("=====================end of trace=====================")
        #print('\n')
        return ret

    return wrap


@timing
def algoTime(csp):
    print("====================start of trace====================")
    print("Running " + csp.__class__.__name__ + " algorithm")
    m = backtracking_search(csp)
    print("Number of consistency checks: "+str(csp.counter))
    if m is not False:
        for i in range(len(m)):
            print("M" + str(i) + ": " + str(m[i]))

        return True, m
    else:
        print("No solution")
        return False, m



#do not modify the function names
#You are given L and M as input
#Each of your functions should return the minimum possible L value alongside the marker positions
#Or return -1,[] if no solution exists for the given L

#Your backtracking function implementation
def BT(L, M):
    "*** YOUR CODE HERE ***"

    # The set of Golomb Rulers

    gRuler = Golomb_Ruler(M, L)
    csp_bt = CSPBackTrack(gRuler)

    print('\n')
    print("Implementation of CSP for the problem Golomb Ruler by Gourab Bhattacharyya - 170048888")
    print ("Golomb Ruler - M: " + str(gRuler.M) + ", L: " + str(gRuler.L))

    result, pathDict = algoTime(csp_bt)

    if result:
        resultList = []
        for key, value in pathDict.iteritems():
            resultList.append(value)
        print "Final Result", (len(pathDict), resultList)
        print('\n')
        return (len(pathDict), resultList)
    else:
        print "Final Result", (-1,[])
        print('\n')
        return (-1,[])


#Your backtracking+Forward checking function implementation
def FC(L, M):
    "*** YOUR CODE HERE ***"

    gRuler = Golomb_Ruler(M, L)
    csp_bt_fc = CSPBTForwardCheck(gRuler)

    print('\n')
    print("Implementation of CSP for the problem Golomb Ruler by Gourab Bhattacharyya - 170048888")
    print ("Golomb Ruler - M: " + str(gRuler.M) + ", L: " + str(gRuler.L))

    result, pathDict = algoTime(csp_bt_fc)

    if result:
        resultList = []
        for key, value in pathDict.iteritems():
            resultList.append(value)
        print "Final Result", (len(pathDict), resultList)
        print('\n')
        return (len(pathDict), resultList)
    else:
        print "Final Result", (-1,[])
        print('\n')
        return (-1,[])


#Bonus: backtracking + constraint propagation
def CP(L, M):
    "*** YOUR CODE HERE ***"

    gRuler = Golomb_Ruler(M, L)
    csp_bt_cp = CSPBTConsProp(gRuler)

    print('\n')
    print("Implementation of CSP for the problem Golomb Ruler by Gourab Bhattacharyya - 170048888")
    print ("Golomb Ruler - M: " + str(gRuler.M) + ", L: " + str(gRuler.L))

    result, pathDict = algoTime(csp_bt_cp)

    if result:
        resultList = []
        for key, value in pathDict.iteritems():
            resultList.append(value)
        print "Final Result", (len(pathDict), resultList)
        print('\n')
        return (len(pathDict), resultList)
    else:
        print "Final Result", (-1,[])
        print('\n')
        return (-1,[])






#Test
#---------
BT(25, 7)
FC(25, 7)
CP(25, 7)