from math import log
from scipy import stats
import sys
import csv
import numpy as np
import pickle as pkl
import argparse


# ==================================ID3Tree Class starts==================================
class ID3Tree:

    def __init__(self, threshold):
        # reference to training set and label
        self.threshold = threshold
        self.number_nodes = 0

    def trainDTree(self, train, label):
        '''
        train a decision tree by training set and labels
        :param train: training data
        :param label: label for training data
        '''

        print "Log::Start Training of Tree"
        used = []
        for i in range(0, len(train[0])):
            used.append(False)

        self.root = self.generateTree(train, label, used)
        print "Log::Done Training"
        return self.root

    def generateTree(self, train, label, used):
        '''
        :param train: generate a tree
        :param label:
        :return: a node that is the root of the subtree
        '''

        root = TreeNode('F')  # data=str(-1))
        tmp_label, isSame = self.getMajority(label)
        # add one node here
        self.number_nodes += 1

        # if all the feature have been used
        if False not in used:
            print "No feature to use"
            # mark this node as leaf node
            root.setLeaf()

            # set predict label as majority label
            root.setLabel(tmp_label)

            return root

        elif isSame:
            # if the label are all the same
            root.setLeaf()
            root.setLabel(tmp_label)

            return root

        else:
            # choose the best feature to classify
            best_index = self.chooseBestFeature(train, label, used)
            print "Choose feature::%d as best one" % best_index

            if best_index >= len(train[0]):
                print "Big error::best feature index exceeds limit"
                exit()

            # set the best_index as feature_index
            root.setFeatureIndex(best_index)

            split_dataset = {}
            # traversing for every value for this feature
            for val in self.getValues(train, best_index):
                # generate subtree here
                sub_train, sub_label, new_used = self.splitDataSet(train, best_index, label, used, val)
                split_dataset[val] = [sub_train, sub_label, new_used]

            # use the chi-square test to test relevance
            # if irrelevant just stops here and uses majority as label
            num_pos = 0.0
            num_neg = 0.0
            for l in label:
                if l == 0:
                    num_neg += 1
                else:
                    num_pos += 1

            S = 0.0
            for k, v in split_dataset.iteritems():
                pi = num_pos * len(v[1]) / float(len(train))
                ni = num_neg * len(v[1]) / float(len(train))
                r_pi = 0.0
                r_ni = 0.0

                for l in v[1]:
                    if l == 0:
                        r_ni += 1
                    else:
                        r_pi += 1

                # add to S
                tmp = 0
                if r_pi != 0:
                    tmp += pow(r_pi - pi, 2) / r_pi

                if r_ni != 0:
                    tmp += pow(r_ni - ni, 2) / r_ni

                S += tmp

            # compute the p_value by scipy
            p_value = 1 - stats.chi2.cdf(S, len(split_dataset))
            self.threshold = np.float64(self.threshold)
            # print "P_val, Threshold", p_value, p_value.dtype, self.threshold, self.threshold.dtype

            if p_value < self.threshold:
                count = 0
                print "chi-square p-value is: ", p_value, self.threshold
                for k, v in split_dataset.iteritems():
                    # if count == 5:
                    #    break

                    child = self.generateTree(v[0], v[1], v[2])
                    # add child to this root node
                    root.addChild(child, k)

                    # count += 1

            else:
                root.setLeaf()
                root.setLabel(tmp_label)
                return root

        return root

    def getMajority(self, label):
        '''
        get the majority label for current labels
        :param label: label vector for training set
        :return: 0 or 1 as the majority label, and if all the label are the same
        '''
        pos_num = 0.0
        neg_num = 0.0
        for l in label:
            if l == 0:
                neg_num += 1
            elif l == 1:
                pos_num += 1
            else:
                print "Error::wrong label not 1 or 0!"
                exit()

        '''
        if pos_num == neg_num:
            return 1, True
        '''
        if pos_num == 0 or neg_num == 0:
            if pos_num > neg_num:
                return 1, True
            else:
                return 0, True
        else:
            # in training set there are more negative set than positive example
            if neg_num > pos_num:  # (neg_num/pos_num) > (32193.0 / 7807.0):
                return 0, False
            else:
                return 1, False


    def chooseBestFeature(self, train, label, used_feature):
        # choose the best feature which maximize entropy gain. used_feature is a list of bool of dimension of training set size

        # total dimension of feature
        n = len(train[0])
        # entropy under current label
        ent = self.targetEntropy(label)
        max = -1
        max_index = -1

        # find the largest gain
        for i in range(0, n):
            # if this feature is no longer available
            if used_feature[i]:
                continue

            # find largest gain
            f_entropy = self.featureEntropy(train, i, label)
            gain = ent - f_entropy
            if gain < -1e-10:
                print "Big error::gain smaller than 0"
                print gain
                exit()

            if gain > max:
                max = gain
                max_index = i

        return max_index

    def targetEntropy(self, label):

        pos = 0.0
        neg = 0.0
        # total = len(label)

        for l in label:
            if l == 0:
                neg += 1
            elif l == 1:
                pos += 1
            else:
                print "Error::Strange case happen!"
                exit()

        entropyPos = 0.0
        entropyNeg = 0.0
        total = pos + neg
        if pos != 0:
            entropyPos = -1 * (float(pos) / total) * log((float(pos) / total), 2)
        if neg != 0:
            entropyNeg = -1 * (float(neg) / total) * log((float(neg) / total), 2)

        entropy = entropyPos + entropyNeg
        return entropy

    def featureEntropy(self, train, f_index, label):
        # If split on f_index attribute, how much gain can I get
        # collect how many value and their frequency
        val_freq = {}
        total = len(train)
        for feature in train:
            val = feature[f_index]
            if val in val_freq:
                val_freq[val] += 1.0
            else:
                val_freq[val] = 1.0

        # now compute the entropy for each of this value and average them by frequency
        f_entropy = 0.0
        for k, v in val_freq.iteritems():
            weight = v / total
            sublabel = []
            for f, l in zip(train, label):
                if f[f_index] == k:
                    sublabel.append(l)
            f_entropy += weight * self.targetEntropy(sublabel)
        return f_entropy

    def getValues(self, train, index):
        '''
        :param train: training data
        :param index: which feature to extract all the possible values from
        :return: a list containing all the possible value for this feature
        '''
        values = []
        for sample in train:
            val = sample[index]
            if val not in values:
                values.append(val)

        return values

    def splitDataSet(self, train, f_index, label, used, f_value):
        '''
        Split the training set: extract subset whose f_index value is f_value
        And extract subset of label as well. Also update the used list
        :param train: whole training set to be split
        :param f_index: based on which feature should we split
        :param label: the label set to be split
        :param used: which feature have we used now, update used[f_index] to be True
        :param f_value: feature value of f_index
        :return: subset for train and subset for label
        '''

        sub_train = []
        sub_label = []
        new_used = used[:]

        for t, l in zip(train, label):
            if t[f_index] == f_value:
                # collect sample where feature value is f_value
                sub_train.append(t)
                sub_label.append(l)

        new_used[f_index] = True
        return sub_train, sub_label, new_used

    def predictSet(self, test_feature):

        predict = []
        cnt = 0
        for f in test_feature:
            cnt += 1

            if (cnt % 50) == 0:
                print "Test %d sample" % cnt

            label = self.predictOne(f)
            predict.append(label)

        print "Done predict test set"

        return predict

    def predictOne(self, sample):
        ''' use feature to predict label for new sample
        :param sample: a feature vector for a new sample
        :return: 0 or 1 as predict result
        '''

        tmp_node = self.root

        # if we can not judge the label
        # we should loop
        stop = tmp_node.canJudge()
        while not stop:
            f_index = tmp_node.getFeatureIndex()

            # the value for the sample in this feature
            f_val = sample[f_index]
            tmp_node = tmp_node.getNext(f_val)
            stop = tmp_node.canJudge()

        # if current node can judge the label
        return tmp_node.predictLabel()


# ==================================ID3Tree Class ends==================================
'''
TreeNode represents a node in your decision tree
TreeNode can be:
    - A non-leaf node: 
        - data: contains the feature number this node is using to split the data
        - children[0]-children[4]: Each correspond to one of the values that the feature can take

    - A leaf node:
        - data: 'T' or 'F' 
        - children[0]-children[4]: Does not matter, you can leave them the same or cast to None.

'''


# ==================================Node Class starts==================================
# DO NOT CHANGE THIS CLASS
class TreeNode():
    # return data representation for pickled object
    def __getstate__(self):
        org_dict = self.__dict__  # get attribute dictionary
        del org_dict['judgelabel']  # remove judgelabel entry
        del org_dict['nodeValues']  # remove nodeValues entry
        return org_dict

    def __init__(self, data='T', children=[-1] * 5):
        self.nodes = list(children)  # children
        self.data = data
        self.judgelabel = 0
        self.nodeValues = {}

    def save_tree(self, filename):
        obj = open(filename, 'w')
        pkl.dump(self, obj)

    def setLeaf(self):
        self.data = 'F'

    def setLabel(self, label):
        self.judgelabel = label

    def setFeatureIndex(self, f_index):
        self.data = f_index

    def getFeatureIndex(self):
        return self.data

    def canJudge(self):
        if self.data == 'F' or self.data == 'T':
            return True
        else:
            return False

    def predictLabel(self):
        if self.judgelabel != 0 and self.judgelabel != 1:
            print "The label is strange!", self.judgelabel

        return self.judgelabel

    def getNext(self, value):

        '''
        Given value, find correspoding child node, and return it
        :param value: value of the feature
        :return: return the child node corresponds to value
        '''

        isFound = False
        goto_value = value

        while isFound == False:
            for k, v in self.nodeValues.iteritems():
                if k == goto_value:
                    # print "getNext ", k
                    isFound = True
                    return v
            goto_value -= 1
            if goto_value == 0:
                break

        while isFound == False:
            for k, v in self.nodeValues.iteritems():
                if k == goto_value:
                    # print "getNext ", k
                    isFound = True
                    return v
            goto_value += 1

        return None

    def addChild(self, child, value):
        '''
        Add a child tree to the root
        :param child: the child node to be added
        :param value:
        :return:
        '''

        for i in range(len(self.nodes)):
            if self.nodes[i] == -1:
                self.nodes[i] = TreeNode('F')

        isValueFound = False

        for k, v in self.nodeValues.iteritems():
            if k == value:
                isValueFound = True

        if isValueFound == False:
            self.nodes.append(value) #child
            self.nodeValues[value] = child

        else:
            print "Error::this value already has a child!"
            exit()


# ==================================Node Class starts==================================



# ==================================Reader Class starts==================================
class Reader:
    def __init__(self, ftrain, ftest):
        self.ftrain = ftrain
        self.ftest = ftest

    def readData(self):

        Xtrain, Ytrain, Xtest = [], [], []
        with open(self.ftrain, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                rw = map(int, row[0].split())
                Xtrain.append(rw)

        with open(self.ftest, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                rw = map(int, row[0].split())
                Xtest.append(rw)

        ftrain_label = self.ftrain.split('.')[0] + '_label.csv'
        with open(ftrain_label, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                rw = int(row[0])
                Ytrain.append(rw)

        print('Data Loading: done')
        return Xtrain, Ytrain, Xtest


# ==================================Reader Class ends==================================


# ==================================Main method starts==================================
if __name__ == "__main__":
    print "Log::started reading data!"

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', required=True)
    parser.add_argument('-f1', help='training file in csv format', required=True)
    parser.add_argument('-f2', help='test file in csv format', required=True)
    parser.add_argument('-o', help='output labels for the test dataset', required=True)
    parser.add_argument('-t', help='output tree filename', required=True)

    args = vars(parser.parse_args())

    pval = args['p']
    Xtrain_name = args['f1']
    Ytrain_name = args['f1'].split('.')[
                      0] + '_labels.csv'  # labels filename will be the same as training file name but with _label at the end

    Xtest_name = args['f2']
    Ytest_predict_name = args['o']

    tree_name = args['t']

    reader = Reader(Xtrain_name, Xtest_name)
    Xtrain, Ytrain, Xtest = reader.readData()

    print "Log::finished reading data!"

    dc_tree = ID3Tree(pval)
    rootNode = dc_tree.trainDTree(Xtrain, Ytrain)

    predict = dc_tree.predictSet(Xtest)

    print "Node in the decision tree is: %d" % dc_tree.number_nodes

    print "Prediction is : ", len(predict)

    rootNode.save_tree(tree_name)
    # write the prediction result in csv file
    '''
    with open(outputFileName, "w") as f:
        wr = csv.writer(f, delimiter="\n")
        wr.writerow(predict)
    '''

    with open(Ytest_predict_name, "wb") as f:
        writer = csv.writer(f, delimiter="\n")
        writer.writerow(predict)
    print("Output files generated")


    # ==================================Main method ends==================================
