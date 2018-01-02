#Naive Bayes algorithm to classify spam
import sys
import math

if __name__ == '__main__':

    # validate the input arguments
    if (len(sys.argv) != 7):
        print "Wrong Parameters"
        sys.exit()
    training_file = sys.argv[2]
    testing_file = sys.argv[4]
    output_file = sys.argv[6]

    spam_dict = {}
    ham_dict = {}
    spamCount = 0
    hamCount = 0
    spamWordCount = 0
    hamWordCount = 0
    message = ""
    with open(training_file,"r") as f:
        emails = f.read().split("\n")
        for mail in emails:
            if len(mail) < 2:
                continue
            message = mail.split(" ")
            if len(message) > 2:
                if  message[1] == "ham" :
                    hamCount += 1
                    for i in range(2, len(message), 2):
                        try:
                            ham_dict[message[i]] += float(message[i+1])
                        except KeyError:
                            ham_dict[message[i]] = float(message[i+1])
                        hamWordCount += float(message[i+1])
                else:
                    spamCount += 1
                    for i in range(2, len(message), 2):
                        try:
                            spam_dict[message[i]] += float(message[i+1])
                        except KeyError:
                            spam_dict[message[i]] = float(message[i+1])
                        spamWordCount += float(message[i+1])
                
    vocabulary = set(spam_dict.keys()).union(set(ham_dict.keys()))
    vocabulary_size = len(vocabulary)
    spam_dict.update((k, float(v) + 0.05/ spamWordCount + 0.05*vocabulary_size) for k,v in spam_dict.items())
    ham_dict.update((k, float(v) + 0.05/ hamWordCount + 0.05*vocabulary_size) for k,v in ham_dict.items())

    total_Count = spamCount + hamCount
    p_spam = float(spamCount) / total_Count
    p_ham = float(hamCount) / total_Count

    correctPred = 0
    incorrectPred = 0
    resMessage = ""
    smoothingParametr = math.log(0.5)
    with open(testing_file,"r") as f:
        emails = f.read().split("\n")
        for mail in emails:
            if len(mail) < 2:
                continue
            message = mail.split(" ")
            if len(message) > 2:
                resMessage += message[0]
                actualResult = message[1]
                ham_prob = 1.0
                spam_prob = 1.0
                for i in range(2, len(message), 2):
                    
                    if len(message[i]) > 2:
                        if message[i] in spam_dict:
                            spam_prob += float(message[i+1]) * math.log(spam_dict[message[i]])
                        if message[i] in ham_dict:
                            ham_prob += float(message[i+1]) * math.log(ham_dict[message[i]])
                        else:
                        	spam_prob += smoothingParametr
                        	ham_prob += smoothingParametr
                spam_prob = spam_prob  + math.log(p_spam)
                ham_prob = ham_prob + math.log(p_ham)
                if ham_prob > spam_prob:
                    res = "ham"
                else:
                    res = "spam"
                if res == actualResult:
                    correctPred += 1
                else:
                    incorrectPred += 1
                resMessage += "," + res + "\n"

    with open(output_file,"w") as f:
        f.write(resMessage)

    print "Correct Labels: ", correctPred
    print "Incorrect Labels: ", incorrectPred
    print "Accuracy: ", (float(correctPred) * 100.0)/ (correctPred + incorrectPred), "%"
