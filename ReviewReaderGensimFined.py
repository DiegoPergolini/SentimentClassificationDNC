import json
import numpy as np
import random
import gensim
import nltk
import tensorflow as tf

# ==============================================================================
################################# AUTHOR: ######################################
############################# DIEGO PERGOLINI ##################################
# ==============================================================================



def getBalancedReviewsAndOverall(trainingInstances, testingInstances,
                                 fileName, max_lenght, randomized=False, seed=9):
    listT = []
    listTe = []
    in_file = open(fileName, "r")
    numTrain = 0
    numTest = 0
    countTrain = [0, 0, 0, 0, 0]
    countTest = [0, 0, 0, 0, 0]
    i = 0
    ranT = random.Random(seed)
    ranTe = random.Random(seed)
    while i < trainingInstances + testingInstances:
        raw = in_file.readline()
        if(i%100000==0):
            print (i)
            print (countTrain)
            print (countTest)
        if raw == "":
            break
            print(countTrain)
            print (countTest)
            print(numTrain)
            print(numTest)
        jsonLine = json.loads(raw)

        review = jsonLine['reviewText']
        index = int(jsonLine['overall']) - 1
        num_words = len(gensim.utils.simple_preprocess(review))
        if num_words <= max_lenght:
            if randomized:
                if random.Random(seed).randint(0, 1) == 1:
                    if countTrain[index] < (trainingInstances/5):
                        countTrain[index]+=1
                        current = ((review), jsonLine['overall'])
                        listT.append(current)
                        numTrain += 1
                        i+=1
                    elif countTest[index] < (testingInstances/5):
                        countTest[index]+=1
                        current = ((review), jsonLine['overall'])
                        listTe.append(current)
                        numTest += 1
                        i+=1
            else:
                    if countTrain[index] < (trainingInstances/5):
                        countTrain[index]+=1
                        current = (( review), jsonLine['overall'])
                        listT.append(current)
                        numTrain += 1
                        i+=1
                    elif countTest[index] < (testingInstances/5):
                        countTest[index]+=1
                        current = ((review), jsonLine['overall'])
                        listTe.append(current)
                        numTest += 1
                        i+=1

    print(countTrain)
    print (countTest)
    print(numTrain)
    print(numTest)

    ranT.shuffle(listT)
    ranTe.shuffle(listTe)
    in_file.close()
    return listT, listTe

def getCrossDomainReviewsAndOverall(trainingInstances, testingInstances, fileTraining, fileTesting, max_lenght, randomized=False,
                                    seed=9):
    listT, _ = getBalancedReviewsAndOverall(trainingInstances, 0, fileTraining, max_lenght, randomized, seed)
    _, listTe = getBalancedReviewsAndOverall(0,testingInstances,fileTesting, max_lenght, randomized, seed)
    return listT, listTe


def getW2vRepresentation(text,loadedModel):
    countDeleted = 0
    toReturn = []
    for element in gensim.utils.simple_preprocess(text):
        try:
            toReturn.append(loadedModel[element].tolist())
        except KeyError:
            countDeleted+=1
    return toReturn

def balancedGetDataset(allLines, modelName, FLAGS):
    wordSize = FLAGS.word_dimension
    batch_size = FLAGS.batch_size
    max_lenght = FLAGS.max_lenght
    loadedModel = gensim.models.KeyedVectors.load_word2vec_format(modelName, binary=True)

    random.shuffle(allLines[0])
    random.shuffle(allLines[1])

    batched_input = []
    batched_label = []
    batched_mask = []

    i = 0
    num_batch = 0
    batch_completed = False
    while i < (len(allLines[0]) + 1):
        if not batch_completed:
            if len(batched_input) < batch_size:
                data = getW2vRepresentation(allLines[0][i][0], loadedModel)
                data = data[:max_lenght]
                if len(data) > 0:
                    padded = np.lib.pad(data, ((max_lenght - len(data), 0), (0, 0)), 'constant', constant_values=(0))
                    delta = 1.0 / (len(data))
                else:
                    padded = np.zeros(shape=[max_lenght, wordSize])
                    delta = 1.0
                batched_input.append(padded)
                if True:
                    label = np.zeros(shape=[max_lenght, 5])
                    index = int(allLines[0][i][1]) - 1
                    for j in range(0, max_lenght):
                        np.put(label[j], index, 1.0)
                    batched_label.append(label)
                    mask = np.zeros(max_lenght)
                    weight = np.zeros(len(data))
                    for number in range(0, len(data)):
                        np.put(weight, number, delta * (number + 1))
                    np.put(mask, np.arange(max_lenght - len(data), max_lenght), weight)
                    batched_mask.append(mask)
                    i += 1
            else:
                reshaped = np.reshape(batched_input, newshape=[batch_size, max_lenght, wordSize])
                labels = np.reshape(batched_label, newshape=[batch_size, max_lenght, 5])
                masks = np.reshape(batched_mask, newshape=[batch_size, max_lenght])
                toAppend = (reshaped, labels, masks)

                num_batch += 1
                if (len(allLines[0]) // batch_size) == num_batch:
                    batch_completed = True
                batched_input = []
                batched_label = []
                batched_mask = []
                yield toAppend
        else:
            break
    i = 0
    batch_completed = False
    num_batch = 0
    while i < (len(allLines[1]) + 1):
        if not batch_completed:
            if len(batched_input) < batch_size:
                data = getW2vRepresentation(allLines[1][i][0], loadedModel)
                data = data[:max_lenght]
                if len(data) > 0:
                    padded = np.lib.pad(data, ((max_lenght - len(data), 0), (0, 0)), 'constant', constant_values=(0))
                else:
                    padded = np.zeros(shape=[max_lenght, wordSize])
                batched_input.append(padded)
                if True:
                    label = np.zeros(shape=[max_lenght, 5])
                    index = int(allLines[1][i][1]) - 1
                    for j in range(0, max_lenght):
                        np.put(label[j], index, 1.0)
                    batched_label.append(label)
                    mask = np.zeros(max_lenght)
                    np.put(mask, np.arange(max_lenght - len(data), max_lenght), 1.0)
                    batched_mask.append(mask)
                    i += 1
            else:
                reshaped = np.reshape(batched_input, newshape=[batch_size, max_lenght, wordSize])
                labels = np.reshape(batched_label, newshape=[batch_size, max_lenght, 5])
                masks = np.reshape(batched_mask, newshape=[batch_size, max_lenght])
                toAppend = (reshaped, labels, masks)
                num_batch += 1
                if (len(allLines[1]) // batch_size) == num_batch:
                    batch_completed = True
                batched_input = []
                batched_label = []
                batched_mask = []

                yield toAppend
        else:
            break





