import json
import numpy as np
import random
import gensim

import tensorflow as tf

# ==============================================================================
################################# AUTHOR: ######################################
############################# DIEGO PERGOLINI ##################################
# ==============================================================================



def getBalancedReviewsAndOverall(numPositiveTraining,numNegativeTraining,numPositiveTesting,numNegativeTesting,fileName,max_lenght,randomized = False,seed=9):
    listT = []
    listTe = []
    in_file = open(fileName, "r")
    posT = 0
    negT = 0
    posTe = 0
    negTe = 0
    ranT = random.Random(seed)
    ranTe = random.Random(seed)
    while posT < numPositiveTraining or negT < numNegativeTraining or posTe < numPositiveTesting or negTe < numNegativeTesting:
            raw = in_file.readline()
            if raw == "":
                break
                print(posT,negT,posTe,negTe)
            jsonLine = json.loads(raw)

            review = jsonLine['reviewText']
            title = jsonLine['summary']
            num_words = len(gensim.utils.simple_preprocess(review))+len(gensim.utils.simple_preprocess(title))
            if randomized:
                    if random.Random(seed).randint(0, 1) == 1:
                        if jsonLine['overall'] < 3.0:
                            if negT < numNegativeTraining:
                                    current = ((title+review), jsonLine['overall'])
                                    listT.append(current)
                                    negT += 1
                            elif negTe < numNegativeTesting:
                                    current = ((title + review), jsonLine['overall'])
                                    listTe.append(current)
                                    negTe += 1
                        else:
                            if jsonLine['overall']> 3.0:
                                if posT < numPositiveTraining:
                                        current = ((title+review), jsonLine['overall'])
                                        listT.append(current)
                                        posT += 1
                                elif posTe < numPositiveTesting:
                                    current = ((title + review), jsonLine['overall'])
                                    listTe.append(current)
                                    posTe += 1
            else:
                    if jsonLine['overall'] < 3.0:
                        if negT < numNegativeTraining:
                            current = ((title + review), jsonLine['overall'])
                            listT.append(current)
                            negT += 1
                        elif negTe < numNegativeTesting:
                            current = ((title + review), jsonLine['overall'])
                            listTe.append(current)
                            negTe += 1
                    else:
                        if jsonLine['overall'] > 3.0:
                            if posT < numPositiveTraining:
                                current = ((title + review), jsonLine['overall'])
                                listT.append(current)
                                posT += 1
                            elif posTe < numPositiveTesting:
                                current = ((title + review), jsonLine['overall'])
                                listTe.append(current)
                                posTe += 1
                    print(posT,negT,posTe,negTe)
    ranT.shuffle(listT)
    ranTe.shuffle(listTe)
    in_file.close()
    return listT,listTe


def getCrossDomainReviewsAndOverall(numTraining,numTesting,fileTraining,fileTesting,max_lenght,randomized= False,seed=9):
    listT,_ = getBalancedReviewsAndOverall(numTraining,numTraining,0,0,fileTraining,max_lenght,randomized,seed)
    _,listTe = getBalancedReviewsAndOverall(0, 0,numTesting,numTesting,fileTesting,max_lenght,randomized,seed)
    return listT,listTe


def getW2vRepresentation(text,loadedModel):
    countDeleted = 0
    toReturn = []
    for element in gensim.utils.simple_preprocess(text):
        try:
            toReturn.append(loadedModel[element].tolist())
        except KeyError:
            countDeleted+=1
    return toReturn

def balancedGetDataset(allLines,modelName,FLAGS):
    wordSize = FLAGS.word_dimension
    batch_size = FLAGS.batch_size
    max_lenght = FLAGS.max_lenght
    loadedModel = gensim.models.KeyedVectors.load_word2vec_format(modelName,binary=True)

    random.shuffle(allLines[0])
    random.shuffle(allLines[1])

    batched_input = []
    batched_label = []
    batched_mask = []

    i = 0
    num_batch = 0
    batch_completed = False
    if len(allLines[0])>0:
        while i < (len(allLines[0])+1):
            if not batch_completed:
                if len(batched_input) < batch_size:
                    data = getW2vRepresentation(allLines[0][i][0], loadedModel)
                    data = data[:max_lenght]
                    if len(data)>0:
                        padded = np.lib.pad(data, ((max_lenght-len(data),0),(0,0)), 'constant', constant_values=(0))
                        delta = 1.0 / (len(data))
                    else:
                        padded = np.zeros(shape=[max_lenght, wordSize])
                        delta = 1.0
                    batched_input.append(padded)
                    if allLines[0][i][1] < 3:
                        label = np.zeros(shape=[max_lenght,2])
                        for j in range(0,max_lenght):
                            np.put(label[j],1,1.0)
                        batched_label.append(label)


                        mask = np.zeros(max_lenght)
                        weight = np.zeros(len(data))
                        for number in range(0, len(data)):
                            np.put(weight, number, delta * (number + 1))
                        np.put(mask, np.arange(max_lenght - len(data), max_lenght), weight)
                        batched_mask.append(mask)
                        i += 1
                    else:
                        label = np.zeros(shape=[max_lenght,2])
                        for j in range(0,max_lenght):
                            np.put(label[j],0,1.0)
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
                    labels = np.reshape(batched_label,newshape=[batch_size,max_lenght,2])
                    masks = np.reshape(batched_mask,newshape=[batch_size,max_lenght])
                    toAppend = (reshaped,labels,masks)

                    num_batch+=1
                    if (len(allLines[0])//batch_size) == num_batch:
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
    while i < (len(allLines[1])+1):
        if not batch_completed:
            if len(batched_input) < batch_size:
                data = getW2vRepresentation(allLines[1][i][0], loadedModel)
                data = data[:max_lenght]
                if len(data)>0:
                    padded = np.lib.pad(data, ((max_lenght-len(data),0),(0,0)), 'constant', constant_values=(0))
                else:
                    padded = np.zeros(shape=[max_lenght, wordSize])
                batched_input.append(padded)
                if allLines[1][i][1] < 3:
                    label = np.zeros(shape=[max_lenght, 2])
                    for j in range(0, max_lenght):
                        np.put(label[j], 1, 1.0)
                    batched_label.append(label)
                    mask = np.zeros(max_lenght)
                    np.put(mask, np.arange(max_lenght - len(data), max_lenght), 1.0)
                    batched_mask.append(mask)
                    i += 1

                else:
                    label = np.zeros(shape=[max_lenght, 2])
                    for j in range(0, max_lenght):
                        np.put(label[j], 0, 1.0)
                    batched_label.append(label)
                    mask = np.zeros(max_lenght)
                    np.put(mask, np.arange(max_lenght - len(data), max_lenght), 1.0)
                    batched_mask.append(mask)
                    i += 1
            else:
                reshaped = np.reshape(batched_input, newshape=[batch_size, max_lenght, wordSize])
                labels = np.reshape(batched_label, newshape=[batch_size, max_lenght, 2])
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
