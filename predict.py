import pandas as pd
import sys
import re
import math
import pickle

attribute_map = {'WordCount>5.1': 0, 'LetterEFreq': 1, 'LetterNFreq': 2, 'The?': 3, 'De?': 4,
                 'RepeatLetters': 5, 'IJ?': 6, 'VowelCount>14': 7, 'Q?': 8, 'Len1Words': 9}


def main():
    model_path = sys.argv[1]
    file_path = sys.argv[2]
    df = create_df(file_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        print(model)
        model_type = model['LearningType']
        if model_type == 'dt':
            for index, row in df.iterrows():
                predictDTree(model, row)
        elif model_type == 'ada':
            for index, row in df.iterrows():
                predictAdaTree(model, row)


def predictDTree(model, row):
    if row[attribute_map[list(model.keys())[0]]] == 1:
        if model[list(model.keys())[0]][1] == 'en' or model[list(model.keys())[0]][1] == 'nl':
            print(model[list(model.keys())[0]][1])
            return
        predictDTree(model[list(model.keys())[0]][1], row)
    else:
        if model[list(model.keys())[0]][0] == 'en' or model[list(model.keys())[0]][0] == 'nl':
            print(model[list(model.keys())[0]][0])
            return
        predictDTree(model[list(model.keys())[0]][0], row)


def predictAdaTree(model, row):
    predict_value = 0
    for key in model.keys():
        if key == 'LearningType':
            continue
        if model[key][row[key]] == 'en':
            predict_value += model[key]['significance']
        else:
            predict_value -= model[key]['significance']
    if predict_value > 0:
        print('en')
    else:
        print('nl')


def create_df(file_path):
    dataList = []
    with open(file_path) as f:
        line = f.readline().strip().lower()
        while line:
            tempList = []
            sentence = line
            tempList.append(avgWordCount(sentence))
            tempList.append(checkLetterEFreq(sentence))
            tempList.append(checkLetterNFreq(sentence))
            tempList.append(checkThe(sentence))
            tempList.append(checkDe(sentence))
            tempList.append(checkTwoLetter(sentence))
            tempList.append(checkIJ(sentence))
            tempList.append(countVowel(sentence))
            tempList.append(checkLetterQ(sentence))
            tempList.append(checkWordLength1(sentence))
            dataList.append(tempList)
            line = f.readline().strip().lower()
    return pd.DataFrame(dataList, columns=['WordCount>5.1', 'LetterEFreq', 'LetterNFreq', 'The?', 'De?',
                                           'RepeatLetters', 'IJ?', 'VowelCount>14', 'Q?', 'Len1Words'])


def avgWordCount(sentence):
    words = sentence.split()
    count = 0
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        count += len(word)
    avgCount = count / 15
    if avgCount > 5.1:
        return 1
    else:
        return 0


def checkThe(sentence):
    words = sentence.split()
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        if word == 'the':
            return 1
    return 0


def checkDe(sentence):
    words = sentence.split()
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        if word == 'de':
            return 1
    return 0


def checkTwoLetter(sentence):
    words = sentence.split()
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        for i in range(0, len(word) - 1):
            if word[i] == word[i + 1]:
                return 1
    return 0


def checkLetterNFreq(sentence):
    words = sentence.split()
    totalCount = 0
    countN = 0
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        totalCount += len(word)
        for letter in word:
            if letter == 'n':
                countN += 1
    if countN / totalCount > 0.085:
        return 1
    else:
        return 0


def checkLetterEFreq(sentence):
    words = sentence.split()
    totalCount = 0
    countE = 0
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        totalCount += len(word)
        for letter in word:
            if letter == 'e':
                countE += 1
    if countE / totalCount > 0.15:
        return 1
    else:
        return 0


def checkIJ(sentence):
    words = sentence.split()
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        for i in range(0, len(word) - 1):
            if word[i] == 'i' and word[i + 1] == 'j':
                return 1
    return 0


def countVowel(sentence):
    words = sentence.split()
    vowelList = ['a', 'e', 'i', 'o', 'u']
    count = 0
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        for i in range(0, len(word) - 1):
            if word[i] in vowelList:
                count += 1
    if count > 14:
        return 1
    else:
        return 0


def checkLetterQ(sentence):
    words = sentence.split()
    for word in words:
        word = re.sub(r'[^\w\s]', '', word)
        for letter in word:
            if letter == 'q':
                return 1
    return 0


def checkWordLength1(sentence):
    words = sentence.split()
    for word in words:
        if len(word) == 1:
            return 1
    return 0


main()
