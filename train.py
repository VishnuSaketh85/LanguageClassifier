import pandas as pd
import sys
import re
import math
import pickle

pd.set_option('display.max_columns', None)

delta = 0.00000000000001
max_depth = 7


def main():
    file_path = sys.argv[1]
    hypothesis_file = sys.argv[2]
    learning_type = sys.argv[3]
    df = create_df(file_path)
    if learning_type == 'dt':
        dTree = decisionTree(df, 0)
        dTree['LearningType'] = 'dt'
        f = open(hypothesis_file + '.pkl', 'wb')
        pickle.dump(dTree, f)
        f.close()
    if learning_type == 'ada':
        estimators = 8
        if estimators > 10:
            estimators = 10
        adaTree = adaBoost(df, estimators)
        adaTree['LearningType'] = 'ada'
        f = open(hypothesis_file + '.pkl', 'wb')
        pickle.dump(adaTree, f)
        f.close()


def decisionTree(df, depth, dTree=None):
    target_entropy = entropy(df['Language'])
    attribute_entropy = get_entropy_attributes(df)
    best_attribute = df.columns[get_best_attribute(attribute_entropy, target_entropy)]
    if dTree is None:
        dTree = {best_attribute: {}}
    if depth > max_depth:
        leaf_node = getLeafNode(df, best_attribute)
        dTree[best_attribute][1] = leaf_node
        if leaf_node == 'en':
            dTree[best_attribute][0] = 'nl'
        else:
            dTree[best_attribute][0]= 'en'
        # if df[best_attribute].value_counts()[1] > df[best_attribute].value_counts()[0]:
        #     dTree[best_attribute][1] = leaf_node
        # else:
        #     dTree[best_attribute][0] = leaf_node
        return dTree
    df1, df2 = splitData(df, best_attribute)
    flag, result = checkLeafNode(df1)
    if flag:
        dTree[best_attribute][1] = result
    else:
        dTree[best_attribute][1] = decisionTree(df1, depth + 1)
    flag, result = checkLeafNode(df2)
    if flag:
        dTree[best_attribute][0] = result
    else:
        dTree[best_attribute][0] = decisionTree(df2, depth + 1)
    return dTree


def splitData(df, best_attribute):
    df1 = df[df[best_attribute] == 1].reset_index(drop=True)
    df2 = df[df[best_attribute] == 0].reset_index(drop=True)
    return df1, df2


def getLeafNode(df, best_attribute):
    target = df['Language']
    count1 = 0
    count2 = 0
    for val in target:
        if val == 'en':
            count1 += 1
        else:
            count2 += 2
    if count1 > count2:
        return 'en'
    else:
        return 'nl'


def checkLeafNode(df):
    target = df['Language']
    count1 = 0
    count2 = 0
    for val in target:
        if val == 'en':
            count1 += 1
        else:
            count2 += 2
    if count1 == 0:
        return True, 'nl'
    elif count2 == 0:
        return True, 'en'
    else:
        return False, ' '


def entropy(target):
    countEn = 0
    countNl = 0
    for val in target:
        if val == 'en':
            countEn += 1
        else:
            countNl += 1
    probEn = countEn / len(target)
    probNl = countNl / len(target)
    return -probEn * math.log2(probEn) - probNl * math.log2(probNl)


def get_entropy_attributes(df):
    attribute_entropy = []
    columns = df.columns[0:len(df.columns) - 1]
    for col in columns:
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        for index, row in df.iterrows():
            if row[col] == 1 and row['Language'] == 'en':
                count1 += 1
            elif row[col] == 0 and row['Language'] == 'en':
                count2 += 1
            elif row[col] == 1 and row['Language'] == 'nl':
                count3 += 1
            elif row[col] == 0 and row['Language'] == 'nl':
                count4 += 1
        prob1 = count1 / (len(df[col][df[col] == 1]) + delta)
        prob2 = count2 / (len(df[col][df[col] == 0]) + delta)
        prob3 = count3 / (len(df[col][df[col] == 1]) + delta)
        prob4 = count4 / (len(df[col][df[col] == 0]) + delta)
        f1 = len(df[col][df[col] == 1]) / len(df)
        f2 = len(df[col][df[col] == 0]) / len(df)
        attribute_entropy.append(abs(
            -f1 * (-prob1 * math.log2(prob1 + delta) - prob3 * math.log2(prob3 + delta)) - f2 * (
                    -prob2 * math.log2(prob2 + delta)
                    - prob4 * math.log2(prob4 + delta))))
    return attribute_entropy


def get_best_attribute(attribute_entropy, target_entropy):
    best_attribute = 0
    max_infoGain = 0
    index = 0
    for att_ent in attribute_entropy:
        if target_entropy - att_ent > max_infoGain:
            best_attribute = index
            max_infoGain = target_entropy - att_ent
        index += 1

    return best_attribute


def create_df(file_path):
    dataList = []
    with open(file_path) as f:
        line = f.readline().strip().lower()
        while line:
            tempList = []
            target_sentence_split = line.split('|')
            target = target_sentence_split[0]
            sentence = target_sentence_split[1]
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
            tempList.append(target)
            dataList.append(tempList)
            line = f.readline().strip().lower()
    return pd.DataFrame(dataList, columns=['WordCount>5.1', 'LetterEFreq', 'LetterNFreq', 'The?', 'De?',
                                           'RepeatLetters', 'IJ?', 'VowelCount>14', 'Q?', 'Len1Words', 'Language'])


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


def adaBoost(df, estimators):
    unqEstimators = set()
    df['initial_weights'] = 1 / len(df)
    target_entropy = entropy_weight(df)
    attribute_entropy = get_entropy_attributes_weight(df)
    best_attribute = df.columns[get_best_attribute_weights(attribute_entropy, target_entropy, unqEstimators, df)]
    unqEstimators.add(best_attribute)
    index = 1
    adaTree = {}
    for i in range(0, estimators):
        best_attribute = decision_stump(df, best_attribute, index, adaTree, unqEstimators)
        unqEstimators.add(best_attribute)
    return adaTree


def decision_stump(df, attribute, index, adaTree, unqEstimators):
    classify1, classify2, correct_predictions, incorrect_predictions = classify(df, attribute)
    total_weight = 0
    for index, row in df.iterrows():
        if index in incorrect_predictions:
            total_weight += row[-1]
    significance = get_significance(total_weight)
    new_weights = get_new_weights(df, significance, correct_predictions, incorrect_predictions)
    df['new_weights' + str(index)] = normalize(new_weights)
    adaTree[attribute] = {}
    adaTree[attribute][1] = classify1
    adaTree[attribute][0] = classify2
    adaTree[attribute]['significance'] = significance
    next_best_attribute = get_nextBestAttr(df, incorrect_predictions, attribute, unqEstimators)
    return next_best_attribute


def get_best_attribute_weights(attribute_entropy, target_entropy, unqEstimators, df):
    best_attribute = 0
    max_infoGain = 0
    index = 0
    for att_ent in attribute_entropy:
        if target_entropy - att_ent > max_infoGain:
            if df.columns[index] not in unqEstimators:
                best_attribute = index
                max_infoGain = target_entropy - att_ent
        index += 1

    return best_attribute


def entropy_weight(df):
    weightEn = 0
    weightNl = 0
    totalWeight = 0
    for index, row in df.iterrows():
        totalWeight += row[-1]

        if row['Language'] == 'en':
            weightEn += row[-1]
        else:
            weightNl += row[-1]
    probEn = weightEn / totalWeight
    probNl = weightNl / totalWeight
    return -probEn * math.log2(probEn) - probNl * math.log2(probNl)


def get_entropy_attributes_weight(df):
    attribute_entropy = []
    columns = ['WordCount>5.1', 'LetterEFreq', 'LetterNFreq', 'The?', 'De?',
               'RepeatLetters', 'IJ?', 'VowelCount>14', 'Q?', 'Len1Words']
    for col in columns:
        count1 = 0
        count2 = 0
        count3 = 0
        count4 = 0
        totalWeight = 0
        for index, row in df.iterrows():
            totalWeight += row[-1]
            if row[col] == 1 and row['Language'] == 'en':
                count1 += row[-1]
            elif row[col] == 0 and row['Language'] == 'en':
                count2 += row[-1]
            elif row[col] == 1 and row['Language'] == 'nl':
                count3 += row[-1]
            elif row[col] == 0 and row['Language'] == 'nl':
                count4 += row[-1]
        prob1 = count1 / totalWeight
        prob2 = count2 / totalWeight
        prob3 = count3 / totalWeight
        prob4 = count4 / totalWeight
        f1 = len(df[col][df[col] == 1]) / len(df)
        f2 = len(df[col][df[col] == 0]) / len(df)
        attribute_entropy.append(abs(
            -f1 * (-prob1 * math.log2(prob1 + delta) - prob3 * math.log2(prob3 + delta)) - f2 * (
                    -prob2 * math.log2(prob2 + delta)
                    - prob4 * math.log2(prob4 + delta))))
    return attribute_entropy


def get_nextBestAttr(df, incorrect_predictions, attribute, unqEstimators):
    target_entropy = entropy_weight(df)
    attribute_entropy = get_entropy_attributes_weight(df)
    best_attribute = df.columns[get_best_attribute_weights(attribute_entropy, target_entropy, unqEstimators, df)]
    return best_attribute


def normalize(new_weights):
    normalized_weights = []
    Sum = sum(new_weights)
    for weight in new_weights:
        normalized_weights.append(weight / Sum)
    return normalized_weights


def get_new_weights(df, significance, correct_predictions, incorrect_predictions):
    new_weights = []
    for index, row in df.iterrows():
        if index in correct_predictions:
            new_weights.append(row[-1] * math.exp(-significance))
        elif index in incorrect_predictions:
            new_weights.append(row[-1] * math.exp(significance))
    return new_weights


def get_significance(total_weight):
    return (1 / 2) * math.log((1 - total_weight) / total_weight)


def classify(df, attribute):
    countEn_pos = 0
    countEn_neg = 0
    countNl_pos = 0
    countNl_neg = 0
    incorrect_predictions = set()
    correct_predictions = set()
    for index, row in df.iterrows():
        if row[attribute] == 1 and row['Language'] == 'en':
            countEn_pos += 1
        elif row[attribute] == 1 and row['Language'] == 'nl':
            countNl_pos += 1
        elif row[attribute] == 0 and row['Language'] == 'en':
            countEn_neg += 1
        elif row[attribute] == 0 and row['Language'] == 'nl':
            countNl_neg += 1
    if countEn_pos > countNl_pos:
        classify1 = 'en'
    else:
        classify1 = 'nl'
    if countEn_neg > countNl_neg:
        classify2 = 'en'
    else:
        classify2 = 'nl'
    for index, row in df.iterrows():
        if row[attribute] == 1 and row['Language'] == classify1:
            correct_predictions.add(index)
        elif row[attribute] == 1 and row['Language'] != classify1:
            incorrect_predictions.add(index)
        elif row[attribute] == 0 and row['Language'] == classify2:
            correct_predictions.add(index)
        elif row[attribute] == 0 and row['Language'] != classify2:
            incorrect_predictions.add(index)
    return classify1, classify2, correct_predictions, incorrect_predictions


main()
