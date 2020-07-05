import math


def adaBoost(df, estimators):
    columns = df.columns()[0: estimators]
    df['initial_weights'] = 1 / len(df)
    index = 1
    adTree = {}
    for col in columns:
        decision_stump(df, col, index, adTree)


def decision_stump(df, attribute, index, adTree):
    classify1, classify2, correct_predictions, incorrect_predictions = classify(df, attribute)
    total_weight = 0
    for index, row in df.iterrows():
        if index in incorrect_predictions:
            total_weight += row[-1]
    significance = get_significance(total_weight)
    new_weights = get_new_weights(df, significance, correct_predictions, incorrect_predictions)
    df['new_weights' + str(index)] = normalize(new_weights)
    adTree[attribute][1] = classify1
    adTree[attribute][0] = classify2
    adTree[attribute]['significance'] = significance


def normalize(new_weights):
    normalized_weights = []
    max_value = max(new_weights)
    min_value = min(new_weights)
    for weight in new_weights:
        normalized_weights.append((weight - min_value) / (max_value - min_value))
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
    return 1 / 2 * math.log2((1 - total_weight) / total_weight)


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
            correct_predictions.add(index)
        elif row[attribute] == 1 and row['Language'] == 'nl':
            countNl_pos += 1
            incorrect_predictions.add(index)
        elif row[attribute] == 0 and row['Language'] == 'en':
            countEn_neg += 1
            incorrect_predictions.add(index)
        elif row[attribute] == 0 and row['Language'] == 'nl':
            countNl_neg += 1
            correct_predictions.add(index)
    if countEn_pos > countNl_pos:
        classify1 = 'en'
    else:
        classify1 = 'nl'
    if countEn_neg > countNl_neg:
        classify2 = 'en'
    else:
        classify2 = 'nl'
    return classify1, classify2, correct_predictions, incorrect_predictions
