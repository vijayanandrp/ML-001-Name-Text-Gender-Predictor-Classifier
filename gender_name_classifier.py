#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import os
import random
from zipfile import ZipFile
from nltk import NaiveBayesClassifier, classify


__author__ = "Vijay Anand P"
__website__ = "www.informationcorners.com"
__credits__ = ["Stephen Holiday"]
__license__ = "GPL"
__version__ = "1.0.1"
__status__ = "Development"
__tags__ = ["gender classifier", "gender predictor",
            "male female names data_set", "machine learning"]

"""
Useful link = http://www.nltk.org/book/ch06.html

0. Requirement analysis
1. Data set preparation
2. Features extraction
3. Applying Machine learning algorithm
4. Evaluation of the model used.
"""

'''
0. Requirement analysis

Requirement of this project is to develop a tool that will predict or classify 
a given Name as a 'Male' or 'Female' gender by using a self learning algorithm so called 
'Machine Learning Algorithm'.

'''

'''
1. Data set preparation

As the (names, gender, frequency count) are already loaded into a "names.zip" file.
We will do a unzipping process and read all the text files, then create a two collection
'Male' and 'Female' respectively.
'''


def load_names(zip_file='data_set/names.zip'):
    """
     Names  = { Unique_Name[ Male_frequency_count, Female_frequency_count]}

    :param zip_file: names.zip data set
    :return: dict() containing names and frequency count.

    sample output-
    'ELMINA': [0, 1385], 'ALEARA': [0, 17], 'BEMNET': [31, 42], 'YISSELL': [0, 30],
    'IZAIYAH': [271, 0], 'BREELLA': [0, 66], 'VANDON': [79, 0], 'BRIAUNA': [0, 2585],
    'NAIDELIN': [0, 430], 'AMIS': [10, 0], 'NINNA': [0, 69], 'ILETTA': [0, 10],
    'KHRYSTEN': [0, 58], 'LORINA': [0, 3482], 'RINALDO': [1749, 0], 'DALAJAH': [0, 12],
    'DEVREN': [44, 0], 'JIAHAO': [12, 0], 'KAITLIN': [150, 112330], 'DEVRON': [1680, 0],
    'OWYN': [720, 15], 'ADJI': [0, 10], 'CHURCH': [10, 0], 'MISSI': [0, 444], 'KUSHANA': [0, 46],
    'LAYTEN': [334, 22], 'JEANMARC': [317, 0], 'MONTERRIUS': [152, 0], 'SAMHITA': [0, 552],
    'NATACHA': [0, 2709], 'KHALISAH': [0, 6], 'HARLEQUINN': [0, 10]
    """
    if not os.path.isfile(zip_file):
        print('names.zip is missing.')
        exit(-1)

    names = dict()
    gender_map = {'M': 0, 'F': 1}

    unzip = ZipFile(zip_file, 'r')
    files = unzip.namelist()

    for file in files:
        file = unzip.open(file, 'r').read().decode('utf-8')
        rows = [row.strip().split(',') for row in file.split('\n') if len(row) > 1]
        for row in rows:
            if not len(row) == 3:
                continue
            name = row[0].upper()
            gender = gender_map[row[1].upper()]
            count = int(row[2])
            # adding frequency in names dict based on gender
            if name not in names:
                names[name] = [0, 0]
            names[name][gender] += count
    return names


def split_names(names: dict()):
    """
    Converting into tuple (name, male_freq_count, female_freq_count)

    :param names: dict() containing names and frequency count
    :return: names tuple (male_names, female_names)
    """
    if not names:
        print('names dict is none.')
        exit(-1)

    male_names = list()
    female_names = list()

    for name in names.keys():
        counts = names[name]
        # converting into tuple (name, male_freq_count, female_freq_count)
        male_counts, female_counts = counts[0], counts[1]
        data = (name, male_counts, female_counts)

        if male_counts == female_counts:
            continue

        if male_counts > female_counts:
            male_names.append(data)
        else:
            female_names.append(data)

    names = (male_names, female_names)

    total_males_names = len(male_names)
    total_females_names = len(female_names)
    total_names = total_females_names + total_males_names
    print('Data set Overview.\n Total names - {} \n Total males names - '
          '{} \n Total female names - {}'.format(total_names, total_males_names,
                                                 total_females_names))
    return names


'''
2. Features extraction

Now we will read names one by one  and extract features from male and female names.

'''


def extract_feature(name: str):
    """
    Feature/attributes/input/predictors extraction from given name string.
    :param name: string
    :return: dict of feature values
    """
    name = name.upper()
    feature = dict()
    # additional feature extraction
    # feature["first_1"] = name[0]
    # for letter in 'abcdefghijklmnopqrstuvwxyz'.upper():
    #     feature["count({})".format(letter)] = name.count(letter)
    #     feature["has({})".format(letter)] = (letter in name)

    feature.update({
        'last_1': name[-1],
        'last_2': name[-2:],
        'last_3': name[-3:],
        'last_is_vowel': (name[-1] in 'AEIOUY')
    })
    return feature


def get_probability_distribution(name_tuple):
    """
    Applying probability distribution as One name have two outcomes.

    male_probability = total_male_count / (total_male_count + total_female_count)

    No system can hold the value 1.0 in probability so making it to 0.99.

    :param name_tuple: name tuple contains male / female frequency count
    :return: male, female probability
    """
    male_counts = name_tuple[1]
    female_counts = name_tuple[2]
    male_prob = (male_counts * 1.0) / sum([male_counts, female_counts])
    if male_prob == 1.0:
        male_prob = 0.99
    elif male_prob == 0.00:
        male_prob = 0.01
    female_prob = 1.0 - male_prob
    return male_prob, female_prob


def prepare_data_set():
    """
    Preparing feature matrix (X) and response vector (y) - Supervised Learning model.
    :param names: tuple contains males names and female names
    :return:
    """
    feature_set = list()
    male_names, female_names = split_names(load_names())
    names = {'M': male_names, 'F': female_names}

    for gender in names.keys():
        for name in names[gender]:
            features = extract_feature(name[0])
            male_prob, female_prob = get_probability_distribution(name)
            features['m_prob'] = male_prob
            features['f_prob'] = female_prob
            feature_set.append((features, gender))
    random.shuffle(feature_set)
    return feature_set


def validate_data_set(feature_set: list):
    """
    :param feature_set: validation purpose
    :return: None
    """
    data_list = []
    for feature_value, gender in feature_set:
        data_list.append({**feature_value, **{'gender': gender}})

    import pandas as pd
    df = pd.DataFrame(data_list)
    print('Feature matrix shape - ', df.shape)
    # print(df.head(5))
    # print(df.groupby(['gender']).count())

'''
3. Applying Machine learning algorithm 

Now its time to load our feature matrix (X) and response vector (y) to ML algorithm model.
I am choosing naive bayes because it is very good for text prediction and classification.
Naive bayes works better on supervised learning models. Later me may try other algorithms also.

'''


def train_and_test(train_percent=0.80):
    """
    splitting the data set and finding the accuracy.

    :param train_percent: split ratio
    :return:
    """
    feature_set = prepare_data_set()
    validate_data_set(feature_set)
    random.shuffle(feature_set)
    total = len(feature_set)
    cut_point = int(total * train_percent)
    # splitting data set into train and test
    train_set = feature_set[:cut_point]
    test_set = feature_set[cut_point:]

    # fitting feature matrix to the model
    classifier = NaiveBayesClassifier.train(train_set)

    print('Accuracy- ', classify.accuracy(classifier, test_set))
    print('Most informative features')
    informative_features = classifier.most_informative_features(n=5)
    for feature in informative_features:
        print("\t {} = {} ".format(*feature))
    return classifier


"""
4. Evaluation of the model used.

Testing with run time values.
"""


def model_evaluation(classifier):
    """
    Manual testing of the model.
    :param classifier: object to test run time names
    :return:
    """
    print('\n <<<  Testing Module   >>> \n Enter "q" or "quit" to end testing module')
    while 1:
        test_name = input('\n Enter name to classify: ')
        if test_name.lower() == 'q' or test_name.lower() == 'quit':
            print('End')
            exit(1)
        if not test_name:
            continue
        print('{} is classified as {}'.format(test_name,
                                              classifier.classify(extract_feature(test_name))))


model_evaluation(train_and_test())