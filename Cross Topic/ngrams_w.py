# -*- coding: utf-8 -*-

import os
import warnings

import numpy as np
from absl import app
from absl import flags
from absl import logging
from sklearn.exceptions import DataConversionWarning

from utils import loadMyData

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# NLP
import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer

import re
import string
import collections

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score

"""## Flags"""

logging.set_verbosity(logging.ERROR)

randSeed = 15
np.random.seed(randSeed)

# region Flags definition
FLAGS = flags.FLAGS

if "flags_defined" not in globals():
    flags.DEFINE_string('dataset_name', None, 'Dataset name')
    flags.DEFINE_integer("guardian_case", 1, "How to split the topics into: train/valid/test")
    flags.DEFINE_string("vocab_source", "same",
                        "Whether to use an external dataset for vocabulary features ['same', <external_dataset_name>]")

    flags.DEFINE_string('dataset_path', './myData', 'The path to my datasets folder')

    flags.DEFINE_bool('lower_case', False, 'Whether to keep sentence case or force lower case')
    flags.DEFINE_bool('remove_sw', False, 'Whether to remove stopwords or keep them')
    flags.DEFINE_integer('freq_threshold', 5, 'Word corpus frequency in order to keep it in text')


    flags.DEFINE_integer("k", 0, "The number of words to get either from the training set or the external one")
    flags.DEFINE_string("ngram_level", "word", "$n$-gram level ['word', 'char'].")
    flags.DEFINE_integer("min_n", 1, "Min value of n in (n-gram). Default:1")
    flags.DEFINE_integer("max_n", 1, "Max value of n in (n-gram). Default:1")
    flags.DEFINE_integer("min_freq", 1, "Minimum frequency for a word to have an embedding.")
    flags.DEFINE_integer("run_times", 1, "The number of times to repeat an experiment -classification part, Default:1")

    flags.DEFINE_bool('mask_digits', False, "Whether to mask digits with #")
    flags.DEFINE_enum('scenario', None, ['same', 'cross'], "Whether authorship is same-topic or cross-topic")


    flags.DEFINE_bool('verbose', False, 'Show output or supress it')

flags_defined = True
# endregion

def main(argv):
    print("case \t Scenario \t Ft \t n \t num_feats \t train \t valid \t test")
    with open('/' + str(os.path.basename(__file__).split('.')[0]) + '.txt', '+w') as resFile:
        resFile.write("case \t Scenario \t Ft \t n \t num_feats \t train \t valid \t test \n")

        for case in range(1, 13):
            FLAGS.guardian_case = case
            """## Reading Data"""
            
            # region read data
            FLAGS.mask_digits = True
            datasets = loadMyData(FLAGS)

            train_1_x_raw, train_1_y, train_1_t = datasets[0][0], datasets[0][1], datasets[0][2]
            train_2_x_raw, train_2_y, train_2_t = datasets[0][3], datasets[0][4], datasets[0][5]

            valid_x_raw, valid_y = datasets[1][0], datasets[1][1]

            tests_x_raw, tests_y = datasets[2][0], datasets[2][1]

            # print('Number of instances --> train1: {}, train2: {}, valid: {}, test: {}'.format(len(train_1_y), len(train_2_y),
            #                                                                                    len(valid_y), len(tests_y)))

            # endregion

            for scenario in ['same', 'cross']:
            # for scenario in ['same']:
                FLAGS.scenario = scenario
                """## Data Split and Feat Ex.

                ### Split data based on authorship scenario
                """
                if FLAGS.scenario == 'same':
                    X_raw = train_1_x_raw + train_2_x_raw + valid_x_raw + tests_x_raw
                    Y_raw = train_1_y + train_2_y + valid_y + tests_y

                    train_x, valid_x, y_train, y_valid = train_test_split(
                        X_raw, Y_raw, test_size=0.74, random_state=randSeed)

                    valid_x, tests_x, y_valid, y_tests = train_test_split(
                        valid_x, y_valid, test_size=0.65, random_state=randSeed)

                else:  # 'cross'
                    train_x = train_1_x_raw
                    valid_x = train_2_x_raw
                    tests_x = valid_x_raw + tests_x_raw

                    y_train = train_1_y
                    y_valid = train_2_y
                    y_tests = valid_y + tests_y

                for n in [1, 2, 3]:
                    FLAGS.min_n = n
                    FLAGS.max_n = n

                    
                    resFile.flush()

                    izer = RegexpTokenizer('\w+|[!"$%$&\'+(),-./:;<=>?@[\]^_`{|}~]')

                    if FLAGS.ngram_level == 'word':
                        CV = CountVectorizer(lowercase=False, tokenizer=izer.tokenize,
                                             ngram_range=(FLAGS.min_n, FLAGS.max_n))
                    elif FLAGS.ngram_level == 'char':
                        CV = CountVectorizer(lowercase=False, analyzer='char',
                                             ngram_range=(FLAGS.min_n, FLAGS.max_n))

                    all_feats_ngrams = CV.fit_transform(train_x).toarray()
                    vocab_ = CV.get_feature_names()
                    summed_feats = np.sum(all_feats_ngrams, axis=0)

                    for ft in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
                        FLAGS.freq_threshold = ft

                        # first stylo with POS

                        if ft != 0:
                            keep_ngrams = [vocab_[i] for i in range(len(vocab_)) if summed_feats[i] >= ft]
                            CV.vocabulary_ = dict([(keep_ngrams[i], i) for i in range(len(keep_ngrams))])

                        ## second, n-grams

                        # if FLAGS.embed_init == 'ngrams':
                        train_x_ngrams = CV.transform(train_x).toarray()
                        train_x_ngrams = train_x_ngrams.astype(np.float32)

                        valid_x_ngrams = CV.transform(valid_x).toarray()
                        valid_x_ngrams = valid_x_ngrams.astype(np.float32)

                        tests_x_ngrams = CV.transform(tests_x).toarray()
                        tests_x_ngrams = tests_x_ngrams.astype(np.float32)

                        SS1 = StandardScaler(copy=False)
                        train_x_ngrams = SS1.fit_transform(train_x_ngrams)
                        valid_x_ngrams = SS1.transform(valid_x_ngrams)
                        tests_x_ngrams = SS1.transform(tests_x_ngrams)


                        num_feat = np.shape(train_x_ngrams)[1]

                        """## Classification"""

                        loss = 'hinge'

                        # print(FLAGS.scenario.upper() + " Topic Authorship Attribution")

                        """### all """
                        train, valid, tests = [], [], []
                        for i in range(FLAGS.run_times):
                            clf_ngrams = None
                            clf_ngrams = SGDClassifier(loss=loss, max_iter=2000, tol=1e-3, class_weight='balanced',
                                                      n_jobs=-1)
                            clf_ngrams.fit(train_x_ngrams, y_train)

                            train_score = balanced_accuracy_score(y_train, clf_ngrams.predict(train_x_ngrams))
                            # print("Training acc:\t {:.2f}".format(train_score))
                            train.append(train_score)

                            valid_score = balanced_accuracy_score(y_valid, clf_ngrams.predict(valid_x_ngrams))
                            # print("Validation acc:\t {:.2f}".format(valid_score))
                            valid.append(valid_score)

                            tests_score = balanced_accuracy_score(y_tests, clf_ngrams.predict(tests_x_ngrams))
                            # print("Testing acc:\t {:.2f}".format(tests_score))
                            tests.append(tests_score)

                        resFile.write(
                            f'{FLAGS.guardian_case} \t {FLAGS.scenario} \t {ft} \t {n} \t {num_feat} \t\t'
                        )

                        resFile.write("{:.2f} ± {:.1f} \t {:.2f} ± {:.1f} \t {:.2f} ± {:.1f}\n".format(
                            100 * np.average(train), 100 * np.std(train),
                            100 * np.average(valid), 100 * np.std(valid),
                            100 * np.average(tests), 100 * np.std(tests)
                        ))

                        print(FLAGS.guardian_case, '\t', FLAGS.scenario, "\t", ft, '\t', n, '\t', num_feat, '\t\t',
                              end=' ')
                        print("{:.2f} ± {:.1f} \t {:.2f} ± {:.1f} \t {:.2f} ± {:.1f}".format(
                            100 * np.average(train), 100 * np.std(train),
                            100 * np.average(valid), 100 * np.std(valid),
                            100 * np.average(tests), 100 * np.std(tests)
                        ))

    return 0


if __name__ == "__main__":
    app.run(main)
