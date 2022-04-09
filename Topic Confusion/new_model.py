"""_summary_
A poilerplate for a new model

"""

# NLP
import numpy as np
from absl import app
from absl import flags
from absl import logging

# ML
from sklearn.metrics import balanced_accuracy_score

from utils import loadMyData

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

    flags.DEFINE_bool('small_dataset', False, 'if we use small sample of the dataset for authorship')
    flags.DEFINE_string('dataset_path', './myData', 'The path to my datasets folder')
    flags.DEFINE_bool('lower_case', False, 'Whether to keep sentence case or force lower case')
    flags.DEFINE_bool('remove_sw', False, 'Whether to remove stopwords or keep them')
    flags.DEFINE_integer('freq_threshold', 5, 'Word corpus frequency in order to keep it in text')

    flags.DEFINE_integer('vocab_size', 1000, 'number of words in the vocabulary set')

    flags.DEFINE_integer("k", 0, "The number of words to get either from the training set or the external one")
    flags.DEFINE_string("ngram_level", "word", "$n$-gram level ['word', 'char'].")
    flags.DEFINE_integer("min_n", 1, "Min value of n in (n-gram). Default:1")
    flags.DEFINE_integer("max_n", 1, "Max value of n in (n-gram). Default:1")
    flags.DEFINE_integer("min_freq", 1, "Minimum frequency for a word to have an embedding.")
    flags.DEFINE_integer("run_times", 1, "The number of times to repeat an experiment -classification part, Default:1")
    flags.DEFINE_bool('mask_digits', False, "Whether to mask digits with #")
    flags.DEFINE_enum('scenario', None, ['same', 'cross'], "Whether authorship is same-topic or cross-topic")
    flags.DEFINE_integer('epochs', 2, 'The number of epochs', lower_bound=2)

    flags.DEFINE_bool('verbose', False, 'Show output or supress it')

flags_defined = True
# endregion


def loss_type(y1, y2, random_authors):
    res1 = np.where(random_authors == y1)
    res2 = np.where(random_authors == y2)

    res1 = int(res1[0][0] / 6)
    res2 = int(res2[0][0] / 6)

    # print(res1, random_authors)
    # print(res2, random_authors)

    if res1 == res2:
        l = 'L0'
    elif (res1 == 0 and res2 == 1) or (res1 == 1 and res2 == 0):
        l = 'L1'
    elif (res1 == 2 and res2 == 3) or (res1 == 3 and res2 == 2):
        l = 'L1'
    else:
        l = 'SomethingWrong'

    # print(l)
    return l


def confusion_report(y_tests, preds_tests, random_authors):
    # print(random_authors[:3], random_authors[3:6], random_authors[6:9], random_authors[9:])
    # [(y1, y2, loss_type(y1, y2, random_authors)) for y1, y2 in zip(y_tests, preds_tests) if y1 != y2]

    losses = [(y1, y2, loss_type(y1, y2, random_authors)) for y1, y2 in zip(y_tests, preds_tests) if y1 != y2]
    ttl = len(preds_tests)
    # print('Total # of samples: {}'.format(ttl))

    # true
    cnt_ = sum(y_tests == preds_tests)
    # print("Correctly classified:\t\t {},\t Ratio: {:.2f}".format(cnt_, cnt_ / ttl))

    # false in the same cluster
    cnt_0 = [x for _, _, x in losses].count('L0')
    # print('L0 error, Same-cluster:\t\t {},\t Ratio: {:.2f}'.format(cnt_0, cnt_0 / ttl))

    # false in the Cross cluster
    cnt_1 = [x for _, _, x in losses].count('L1')
    # print('L1 error, Cross-cluster:\t {},\t Ratio: {:.2f}'.format(cnt_1, cnt_1 / ttl))

    return ttl, cnt_, cnt_0, cnt_1



def main(argv):
    
    # this is the results file
    
    header_row = 'Epoch\tname\tn\tft\tk\tnum_feat\ttraining_acc.\tvalid_acc.\ttest_acc.\tnum_samples' \
                '\tcorrectly_classified\t_\tL0\t_\tL1\t_ '

    print(header_row)
    header_row += '\r'
    
    # region read confs (These are the randomly selected authors. stored in confs.txt)
    with open('confs.txt', 'r') as confFile:
        randomConfs = confFile.readlines()
    # enregion

    ### We read the data first
    datasets = loadMyData(FLAGS)
    datasets_raw = [(datasets[0][0], datasets[0][1], datasets[0][2]),
                    (datasets[0][3], datasets[0][4], datasets[0][5]),
                    (datasets[1][0], datasets[1][1], datasets[1][2]),
                    (datasets[2][0], datasets[2][1], datasets[2][2])]

    """
    Not really epochs, but just iterations. Just reducing the different types of flags.
    going over each author in the confs.txt file
    """ 
    for epoch in range(FLAGS.epochs):
        # region read data
        idx = np.array([int(x) for x in randomConfs[epoch][1:-2].split('] [')[0].split()])

        group_1_x_raw, group_1_y, group_1_t = datasets_raw[idx[0]]
        group_2_x_raw, group_2_y, group_2_t = datasets_raw[idx[1]]
        group_3_x_raw, group_3_y, group_3_t = datasets_raw[idx[2]]
        group_4_x_raw, group_4_y, group_4_t = datasets_raw[idx[3]]

        # endregion

        # region split data into groups:
        random_authors = np.array([int(x) for x in randomConfs[epoch][1:-2].split('] [')[1].split()])
        train_x_raw, y_train, train_t, \
        valid_x_raw, y_valid, valid_t, \
        tests_x_raw, y_tests, tests_t = [], [], [], [], [], [], [], [], []

        for i in range(len(group_1_x_raw)):
            x_raw, y, t = group_1_x_raw[i], group_1_y[i], group_1_t[i]

            if y in random_authors[:6]:
                train_x_raw.append(x_raw)
                y_train.append(y)
                train_t.append(t)

            elif y in random_authors[6:]:
                tests_x_raw.append(x_raw)
                y_tests.append(y)
                tests_t.append(t)

        for i in range(len(group_2_x_raw)):
            x_raw, y, t = group_2_x_raw[i], group_2_y[i], group_2_t[i]

            if y in random_authors[:6]:
                tests_x_raw.append(x_raw)
                y_tests.append(y)
                tests_t.append(t)

            elif y in random_authors[6:]:
                train_x_raw.append(x_raw)
                y_train.append(y)
                train_t.append(t)

        for i in range(len(group_3_x_raw)):
            x_raw, y, t = group_3_x_raw[i], group_3_y[i], group_3_t[i]
            valid_x_raw.append(x_raw)
            y_valid.append(y)
            valid_t.append(t)

        for i in range(len(group_4_x_raw)):
            x_raw, y, t = group_4_x_raw[i], group_4_y[i], group_4_t[i]
            valid_x_raw.append(x_raw)
            y_valid.append(y)
            valid_t.append(t)
        # endregion
        
        # ADD YOUR NEW METHOD HERE
        
        ## region 0. process your data
        x_train = MY_PREPROCESS(train_x_raw)
        x_valid = MY_PREPROCESS(valid_x_raw)
        x_tests = MY_PREPROCESS(tests_x_raw)
        
        
        ## region 1. train/fine tune your model        
        YOUR_MODEL.train(x_train, y_train)
        YOUR_MODEL.validate(x_valid, y_valid)
        
        feat_name, best_HP1, best_HP2, best_HP3, best_valid_acc = '<FEAT NAME>', str(0), str(0), str(0), str(0.00)
        info2print = [epoch, feat_name, best_HP1, best_HP2, best_HP3, best_valid_acc]
        ## endregion
        
        ## region 2. test your model
        ### Warning: if your model is stochastic (retraining causes different results), then you need to do this multiple times
        trains = []
        tests = []
        ttls, cnt_s, cnt_0s, cnt_1s = 0, 0, 0, 0
        for run in range(10):
            
            # train your model on the training data
            preds_train = YOUR_MODEL.predict(x_train)
            preds_tests = YOUR_MODEL.predict(x_tests)

            # calculate balanced accuracy # because the data is imbalanced
            trains.append(balanced_accuracy_score(y_train, preds_train))
            tests.append(balanced_accuracy_score(y_tests, preds_tests))

            # calculate the confusion scores by providing the tests predictions and the test true_labels + the authors
            ttl, cnt_, cnt_0, cnt_1 = confusion_report(preds_tests, y_tests, random_authors)

            ttls += ttl
            cnt_s += cnt_
            cnt_0s += cnt_0
            cnt_1s += cnt_1
        
        s2write += '{:.2f}\t\t{:.2f}\t\t{:.2f}\t'.format(np.average(trains), info2print[5], np.average(tests))
            
        # print the outcome 
        s2write += '{:.2f}\t{:.2f}\t{:.2f}\t{}\t{:.2f}\t{}\t{:.2f}'.format(ttls / 10,
                                                                    cnt_s / 10, cnt_s / ttls,
                                                                    cnt_0s / 10, cnt_0s / ttls,
                                                                    cnt_1s / 10, cnt_1s / ttls)
                    
        print(s2write)

        # endregion
        
        
if __name__ == "__main__":
    app.run(main)