import os
import re
import string
import sys
from collections import OrderedDict


def getDS(FLAGS, external=False):
    """
    Returns the row text as list. No preprocessing is performed

    """

    if external:
        dsName = FLAGS.vocab_source
        top_k = FLAGS.k if FLAGS.k > 0 else None
    else:
        dsName = FLAGS.dataset_name

    full_path = os.path.join(FLAGS.dataset_path, dsName)
    print("Reading dataset: {} ...".format(dsName)) if FLAGS.verbose else False

    if dsName.lower() == "4_Guardian_new".lower():
        return getGuardian(full_path, FLAGS.guardian_case, mask_digits=FLAGS.mask_digits)

    elif dsName.lower() == "4_Guardian_old".lower():
        return getGuardian(full_path, FLAGS.guardian_case, mask_digits=FLAGS.mask_digits)

    elif dsName.lower() == "BNC".lower():
        return getBNC(full_path, top_k)

    else:
        try:
            raise ValueError("Dataset not found", dsName)
        except ValueError:
            print("{} not found. The dataset's name is misspelled, or Dataset files dont exist.".format(dsName))
            raise


def getGuardian(full_path, case=1, VERBOSE=False, mask_digits=True):
    """
    Get the Guardian Dataset in specific
    """
    texts = []  # list of text samples
    topics = []  # list of label ids
    authors = []  # list of label ids

    # p_tags_nSpace = re.compile("(<.*?>)|(\s{2,})")
    p_nums = re.compile('\d')
    # p_underscore = re.compile('\w_|_\w')

    for topic in sorted(os.listdir(full_path)):
        path = os.path.join(full_path, topic)
        if os.path.isdir(path):
            for author in sorted(os.listdir(path)):
                subpath = os.path.join(path, author)
                for filename in sorted(os.listdir(subpath)):
                    fpath = os.path.join(subpath, filename)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read()
                    # new preprocessing
                    # 1- Remove html tags
                    # t = re.sub(p_tags_nSpace, ' ', t)
                    # 2- replace numbers with #
                    if mask_digits:
                        t = re.sub(p_nums, "#", t)
                    # 3- solve the underscore problem
                    # t = re.sub(p_underscore, " _ ", t)

                    texts.append(t)
                    f.close()
                    topics.append(topic)
                    authors.append(author)

    print('Found %s texts.' % len(texts)) if VERBOSE else print("", end='')
    assert (len(texts) == len(authors) == len(topics))

    if case == 1:
        train_topics = ['Politics']
        train_topics_1 = ['Society']
        valid_topics = ['UK']
        test_topics = ['World']

    elif case == 2: # actually 2
        train_topics = ['Politics']
        train_topics_1 = ['UK']
        valid_topics = ['Society']
        test_topics = ['World']

    elif case == 3: # actually 3
        train_topics = ['Politics']
        train_topics_1 = ['World']
        valid_topics = ['Society']
        test_topics = ['UK']

    elif case == 4:
        train_topics = ['Society']
        train_topics_1 = ['Politics']
        valid_topics = ['UK']
        test_topics = ['World']

    elif case == 5: # actually 5
        train_topics = ['Society']
        train_topics_1 = ['UK']
        valid_topics = ['Politics']
        test_topics = ['World']

    elif case == 6: # actually 6
        train_topics = ['Society']
        train_topics_1 = ['World']
        valid_topics = ['Politics']
        test_topics = ['UK']

    elif case == 7:
        train_topics = ['UK'];
        train_topics_1 = ['Politics']
        valid_topics = ['Society']
        test_topics = ['World']

    elif case == 8: # 8
        train_topics = ['UK']
        train_topics_1 = ['Society']
        valid_topics = ['Politics']
        test_topics = ['World']

    elif case == 9: # 9
        train_topics = ['UK'];
        train_topics_1 = ['World']
        valid_topics = ['Politics']
        test_topics = ['Society']

    elif case == 10:
        train_topics = ['World']
        train_topics_1 = ['Politics']
        valid_topics = ['Society']
        test_topics = ['UK']

    elif case == 11: # 11
        train_topics = ['World']
        train_topics_1 = ['Society']
        valid_topics = ['Politics']
        test_topics = ['UK']

    elif case == 12: # 12
        train_topics = ['World']
        train_topics_1 = ['UK']
        valid_topics = ['Politics']
        test_topics = ['Society']

    else:
        try:
            raise ValueError
        except ValueError:
            print("Wrong Case number: {} for the Guardian dataset".format(case))
            raise

    x_train_1, y_train_1, y_train_t1 = [], [], []
    x_train_2, y_train_2, y_train_t2 = [], [], []
    x_valid, y_valid, y_valid_t = [], [], []
    x_tests, y_tests, y_tests_t = [], [], []

    for i, (x_i, y_a, y_t) in enumerate(zip(texts, authors, topics)):
        if y_t in train_topics:
            x_train_1.append(x_i)
            y_train_1.append(y_a)
            y_train_t1.append(y_t)

        elif y_t in train_topics_1:
            x_train_2.append(x_i)
            y_train_2.append(y_a)
            y_train_t2.append(y_t)

        elif y_t in valid_topics:
            x_valid.append(x_i)
            y_valid.append(y_a)
            y_valid_t.append(y_t)

        elif y_t in test_topics:
            x_tests.append(x_i)
            y_tests.append(y_a)
            y_tests_t.append(y_t)

    # return texts, (topics, authors)
    return {'x_train1': x_train_1, 'y_train1': y_train_1, 'y_train_t1': y_train_t1,
            'x_train2': x_train_2, 'y_train2': y_train_2, 'y_train_t2': y_train_t2,
            'x_valid': x_valid, 'y_valid': y_valid, 'y_valid_t': y_valid_t,
            'x_tests': x_tests, 'y_tests': y_tests, 'y_tests_t': y_tests_t
            }

    # raise NotImplementedError


def getBNC(full_path, TOP_K=None):
    """
    Get the BNC dataset in specific
    The file all.num contains the list of all words (all) orderered by frequency (num) starting from the highest
    """
    # pdb.set_trace()
    if TOP_K == None:
        return []

    Vocabs = OrderedDict()
    with open(os.path.join(full_path, "all.num"), 'r') as wordListFile:
        # counter = 0
        ##skip the first line which is a header

        lines = wordListFile.readlines()
        TOP_K = len(lines) if TOP_K is None else TOP_K

        i = 1  # Ignore the first line
        while len(Vocabs) < TOP_K:
            try:
                line = lines[i].split()
                i += 1
            except:
                break

            try:
                # Vocabs.append(line[1]) ## we ignore the rest of the data
                token = line[1]
                if token.isdigit() or token in string.punctuation:
                    pass  # ignore it
                else:
                    Vocabs[token] = 0
                    # try:
                    #     Vocabs[token]
                    # except:
                    #     Vocabs[token] = 0

            except:
                print("End of lines")

    # double check the damn list for puncts and nums
    return [
        [" ".join(list(Vocabs.keys()))]
    ]  # we convert it to a list of documents, for consistency


