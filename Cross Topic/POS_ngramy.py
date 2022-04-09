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

"""## A class for parsed docs"""


class Parsed(object):
    def __init__(self, doc='', updateImmediately=False, name='Parsed'):
        self.raw_text = doc
        self.charFeats = CharFeats()
        self.wordFeats = WordFeats()
        self.syntFeats = SyntacticFeats()
        self.posFeats = POSFeats()

        if updateImmediately:
            self.updateAll()

    def updateAll(self):
        # self.updateChar()
        # self.updateWord()
        # self.updateSynt()
        self.updatePOS()

    def updateChar(self):
        self.charFeats.update(self.raw_text)

    def updateWord(self):
        self.wordFeats.update(self.raw_text)

    def updateSynt(self):
        self.syntFeats.update(self.raw_text)

    def updatePOS(self):
        self.posFeats.update(self.raw_text)

    def vectorizeAll(self):
        # print(len(self.charFeats.vectorize()), len(self.wordFeats.vectorize()), len(self.syntFeats.vectorize()))
        # cc = self.posFeats.vectorize(CV)
        return self.charFeats.vectorize() + self.wordFeats.vectorize() + self.syntFeats.vectorize()


class CharFeats():
    def __init__(self, name='Char-Level'):
        self.N_charCount = 0
        self.digits2N_Ratio = 0
        self.letters2N_Ratio = 0
        self.upperCase2N_Ratio = 0
        self.space2N_Ratio = 0
        self.tabs2N_Ratio = 0
        self.alphaCounts = {}  # 26 feats
        self.specialCounts = {}  # 20 feats

    def update(self, doc):
        self.N_charCount = len(doc)
        self.digits2N_Ratio = len(re.findall(re.compile('\d'), doc)) / self.N_charCount
        self.letters2N_Ratio = len(re.findall(re.compile('[a-zA-Z]'), doc)) / self.N_charCount
        self.upperCase2N_Ratio = len(re.findall(re.compile('[A-Z]'), doc)) / self.N_charCount
        self.space2N_Ratio = len(re.findall(re.compile(' '), doc)) / self.N_charCount
        self.tabs2N_Ratio = len(re.findall(re.compile('\t'), doc)) / self.N_charCount
        self.alphaCounts = dict([(a, len(re.findall(re.compile('[' + a + A + ']'), doc))) for a, A in
                                 zip(string.ascii_lowercase, string.ascii_uppercase)])  # 26 feats ie. case insensitive
        self.specialCounts = dict([(sym, len(re.findall(re.compile('[\\' + sym + ']'), doc))) for sym in
                                   "#<>%|{}[]/@~+-*=$&^()_`"])  # 24 feats

    def vectorize(self):
        return [self.N_charCount] + \
               [self.digits2N_Ratio] + \
               [self.letters2N_Ratio] + \
               [self.upperCase2N_Ratio] + \
               [self.space2N_Ratio] + \
               [self.tabs2N_Ratio] + \
               [self.alphaCounts[k] for k in sorted(self.alphaCounts.keys())] + \
               [self.specialCounts[k] for k in sorted(self.specialCounts.keys())]


class WordFeats(object):
    def __init__(self, name='Word-Level'):
        self.T_wordCount = 0
        self.avgSentLenInChar = 0
        self.avgWordLenInChar = 0
        self.charInWord2N_Ratio = 0
        self.shortWords2T_Ratio = 0
        self.wordsLength2T_Ratio = {}  # 20 feats
        self.types2T_Ratio = 0
        self.vocabRichness = 0  # Yule's K measure : https://gist.github.com/magnusnissel/d9521cb78b9ae0b2c7d6
        self.hapexLegomena = 0
        self.hapexDislegomena = 0

    def update(self, doc):
        self.T_wordCount = len(re.findall(re.compile('\w+'), doc))
        self.avgSentLenInChar = np.average([len(sent) + 1 for sent in re.split('\.\s', doc)])

        wordsInDoc = re.findall('\w+', doc)

        self.avgWordLenInChar = np.average([len(token) for token in wordsInDoc])
        self.charInWord2N_Ratio = np.sum([len(token) for token in wordsInDoc]) / len(doc)
        self.shortWords2T_Ratio = len([token for token in wordsInDoc if len(token) < 4]) / self.T_wordCount

        theRatio = 1 / self.T_wordCount
        self.wordsLength2T_Ratio = dict([(i, 0) for i in range(21)])
        for token in wordsInDoc:
            try:
                self.wordsLength2T_Ratio[len(token)] += theRatio  # 20 feats
            except:  # in case a word is longer
                self.wordsLength2T_Ratio[0] += theRatio

        self.types2T_Ratio = len(set(wordsInDoc)) / self.T_wordCount
        self.vocabRichness = self.Yolks_k(
            doc)  # Yule's K measure : https://gist.github.com/magnusnissel/d9521cb78b9ae0b2c7d6
        self.hapexLegomena = len([token for token in list(set(wordsInDoc)) if wordsInDoc.count(token) == 1])
        self.hapexDislegomena = len([token for token in list(set(wordsInDoc)) if wordsInDoc.count(token) == 2])

    def vectorize(self):
        return [self.T_wordCount] + \
               [self.avgSentLenInChar] + \
               [self.avgWordLenInChar] + \
               [self.charInWord2N_Ratio] + \
               [self.shortWords2T_Ratio] + \
               [self.wordsLength2T_Ratio[k] for k in sorted(self.wordsLength2T_Ratio.keys())] + \
               [self.types2T_Ratio] + \
               [self.vocabRichness] + \
               [self.hapexLegomena] + \
               [self.hapexDislegomena]

    def Yolks_k(self, doc):
        tokens = re.split(r"[^0-9A-Za-z\-'_]+", doc)
        token_counter = collections.Counter(tok.lower() for tok in tokens)
        m1 = sum(token_counter.values())
        m2 = sum([freq ** 2 for freq in token_counter.values()])
        i = (m1 * m1) / (m2 - m1)
        k = 1 / i * 10000
        # return (k, i)
        return k


class SyntacticFeats(object):
    def __init__(self, name='Syntactic'):
        self.punctuCounts = dict([(c, 0) for c in ",.?!:;'\""])  # 8 feats , . ? ! : ; ' "
        self.functCounts = {}  # 303 feats, found only 277

        # https://semanticsimilarity.files.wordpress.com/2013/08/jim-oshea-fwlist-277.pdf
        self.fnWords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against',
                        'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always',
                        'am', 'among', 'amongst', 'amoungst', 'an', 'and', 'another', 'any', 'anyhow',
                        'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'be',
                        'became', 'because', 'been', 'before', 'beforehand', 'behind', 'being', 'below',
                        'beside', 'besides', 'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot',
                        'could', 'dare', 'despite', 'did', 'do', 'does', 'done', 'down', 'during', 'each',
                        'eg', 'either', 'else', 'elsewhere', 'enough', 'etc', 'even', 'ever', 'every',
                        'everyone', 'everything', 'everywhere', 'except', 'few', 'first', 'for', 'former',
                        'formerly', 'from', 'further', 'furthermore', 'had', 'has', 'have', 'he', 'hence',
                        'her', 'here', 'hereabouts', 'hereafter', 'hereby', 'herein', 'hereinafter', 'heretofore',
                        'hereunder', 'hereupon', 'herewith', 'hers', 'herself', 'him', 'himself', 'his', 'how',
                        'however', 'i', 'ie', 'if', 'in', 'indeed', 'inside', 'instead', 'into', 'is', 'it',
                        'its', 'itself', 'last', 'latter', 'latterly', 'least', 'less', 'lot', 'lots', 'many',
                        'may', 'me', 'meanwhile', 'might', 'mine', 'more', 'moreover', 'most', 'mostly', 'much',
                        'must', 'my', 'myself', 'namely', 'near', 'need', 'neither', 'never', 'nevertheless',
                        'next', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere',
                        'of', 'off', 'often', 'oftentimes', 'on', 'once', 'one', 'only', 'onto', 'or', 'other',
                        'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over',
                        'per', 'perhaps', 'rather', 're', 'same', 'second', 'several', 'shall', 'she', 'should',
                        'since', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes',
                        'somewhat', 'somewhere', 'still', 'such', 'than', 'that', 'the', 'their', 'theirs',
                        'them', 'themselves', 'then', 'thence', 'there', 'thereabouts', 'thereafter', 'thereby',
                        'therefore', 'therein', 'thereof', 'thereon', 'thereupon', 'these', 'they', 'third',
                        'this', 'those', 'though', 'through', 'throughout', 'thru', 'thus', 'to', 'together',
                        'too', 'top', 'toward', 'towards', 'under', 'until', 'up', 'upon', 'us', 'used', 'very',
                        'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever',
                        'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether',
                        'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'whyever',
                        'will', 'with', 'within', 'without', 'would', 'yes', 'yet', 'you', 'your', 'yours', 'yourself',
                        'yourselves']

    def update(self, doc):
        self.punctuCounts = dict([(p, len(re.findall('\\' + p, doc))) for p in self.punctuCounts.keys()])
        doc_lower = re.findall('\w+', doc.lower())
        self.functCounts = dict([(k, doc_lower.count(k)) for k in self.fnWords])

    def vectorize(self):
        return [self.punctuCounts[k] for k in sorted(self.punctuCounts.keys())] + \
               [self.functCounts[k] for k in sorted(self.functCounts.keys())]


class POSFeats(object):
    def __init__(self, name='POS'):
        self.tags = []
        self.pos_ngrams = []

    def update(self, doc):
        sentences = nltk.sent_tokenize(doc)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        # self.tagged_doc = [[tag for (word, tag) in nltk.pos_tag(sent)] for sent in sentences]
        self.tags = [tag for sent in sentences for (word, tag) in nltk.pos_tag(sent)]

    # def vectorize(self):
    #     # self.pos_ngrams = CV.transform(self.tags).toarray()
    #     return self.pos_ngrams


def extract_all_feats(docs):  # except POS ngrams
    processed_docs = []
    pos_tags = []
    for doc in docs:
        p_doc = Parsed(doc)
        p_doc.updateAll()
        processed_docs.append(p_doc)
        pos_tags.append(p_doc.posFeats.tags)

    return processed_docs


def get_all_feats(processedDocs, CV=None):
    vectors = []
    for p_doc in processedDocs:
        # stylo_vecs = p_doc.vectorizeAll()
        stylo_vecs = []
        ngrams_vecs = CV.transform([p_doc.posFeats.tags]).toarray()

        ngrams_vecs = ngrams_vecs.astype(np.int).tolist()[0]
        stylo_vecs.extend(ngrams_vecs)

        vectors.append(stylo_vecs)

    return vectors

def main(argv):
    print("case \t Scenario \t Ft \t n \t num_feats \t train \t valid \t test")
    with open('/' + str(os.path.basename(__file__).split('.')[0]) + '.txt', '+w') as resFile:
        resFile.write("case \t Scenario \t Ft \t n \t num_feats \t train \t valid \t test \n")

        for case in range(1, 13):
            FLAGS.guardian_case = case
            """## Reading Data"""
            # first time, read with no preprocessing for stylo metric features
            # region read data
            datasets = loadMyData(FLAGS)

            train_1_x_raw, train_1_y, train_1_t = datasets[0][0], datasets[0][1], datasets[0][2]
            train_2_x_raw, train_2_y, train_2_t = datasets[0][3], datasets[0][4], datasets[0][5]

            valid_x_raw, valid_y = datasets[1][0], datasets[1][1]

            tests_x_raw, tests_y = datasets[2][0], datasets[2][1]

            # print('Number of instances --> train1: {}, train2: {}, valid: {}, test: {}'.format(len(train_1_y), len(train_2_y),
            #                                                                                    len(valid_y), len(tests_y)))

            # endregion

            # for scenario in ['same', 'cross']:
            for scenario in ['cross']:
                FLAGS.scenario = scenario
                """## Data Split and Feat Ex.

                ### Split data based on authorship scenario
                """
                if FLAGS.scenario == 'same':
                    X_raw = train_1_x_raw + train_2_x_raw + valid_x_raw + tests_x_raw
                    Y_raw = train_1_y + train_2_y + valid_y + tests_y

                    train_x, valid_x, y_train, y_valid = train_test_split(
                        X_raw, Y_raw, test_size=0.70, random_state=randSeed)

                    valid_x, tests_x, y_valid, y_tests = train_test_split(
                        valid_x, y_valid, test_size=0.56, random_state=randSeed)

                else:  # 'cross'
                    train_x = train_1_x_raw
                    valid_x = train_2_x_raw
                    tests_x = valid_x_raw + tests_x_raw

                    y_train = train_1_y
                    y_valid = train_2_y
                    y_tests = valid_y + tests_y

                train_x_stylo_raw = extract_all_feats(train_x)
                valid_x_stylo_raw = extract_all_feats(valid_x)
                tests_x_stylo_raw = extract_all_feats(tests_x)

                #start over for ngrams , but mask digits
                FLAGS.mask_digits = True
                datasets = loadMyData(FLAGS)

                train_1_x_raw, train_1_y, train_1_t = datasets[0][0], datasets[0][1], datasets[0][2]
                train_2_x_raw, train_2_y, train_2_t = datasets[0][3], datasets[0][4], datasets[0][5]

                valid_x_raw, valid_y = datasets[1][0], datasets[1][1]

                tests_x_raw, tests_y = datasets[2][0], datasets[2][1]
                # endregion

                if FLAGS.scenario == 'same':
                    X_raw = train_1_x_raw + train_2_x_raw + valid_x_raw + tests_x_raw
                    Y_raw = train_1_y + train_2_y + valid_y + tests_y

                    train_x, valid_x, y_train, y_valid = train_test_split(
                        X_raw, Y_raw, test_size=0.70, random_state=randSeed)

                    valid_x, tests_x, y_valid, y_tests = train_test_split(
                        valid_x, y_valid, test_size=0.56, random_state=randSeed)

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

                    pos_ft, pos_n = n, n

                    resFile.flush()

                    ## elif FLAGS.embed_init =='stylo':
                    posCV = CountVectorizer(lowercase=False, analyzer='word', tokenizer=lambda x: x,
                                            # token_pattern =re.compile("\S+"),
                                            ngram_range=(pos_n, pos_n))


                    izer = RegexpTokenizer('\w+|[!"$%$&\'+(),-./:;<=>?@[\]^_`{|}~]')

                    if FLAGS.ngram_level == 'word':
                        CV = CountVectorizer(lowercase=False, tokenizer=izer.tokenize,
                                             ngram_range=(FLAGS.min_n, FLAGS.max_n))
                    elif FLAGS.ngram_level == 'char':
                        CV = CountVectorizer(lowercase=False, analyzer='char',
                                             ngram_range=(FLAGS.min_n, FLAGS.max_n))

                    # some = counter['Ken']
                    all_feats = posCV.fit_transform([p_doc.posFeats.tags for p_doc in train_x_stylo_raw]).toarray()
                    vocab_pos = posCV.get_feature_names()
                    summed_feats_pos = np.sum(all_feats, axis=0)

                    all_feats_ngrams = CV.fit_transform(train_x).toarray()
                    vocab_ = CV.get_feature_names()
                    summed_feats = np.sum(all_feats_ngrams, axis=0)

                    for ft in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
                        FLAGS.freq_threshold = ft

                        # first stylo with POS

                        if ft != 0:
                            keep = [vocab_pos[i] for i in range(len(vocab_pos)) if summed_feats_pos[i] >= pos_ft]
                            posCV.vocabulary_ = dict([(keep[i], i) for i in range(len(keep))])

                            keep_ngrams = [vocab_[i] for i in range(len(vocab_)) if summed_feats[i] >= ft]
                            CV.vocabulary_ = dict([(keep_ngrams[i], i) for i in range(len(keep_ngrams))])


                        train_x_stylo = get_all_feats(train_x_stylo_raw, posCV)
                        # train_x_stylo = all_feats(train_x)

                        valid_x_stylo = get_all_feats(valid_x_stylo_raw, posCV)
                        # valid_x_stylo = all_feats(valid_x)

                        tests_x_stylo = get_all_feats(tests_x_stylo_raw, posCV)
                        # tests_x_stylo = all_feats(tests_x)

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

                        SS2 = StandardScaler(copy=False)
                        train_x_stylo = SS2.fit_transform(train_x_stylo)
                        valid_x_stylo = SS2.transform(valid_x_stylo)
                        tests_x_stylo = SS2.transform(tests_x_stylo)

                        train_x_all = np.hstack((train_x_ngrams, train_x_stylo))
                        valid_x_all = np.hstack((valid_x_ngrams, valid_x_stylo))
                        tests_x_all = np.hstack((tests_x_ngrams, tests_x_stylo))

                        num_feat = np.shape(train_x_all)[1]

                        """## Classification"""

                        loss = 'hinge'

                        # print(FLAGS.scenario.upper() + " Topic Authorship Attribution")

                        """### all """
                        train, valid, tests = [], [], []
                        for i in range(FLAGS.run_times):
                            clf_all = None
                            clf_all = SGDClassifier(loss=loss, max_iter=2000, tol=1e-3, class_weight='balanced',
                                                      n_jobs=-1)
                            clf_all.fit(train_x_all, y_train)

                            train_score = balanced_accuracy_score(y_train, clf_all.predict(train_x_all))
                            # print("Training acc:\t {:.2f}".format(train_score))
                            train.append(train_score)

                            valid_score = balanced_accuracy_score(y_valid, clf_all.predict(valid_x_all))
                            # print("Validation acc:\t {:.2f}".format(valid_score))
                            valid.append(valid_score)

                            tests_score = balanced_accuracy_score(y_tests, clf_all.predict(tests_x_all))
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
