# -*- coding: utf-8 -*-
from absl import logging

import os
from absl import flags
from absl import app
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

import numpy as np
from utils import loadMyData
import myDatasets

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
from sklearn.model_selection import StratifiedShuffleSplit
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
        self.updateChar()
        self.updateWord()
        self.updateSynt()
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


class MaskingDemon(object):
    def __init__(self):
        super(MaskingDemon, self).__init__()

def Masking(docs, FLAGS, demon, mode="MA"):

    docs = [re.findall('\w+|[!"$%$&\'+(),-./:;<=>?@[\]^_`{|}~]', doc) for doc in docs]

    newDocs = []
    for doc in docs:
        newDoc = []
        for tok in doc:
            if not tok.lower() in demon.preserve_list:
                tok = re.sub('[A-Za-z]', '*', tok)
                # tok = ""

            if mode == "SA":
                tok = re.sub('\*\*+', '*', tok)
                tok = re.sub('\#\#+', '#', tok)

            newDoc.append(tok)
        newDocs.append(newDoc)

    return newDocs


def vectorize_Docs(newDocs, demon, FLAGS, training_set=False):
    newDocs = [" ".join(doc).strip() for doc in newDocs]

    if training_set:
        # # Instead we are going to use corpus frequency, and ignore max_features
        # counter = collections.Counter([x for w in newDocs for x in w])
        # toIgnore = [key for key, val in counter.items() if val < FLAGS.freq_threshold]
        # # some = counter['Ken']

        if demon.corpus_counts == []:
            CV = CountVectorizer(lowercase=False, analyzer='char',
                                 ngram_range=(FLAGS.min_n, FLAGS.max_n))
            all_feats = CV.fit_transform(newDocs).toarray()
            vocab_ = CV.get_feature_names()
            summed_feats = np.sum(all_feats, axis=0)

            demon.corpus_counts = dict([(vocab_[i], summed_feats[i]) for i in range(len(vocab_))])

        new_vocab = [key for key, val in demon.corpus_counts.items() if val >= FLAGS.freq_threshold]

        # if not hasattr(demon, 'CV'):
        #     # izer = RegexpTokenizer('\*+|\#+|\w+|[!"$%$&\'+(),-./:;<=>?@[\]^_`{|}~]')

        demon.CV = CountVectorizer(analyzer='char', lowercase=False, tokenizer=lambda x : x,
                                   ngram_range=(FLAGS.min_n, FLAGS.max_n), vocabulary=new_vocab)

        docCount = demon.CV.transform(newDocs).toarray()  ##change it to an array

    else:
        docCount = demon.CV.transform(newDocs).toarray()

    return docCount

def getPreservedWords(external_source, puncs=False):
    presList = external_source[0][0].split(" ")

    if puncs:
        for p in "!\"$%&'()+,-./:;<=>?@[]^_`{|}~":  # the hashtag and star are removed
            presList.append(p)

    return presList

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
        stylo_vecs = p_doc.vectorizeAll()
        ngrams_vecs = CV.transform([p_doc.posFeats.tags]).toarray()

        ngrams_vecs = ngrams_vecs.astype(np.int).tolist()[0]
        stylo_vecs.extend(ngrams_vecs)

        vectors.append(stylo_vecs)

    return vectors


def main(argv):
    print("case \t Scenario \t K \t Ft \t n \t num_feats \t train \t valid \t test")
    with open('/' + str(os.path.basename(__file__).split('.')[0]) + '.txt', '+w') as resFile:
        resFile.write("case \t Scenario \t K \t Ft \t n \t num_feats \t train \t valid \t test \n")
        for case in range(1, 13):
            FLAGS.guardian_case = case

            """## Reading Data"""

            # region read data
            datasets = loadMyData(FLAGS)

            train_1_x_raw, train_1_y, train_1_t = datasets[0][0], datasets[0][1], datasets[0][2]
            train_2_x_raw, train_2_y, train_2_t = datasets[0][3], datasets[0][4], datasets[0][5]

            valid_x_raw, valid_y = datasets[1][0], datasets[1][1]

            tests_x_raw, tests_y = datasets[2][0], datasets[2][1]

            # print('Number of instances --> train1: {}, train2: {}, valid: {}, test: {}'.format(len(train_1_y), len(train_2_y),
            #                                                                                    len(valid_y), len(tests_y)))

            # print("case \t Scenario \t K \t Ft \t n \t train \t valid \t test")
            # endregion

            # for scenario in ['same', 'cross']:
            for scenario in ['cross']:
                FLAGS.scenario = scenario

                """
                ## Data Split and Feat Ex.
                ### Split data based on authorship scenario
                """

                if FLAGS.scenario == 'same':
                    X_raw = train_1_x_raw + train_2_x_raw + valid_x_raw + tests_x_raw
                    Y_raw = train_1_y + train_2_y + valid_y + tests_y

                    train_x, valid_x, y_train, y_valid = train_test_split(
                        X_raw, Y_raw, test_size=0.70, random_state=42)

                    valid_x, tests_x, y_valid, y_tests = train_test_split(
                        valid_x, y_valid, test_size=0.56, random_state=42)

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

                # start over for ngrams , but mask digits
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

                for k in [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]:
                    FLAGS.k = k

                    # elif masking
                    demon = None
                    demon = MaskingDemon()

                    demon.external_source = myDatasets.getDS(FLAGS, external=True)
                    demon.preserve_list = getPreservedWords(demon.external_source) if len(demon.external_source) > 0 else []

                    train_x_masked_raw = Masking(train_x, FLAGS, demon)
                    valid_x_masked_raw = Masking(valid_x, FLAGS, demon)
                    tests_x_masked_raw = Masking(tests_x, FLAGS, demon)

                    for n in [3, 4, 5, 6, 7, 8]:
                        FLAGS.min_n = n
                        FLAGS.max_n = n

                        posCV = CountVectorizer(lowercase=False, analyzer='word', tokenizer=lambda x: x,
                                                # token_pattern =re.compile("\S+"),
                                                ngram_range=(n, n))

                        # some = counter['Ken']
                        all_feats = posCV.fit_transform([p_doc.posFeats.tags for p_doc in train_x_stylo_raw]).toarray()
                        vocab_pos = posCV.get_feature_names()
                        summed_feats_pos = np.sum(all_feats, axis=0)

                        demon.corpus_counts = []

                        for ft in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
                            if 'CV' in demon.__dict__.keys():
                                del demon.CV

                            keep = [vocab_pos[i] for i in range(len(vocab_pos)) if summed_feats_pos[i] >= ft]
                            posCV.vocabulary_ = dict([(keep[i], i) for i in range(len(keep))])

                            if not posCV.vocabulary_:
                                continue

                            FLAGS.freq_threshold = ft
                            resFile.flush()

                            """### Feature extraction"""
                            # mask the dataset
                            train_x_masked, valid_x_masked, tests_x_masked= None, None, None

                            train_x_masked = vectorize_Docs(train_x_masked_raw, demon, FLAGS, training_set=True)
                            valid_x_masked = vectorize_Docs(valid_x_masked_raw, demon, FLAGS)
                            tests_x_masked = vectorize_Docs(tests_x_masked_raw, demon, FLAGS)

                            train_x_stylo = get_all_feats(train_x_stylo_raw, posCV)
                            valid_x_stylo = get_all_feats(valid_x_stylo_raw, posCV)
                            tests_x_stylo = get_all_feats(tests_x_stylo_raw, posCV)

                            train_x_masked_all = np.hstack((train_x_masked, train_x_stylo))
                            valid_x_masked_all = np.hstack((valid_x_masked, valid_x_stylo))
                            tests_x_masked_all = np.hstack((tests_x_masked, tests_x_stylo))

                            SS2 = StandardScaler(copy=False)
                            train_x_masked_all = SS2.fit_transform(train_x_masked_all)
                            valid_x_masked_all = SS2.transform(valid_x_masked_all)
                            tests_x_masked_all = SS2.transform(tests_x_masked_all)

                            num_feat = np.shape(train_x_masked_all)[1]
                            """## Classification"""

                            loss = 'hinge'

                            # print(FLAGS.scenario.upper() + " Topic Authorship Attribution")

                            """### ngrams"""
                            train, valid, tests = [], [], []

                            for i in range(FLAGS.run_times):
                                clf_masked = None
                                clf_masked = SGDClassifier(loss=loss, max_iter=2000, tol=1e-3, class_weight='balanced',
                                                           n_jobs=-1)
                                clf_masked.fit(train_x_masked_all, y_train)

                                train_score = balanced_accuracy_score(y_train, clf_masked.predict(train_x_masked_all))
                                # print("Training acc:\t {:.2f}".format(train_score),end='\t')
                                train.append(train_score)

                                valid_score = balanced_accuracy_score(y_valid, clf_masked.predict(valid_x_masked_all))
                                # print("Validation acc:\t {:.2f}".format(valid_score),end='\t')
                                valid.append(valid_score)

                                tests_score = balanced_accuracy_score(y_tests, clf_masked.predict(tests_x_masked_all))
                                # print("Testing acc:\t {:.2f}".format(tests_score))
                                tests.append(tests_score)

                                # runs.append
                            resFile.write(
                                f'{FLAGS.guardian_case} \t {FLAGS.scenario} \t\t {k} \t {ft} \t {n} \t {num_feat} \t\t'
                            )

                            resFile.write("{:.2f} ± {:.1f} \t {:.2f} ± {:.1f} \t {:.2f} ± {:.1f}\n".format(
                                100 * np.average(train), 100 * np.std(train),
                                100 * np.average(valid), 100 * np.std(valid),
                                100 * np.average(tests), 100 * np.std(tests)
                            ))

                            print(FLAGS.guardian_case, '\t\t', FLAGS.scenario, "\t\t", k, '\t', ft, '\t', n, '\t', num_feat, '\t\t', end=' ')
                            print("{:.2f} ± {:.1f} \t {:.2f} ± {:.1f} \t {:.2f} ± {:.1f}".format(
                                100 * np.average(train), 100 * np.std(train),
                                100 * np.average(valid), 100 * np.std(valid),
                                100 * np.average(tests), 100 * np.std(tests)
                            ))

    return 0

if __name__ == "__main__":
    app.run(main)
