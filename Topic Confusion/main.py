import collections
import re

# NLP
import nltk
import numpy as np
from absl import app
from absl import flags
from absl import logging
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import CountVectorizer
# ML
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

import myDatasets
# classes
from myClasses import Parsed
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

# region Define classes and functions
class MaskingDemon(object):
    def __init__(self):
        super(MaskingDemon, self).__init__()
        self.preserve_list = None


def getPreservedWords(external_source, puncs=False):
    presList = external_source[0][0].split(" ")

    if puncs:
        for p in "!\"$%&'()+,-./:;<=>?@[]^_`{|}~":  # the hashtag and star are removed
            presList.append(p)

    return presList


def Masking(docs, demon, mode="MA"):
    # mask_buckets = FLAGS.BUCKET_MASK
    # if not hasattr(demon, 'CV'):
    #     print("Creating Count Vectorizer") if FLAGS.verbose else None
    #     # demon.CV = CountVectorizer(vocabulary=demon.preserve_list, tokenizer=izer.tokenize)
    #     if FLAGS.REM_MASK_FEATS:
    #         izer = RegexpTokenizer('\w+|[!"$%$&\'+(),-./:;<=>?@[\]^_`{|}~]')
    #     else:
    #         izer = RegexpTokenizer('\*+|\#+|\w+|[!"$%$&\'+(),-./:;<=>?@[\]^_`{|}~]')

    #     demon.CV = CountVectorizer(analyzer=FLAGS.NGRAM_LEVEL, lowercase=FLAGS.LOWER_CASE, tokenizer=izer.tokenize,
    #                                ngram_range=(FLAGS.MIN_N, FLAGS.MAX_N),
    #                                min_df=FLAGS.MIN_FREQ)

    # docs, demon.Max_Seq_Len = tokenizeDocs(docs)

    # docs = [doc.split(' ') for doc in docs]
    docs = [re.findall('\w+|[!"$%$&\'+(),-./:;<=>?@[\]^_`{|}~]', doc) for doc in docs]
    # demon.Max_Seq_Len = len(max(docs, key=len))

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

            # if FLAGS.BUCKET_MASK:
            #     # buckets are 1,2,3,4,5, 6-12, 13 #where, nevertheless, more
            #     if 6 < len(tok) <= 12:
            #         tok = tok[:6]
            #     elif len(tok) > 12:
            #         tok = tok[:13]

            newDoc.append(tok)
        newDocs.append(newDoc)

    return newDocs


def vectorize_Docs(newDocs, demon, FLAGS, training_set=False):
    if training_set:
        # # Instead we are going to use corpus frequency, and ignore max_features
        # counter = collections.Counter([x for w in newDocs for x in w])
        # toIgnore = [key for key, val in counter.items() if val < FLAGS.freq_threshold]
        # # some = counter['Ken']

        if demon.corpus_counts == []:
            corpus_counts = collections.Counter()
            for n in range(FLAGS.min_n, FLAGS.max_n + 1):
                corpus_counts.update([" ".join(gram) for doc in newDocs for gram in nltk.ngrams(doc, n)])
            demon.corpus_counts = corpus_counts

        new_vocab = [key for key, val in demon.corpus_counts.items() if val >= FLAGS.freq_threshold]

        # if not hasattr(demon, 'CV'):
        #     # izer = RegexpTokenizer('\*+|\#+|\w+|[!"$%$&\'+(),-./:;<=>?@[\]^_`{|}~]')

        demon.CV = CountVectorizer(analyzer=FLAGS.ngram_level, lowercase=False, tokenizer=lambda x: x,
                                   ngram_range=(FLAGS.min_n, FLAGS.max_n), vocabulary=new_vocab)

        docCount = demon.CV.transform(newDocs).toarray()  ##change it to an array

    else:
        docCount = demon.CV.transform(newDocs).toarray()

    return docCount


def vectorize_Docs_char(newDocs, demon, FLAGS, training_set=False):
    newDocs = [" ".join(doc).strip() for doc in newDocs]

    if training_set:
        # # Instead we are going to use corpus frequency, and ignore max_features
        # counter = collections.Counter([x for w in newDocs for x in w])
        # toIgnore = [key for key, val in counter.items() if val < FLAGS.freq_threshold]
        # # some = counter['Ken']

        if demon.corpus_counts == []:
            CV = CountVectorizer(lowercase=False, analyzer='char', ngram_range=(FLAGS.min_n, FLAGS.max_n))
            all_feats = CV.fit_transform(newDocs).toarray()
            vocab_ = CV.get_feature_names()
            summed_feats = np.sum(all_feats, axis=0)

            demon.corpus_counts = dict([(vocab_[i], summed_feats[i]) for i in range(len(vocab_))])

        new_vocab = [key for key, val in demon.corpus_counts.items() if val >= FLAGS.freq_threshold]

        # if not hasattr(demon, 'CV'):
        #     # izer = RegexpTokenizer('\*+|\#+|\w+|[!"$%$&\'+(),-./:;<=>?@[\]^_`{|}~]')

        demon.CV = CountVectorizer(lowercase=False, analyzer='char', tokenizer=lambda x: x,
                                   ngram_range=(FLAGS.min_n, FLAGS.max_n), vocabulary=new_vocab)

        docCount = demon.CV.transform(newDocs).toarray()  ##change it to an array

    else:
        docCount = demon.CV.transform(newDocs).toarray()

    return docCount


def extract_all_feats(docs):  # except POS ngrams
    processed_docs = []
    pos_tags = []
    for doc in docs:
        p_doc = Parsed(doc)
        p_doc.updateAll()
        processed_docs.append(p_doc)
        pos_tags.append(p_doc.posFeats.tags)

    return processed_docs


def get_all_feats(processedDocs, CV):
    vectors = []
    for p_doc in processedDocs:
        stylo_vecs = p_doc.vectorizeAll()
        ngrams_vecs = CV.transform([p_doc.posFeats.tags])

        stylo_vecs.extend([x for x in ngrams_vecs.toarray()[0]])
        vectors.append(stylo_vecs)

    return vectors


def get_POS_feats(processedDocs, CV=None):
    vectors = []
    for p_doc in processedDocs:
        # stylo_vecs = p_doc.vectorizeAll()
        stylo_vecs = []
        ngrams_vecs = CV.transform([p_doc.posFeats.tags]).toarray()
        ngrams_vecs = ngrams_vecs.astype(np.int).tolist()[0]
        stylo_vecs.extend(ngrams_vecs)

        vectors.append(stylo_vecs)

    return vectors


def get_stylo_feats(processedDocs):
    vectors = []
    for p_doc in processedDocs:
        stylo_vecs = p_doc.vectorizeAll()
        # ngrams_vecs = CV.transform([p_doc.posFeats.tags])
        #
        # stylo_vecs.extend([x for x in ngrams_vecs.toarray()[0]])
        vectors.append(stylo_vecs)

    return vectors

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


def tune_char_ngrams(Ns, Fts,
                     train_x_raw, y_train, tests_x_raw, y_tests,
                     train_x_stylo_raw=[], tests_x_stylo_raw=[],
                     with_stylo=False, with_pos=False,
                     loss='hinge', return_best=False):
    best_n, best_ft, best_num_feats, best_acc = 0, 0, 0, 0.0
    for n in Ns:
        CV_char = CountVectorizer(lowercase=False, analyzer='char', ngram_range=(n, n))

        all_feats_ngrams = CV_char.fit_transform(train_x_raw).toarray()
        vocab_ = CV_char.get_feature_names()
        summed_feats = np.sum(all_feats_ngrams, axis=0)

        if with_stylo:
            if with_pos:
                posCV = CountVectorizer(lowercase=False, analyzer='word', tokenizer=lambda x: x,
                                        ngram_range=(n, n))

                all_feats = posCV.fit_transform([p_doc.posFeats.tags for p_doc in train_x_stylo_raw]).toarray()
                vocab_pos = posCV.get_feature_names()
                summed_feats_pos = np.sum(all_feats, axis=0)

        for ft in Fts:
            keep_ngrams = [vocab_[i] for i in range(len(vocab_)) if summed_feats[i] >= ft]
            CV_char.vocabulary_ = dict([(keep_ngrams[i], i) for i in range(len(keep_ngrams))])

            if not CV_char.vocabulary_:
                continue

            train_x_char_ngrams = CV_char.transform(train_x_raw).toarray()
            train_x_char_ngrams = train_x_char_ngrams.astype(np.float32)
            tests_x_char_ngrams = CV_char.transform(tests_x_raw).toarray()
            tests_x_char_ngrams = tests_x_char_ngrams.astype(np.float32)

            if with_stylo:
                if with_pos:
                    keep = [vocab_pos[i] for i in range(len(vocab_pos)) if summed_feats_pos[i] >= ft]
                    posCV.vocabulary_ = dict([(keep[i], i) for i in range(len(keep))])

                    if not posCV.vocabulary_:
                        continue

                    train_x_stylo = get_all_feats(train_x_stylo_raw, posCV)
                    tests_x_stylo = get_all_feats(tests_x_stylo_raw, posCV)
                else:
                    train_x_stylo = get_stylo_feats(train_x_stylo_raw)
                    tests_x_stylo = get_stylo_feats(tests_x_stylo_raw)

                train_x_char_ngrams = np.hstack((train_x_char_ngrams, train_x_stylo))
                tests_x_char_ngrams = np.hstack((tests_x_char_ngrams, tests_x_stylo))

            SS1 = StandardScaler(copy=False)
            train_x_char_ngrams = SS1.fit_transform(train_x_char_ngrams)
            tests_x_char_ngrams = SS1.transform(tests_x_char_ngrams)

            num_feat = np.shape(train_x_char_ngrams)[1]

            if return_best:
                return train_x_char_ngrams, tests_x_char_ngrams

            sum_accs = []
            for run in range(10):
                clf_ngrams = None
                clf_ngrams = SGDClassifier(loss=loss, max_iter=2000, tol=1e-3, class_weight='balanced', n_jobs=-1)
                clf_ngrams.fit(train_x_char_ngrams, y_train)
                preds_train = clf_ngrams.predict(train_x_char_ngrams)
                preds_tests = clf_ngrams.predict(tests_x_char_ngrams)

                sum_accs.append(balanced_accuracy_score(y_tests, preds_tests))

            avg_accs = sum(sum_accs) / len(sum_accs)
            if avg_accs > best_acc:
                best_acc = avg_accs
                best_n = n
                best_ft = ft
                best_num_feats = num_feat
                best_train = preds_train
                best_tests = preds_tests

            elif avg_accs == best_acc and num_feat < best_num_feats:
                best_acc = avg_accs
                best_n = n
                best_ft = ft
                best_num_feats = num_feat
                best_train = preds_train
                best_tests = preds_tests

    return best_n, best_ft, best_num_feats, best_acc, best_train, best_tests


def tune_word_ngrams(Ns, Fts,
                     train_x_raw, y_train, tests_x_raw, y_tests,
                     train_x_stylo_raw=[], tests_x_stylo_raw=[],
                     with_stylo=False, with_pos=False,
                     loss='hinge', return_best=False):
    best_n, best_ft, best_num_feats, best_acc = 0, 0, 0, 0.0
    for n in Ns:
        izer = RegexpTokenizer('\\w+|[!"$%$&\'+(),-./:;<=>?@[]^_`{|}~]')

        CV = CountVectorizer(lowercase=False, tokenizer=izer.tokenize, ngram_range=(n, n))

        all_feats_ngrams = CV.fit_transform(train_x_raw).toarray()
        vocab_ = CV.get_feature_names()
        summed_feats = np.sum(all_feats_ngrams, axis=0)

        if with_stylo:
            if with_pos:
                posCV = CountVectorizer(lowercase=False, analyzer='word', tokenizer=lambda x: x,
                                        ngram_range=(n, n))

                all_feats = posCV.fit_transform([p_doc.posFeats.tags for p_doc in train_x_stylo_raw]).toarray()
                vocab_pos = posCV.get_feature_names()
                summed_feats_pos = np.sum(all_feats, axis=0)

        for ft in Fts:
            keep_ngrams = [vocab_[i] for i in range(len(vocab_)) if summed_feats[i] >= ft]
            CV.vocabulary_ = dict([(keep_ngrams[i], i) for i in range(len(keep_ngrams))])

            if not CV.vocabulary_:
                continue

            train_x_word_ngrams = CV.transform(train_x_raw).toarray()
            train_x_word_ngrams = train_x_word_ngrams.astype(np.float32)
            tests_x_word_ngrams = CV.transform(tests_x_raw).toarray()
            tests_x_word_ngrams = tests_x_word_ngrams.astype(np.float32)

            if with_stylo:
                if with_pos:
                    keep = [vocab_pos[i] for i in range(len(vocab_pos)) if summed_feats_pos[i] >= ft]
                    posCV.vocabulary_ = dict([(keep[i], i) for i in range(len(keep))])

                    if not posCV.vocabulary_:
                        continue

                    train_x_stylo = get_all_feats(train_x_stylo_raw, posCV)
                    tests_x_stylo = get_all_feats(tests_x_stylo_raw, posCV)
                else:
                    train_x_stylo = get_stylo_feats(train_x_stylo_raw)
                    tests_x_stylo = get_stylo_feats(tests_x_stylo_raw)

                train_x_word_ngrams = np.hstack((train_x_word_ngrams, train_x_stylo))
                tests_x_word_ngrams = np.hstack((tests_x_word_ngrams, tests_x_stylo))

            SS1 = StandardScaler(copy=False)
            train_x_word_ngrams = SS1.fit_transform(train_x_word_ngrams)
            tests_x_word_ngrams = SS1.transform(tests_x_word_ngrams)

            if return_best:
                return train_x_word_ngrams, tests_x_word_ngrams

            num_feat = np.shape(train_x_word_ngrams)[1]

            sum_accs = []
            for run in range(10):
                clf_ngrams = None
                clf_ngrams = SGDClassifier(loss=loss, max_iter=2000, tol=1e-3, class_weight='balanced', n_jobs=-1)
                clf_ngrams.fit(train_x_word_ngrams, y_train)
                preds_train = clf_ngrams.predict(train_x_word_ngrams)
                preds_tests = clf_ngrams.predict(tests_x_word_ngrams)

                sum_accs.append(balanced_accuracy_score(y_tests, preds_tests))

            avg_accs = sum(sum_accs) / len(sum_accs)
            if avg_accs > best_acc:
                best_acc = avg_accs
                best_n = n
                best_ft = ft
                best_num_feats = num_feat
                best_train = preds_train
                best_tests = preds_tests

            elif avg_accs == best_acc and num_feat < best_num_feats:
                best_acc = avg_accs
                best_n = n
                best_ft = ft
                best_num_feats = num_feat
                best_train = preds_train
                best_tests = preds_tests

    return best_n, best_ft, best_num_feats, best_acc, best_train, best_tests


def tune_pos_ngrams(Ns, Fts, train_x_stylo_raw, y_train, tests_x_stylo_raw, y_tests, loss='hinge',
                    with_stylo=False, return_best=False):
    best_n, best_ft, best_num_feats, best_acc = 0, 0, 0, 0.0
    for n in Ns:
        posCV = CountVectorizer(lowercase=False, analyzer='word', tokenizer=lambda x: x, ngram_range=(n, n))

        all_feats = posCV.fit_transform([p_doc.posFeats.tags for p_doc in train_x_stylo_raw]).toarray()
        vocab_pos = posCV.get_feature_names()
        summed_feats_pos = np.sum(all_feats, axis=0)

        for ft in Fts:
            keep = [vocab_pos[i] for i in range(len(vocab_pos)) if summed_feats_pos[i] >= ft]
            posCV.vocabulary_ = dict([(keep[i], i) for i in range(len(keep))])

            if not posCV.vocabulary_:
                continue

            if with_stylo:
                train_x_POS_ngrams = get_all_feats(train_x_stylo_raw, posCV)
                tests_x_POS_ngrams = get_all_feats(tests_x_stylo_raw, posCV)
            else:
                train_x_POS_ngrams = get_POS_feats(train_x_stylo_raw, posCV)
                tests_x_POS_ngrams = get_POS_feats(tests_x_stylo_raw, posCV)

            SS1 = StandardScaler(copy=False)
            train_x_POS_ngrams = SS1.fit_transform(train_x_POS_ngrams)
            tests_x_POS_ngrams = SS1.transform(tests_x_POS_ngrams)

            if return_best:
                return train_x_POS_ngrams, tests_x_POS_ngrams
            num_feat = np.shape(train_x_POS_ngrams)[1]

            sum_accs = []
            for run in range(10):
                clf_POS = None
                clf_POS = SGDClassifier(loss=loss, max_iter=2000, tol=1e-3, class_weight='balanced', n_jobs=-1)
                clf_POS.fit(train_x_POS_ngrams, y_train)
                preds_train = clf_POS.predict(train_x_POS_ngrams)
                preds_tests = clf_POS.predict(tests_x_POS_ngrams)

                sum_accs.append(balanced_accuracy_score(y_tests, preds_tests))

            avg_accs = sum(sum_accs) / len(sum_accs)
            if avg_accs > best_acc:
                best_acc = avg_accs
                best_n = n
                best_ft = ft
                best_num_feats = num_feat
                best_train = preds_train
                best_tests = preds_tests

            elif avg_accs == best_acc and num_feat < best_num_feats:
                best_acc = avg_accs
                best_n = n
                best_ft = ft
                best_num_feats = num_feat
                best_train = preds_train
                best_tests = preds_tests

    return best_n, best_ft, best_num_feats, best_acc, best_train, best_tests


def tune_masked_char(Ns, Fts, Ks, preserve_list, train_x_raw, y_train, tests_x_raw, y_tests,
                    train_x_stylo_raw=[], tests_x_stylo_raw=[],
                     with_stylo=False, with_pos=False,
                     loss='hinge', return_best=False):
    best_n, best_ft, best_k, best_num_feats, best_acc = 0, 0, 0, 0, 0.0
    for k in Ks:
        # FLAGS.k = k
        demon = None
        demon = MaskingDemon()

        # demon.external_source = myDatasets.getDS(FLAGS, external=True)
        # demon.preserve_list = getPreservedWords(demon.external_source) if len(demon.external_source) > 0 else []
        demon.preserve_list = preserve_list[:k]

        # mask the dataset
        train_x_masked_raw = Masking(train_x_raw, demon)
        tests_x_masked_raw = Masking(tests_x_raw, demon)

        for n in Ns:
            FLAGS.min_n = n
            FLAGS.max_n = n

            if with_stylo:
                if with_pos:
                    posCV = CountVectorizer(lowercase=False, analyzer='word', tokenizer=lambda x: x,
                                            ngram_range=(n, n))

                    all_feats = posCV.fit_transform([p_doc.posFeats.tags for p_doc in train_x_stylo_raw]).toarray()
                    vocab_pos = posCV.get_feature_names()
                    summed_feats_pos = np.sum(all_feats, axis=0)

            demon.corpus_counts = []
            for ft in Fts:
                if 'CV' in demon.__dict__.keys():
                    del demon.CV

                FLAGS.freq_threshold = ft

                train_x_masked, valid_x_masked, tests_x_masked = None, None, None

                train_x_masked = vectorize_Docs_char(train_x_masked_raw, demon, FLAGS, training_set=True)
                tests_x_masked = vectorize_Docs_char(tests_x_masked_raw, demon, FLAGS)

                if with_stylo:
                    if with_pos:
                        keep = [vocab_pos[i] for i in range(len(vocab_pos)) if summed_feats_pos[i] >= ft]
                        posCV.vocabulary_ = dict([(keep[i], i) for i in range(len(keep))])

                        if not posCV.vocabulary_:
                            continue

                        train_x_stylo = get_all_feats(train_x_stylo_raw, posCV)
                        tests_x_stylo = get_all_feats(tests_x_stylo_raw, posCV)
                    else:
                        train_x_stylo = get_stylo_feats(train_x_stylo_raw)
                        tests_x_stylo = get_stylo_feats(tests_x_stylo_raw)

                    train_x_masked = np.hstack((train_x_masked, train_x_stylo))
                    tests_x_masked = np.hstack((tests_x_masked, tests_x_stylo))

                SS2 = StandardScaler(copy=False)
                train_x_masked = SS2.fit_transform(train_x_masked)
                tests_x_masked = SS2.transform(tests_x_masked)

                if return_best:
                    return train_x_masked, tests_x_masked

                num_feat = np.shape(train_x_masked)[1]
                sum_accs = []
                for run in range(10):
                    clf_POS = None
                    clf_POS = SGDClassifier(loss=loss, max_iter=2000, tol=1e-3, class_weight='balanced', n_jobs=-1)
                    clf_POS.fit(train_x_masked, y_train)
                    preds_train = clf_POS.predict(train_x_masked)
                    preds_tests = clf_POS.predict(tests_x_masked)

                    sum_accs.append(balanced_accuracy_score(y_tests, preds_tests))

                avg_accs = sum(sum_accs) / len(sum_accs)
                if avg_accs > best_acc:
                    best_acc = avg_accs
                    best_n = n
                    best_ft = ft
                    best_k = k
                    best_num_feats = num_feat
                    best_train = preds_train
                    best_tests = preds_tests

                elif avg_accs == best_acc and num_feat < best_num_feats:
                    best_acc = avg_accs
                    best_n = n
                    best_ft = ft
                    best_k = k
                    best_num_feats = num_feat
                    best_train = preds_train
                    best_tests = preds_tests

    return best_n, best_ft, best_k, best_num_feats, best_acc, best_train, best_tests


def tune_masked_word(Ns, Fts, Ks, train_x_raw, y_train, tests_x_raw, y_tests,
                        train_x_stylo_raw=[], tests_x_stylo_raw=[],
                        with_stylo=False, with_pos=False,
                        loss='hinge', return_best=False):
    best_n, best_ft, best_k, best_num_feats, best_acc = 0, 0, 0, 0, 0.0
    for k in Ks:
        FLAGS.k = k
        demon = None
        demon = MaskingDemon()

        demon.external_source = myDatasets.getDS(FLAGS, external=True)
        demon.preserve_list = getPreservedWords(demon.external_source) if len(demon.external_source) > 0 else []

        # mask the dataset
        train_x_masked_raw = Masking(train_x_raw, demon)
        tests_x_masked_raw = Masking(tests_x_raw, demon)

        for n in Ns:
            FLAGS.min_n = n
            FLAGS.max_n = n

            if with_stylo:
                if with_pos:
                    posCV = CountVectorizer(lowercase=False, analyzer='word', tokenizer=lambda x: x,
                                            ngram_range=(n, n))

                    all_feats = posCV.fit_transform([p_doc.posFeats.tags for p_doc in train_x_stylo_raw]).toarray()
                    vocab_pos = posCV.get_feature_names()
                    summed_feats_pos = np.sum(all_feats, axis=0)

            demon.corpus_counts = []
            for ft in Fts:
                if 'CV' in demon.__dict__.keys():
                    del demon.CV

                FLAGS.freq_threshold = ft

                train_x_masked, valid_x_masked, tests_x_masked = None, None, None

                train_x_masked = vectorize_Docs(train_x_masked_raw, demon, FLAGS, training_set=True)
                tests_x_masked = vectorize_Docs(tests_x_masked_raw, demon, FLAGS)

                if with_stylo:
                    if with_pos:
                        keep = [vocab_pos[i] for i in range(len(vocab_pos)) if summed_feats_pos[i] >= ft]
                        posCV.vocabulary_ = dict([(keep[i], i) for i in range(len(keep))])

                        if not posCV.vocabulary_:
                            continue

                        train_x_stylo = get_all_feats(train_x_stylo_raw, posCV)
                        tests_x_stylo = get_all_feats(tests_x_stylo_raw, posCV)
                    else:
                        train_x_stylo = get_stylo_feats(train_x_stylo_raw)
                        tests_x_stylo = get_stylo_feats(tests_x_stylo_raw)

                    train_x_masked = np.hstack((train_x_masked, train_x_stylo))
                    tests_x_masked = np.hstack((tests_x_masked, tests_x_stylo))

                SS2 = StandardScaler(copy=False)
                train_x_masked = SS2.fit_transform(train_x_masked, )
                tests_x_masked = SS2.transform(tests_x_masked)

                if return_best:
                    return train_x_masked, tests_x_masked

                num_feat = np.shape(train_x_masked)[1]
                sum_accs = []
                for run in range(10):
                    clf_POS = None
                    clf_POS = SGDClassifier(loss=loss, max_iter=2000, tol=1e-3, class_weight='balanced', n_jobs=-1)
                    clf_POS.fit(train_x_masked, y_train)
                    preds_train = clf_POS.predict(train_x_masked)
                    preds_tests = clf_POS.predict(tests_x_masked)

                    sum_accs.append(balanced_accuracy_score(y_tests, preds_tests))

                avg_accs = sum(sum_accs) / len(sum_accs)
                if avg_accs > best_acc:
                    best_acc = avg_accs
                    best_n = n
                    best_ft = ft
                    best_k = k
                    best_num_feats = num_feat
                    best_train = preds_train
                    best_tests = preds_tests

                elif avg_accs == best_acc and num_feat < best_num_feats:
                    best_acc = avg_accs
                    best_n = n
                    best_ft = ft
                    best_k = k
                    best_num_feats = num_feat
                    best_train = preds_train
                    best_tests = preds_tests

    return best_n, best_ft, best_k, best_num_feats, best_acc, best_train, best_tests


def fitter10Times(info2print, x_train, x_tests, y_train, y_tests, random_authors, loss='hinge'):
    s2write = '{}\t{}\t{}\t{}\t{}\t{}\t'.format(info2print[0], info2print[1],
                                            info2print[2], info2print[3], info2print[4],
                                            np.shape(x_train)[1])

    # print(s2write, end='\t')

    trains = []
    tests = []
    ttls, cnt_s, cnt_0s, cnt_1s = 0, 0, 0, 0

    for run in range(10):
        # print('{}\t{}'.format(info2print[0], info2print[1], run), end='\t')
        clf_get = None
        clf_get = SGDClassifier(loss=loss, max_iter=2000, tol=1e-3, class_weight='balanced', n_jobs=-1)
        clf_get.fit(x_train, y_train)
        preds_train = clf_get.predict(x_train)
        preds_tests = clf_get.predict(x_tests)

        trains.append(balanced_accuracy_score(y_train, preds_train))
        tests.append(balanced_accuracy_score(y_tests, preds_tests))

        ttl, cnt_, cnt_0, cnt_1 = confusion_report(preds_tests, y_tests, random_authors)

        ttls += ttl
        cnt_s += cnt_
        cnt_0s += cnt_0
        cnt_1s += cnt_1

    # print(, end='\t')
    s2write += '{:.2f}\t\t{:.2f}\t\t{:.2f}\t'.format(np.average(trains), info2print[5], np.average(tests))

    # print('{:.2f}\t{:.2f}\t{:.2f}\t{}\t{:.2f}\t{}\t{:.2f}\t{}\t{:.2f}'.format(ttls / 10,
    #                                                                           cnt_s / 10, cnt_s / ttls,
    #                                                                           cnt_0s / 10, cnt_0s / ttls,
    #                                                                           cnt_1s / 10, cnt_1s / ttls,
    #                                                                           cnt_2s / 10, cnt_2s / ttls))

    s2write += '{:.2f}\t{:.2f}\t{:.2f}\t{}\t{:.2f}\t{}\t{:.2f}'.format(ttls / 10,
                                                                cnt_s / 10, cnt_s / ttls,
                                                                cnt_0s / 10, cnt_0s / ttls,
                                                                cnt_1s / 10, cnt_1s / ttls)

    return s2write
    # return sum_accs_train/ len(sum_accs_train), sum_accs_tests/len(sum_accs_tests)


# endregion


def main(argv):
    
    # this is the results file
    fname = 'enter_your_file_name_here'
    with open(fname+'.txt', mode='w') as outFi:
        s2write = 'Epoch\tname\tn\tft\tk\tnum_feat\ttraining_acc.\tvalid_acc.\ttest_acc.\tnum_samples' \
                    '\tcorrectly_classified\t_\tL0\t_\tL1\t_ '

        print(s2write)
        s2write += '\r'
        outFi.writelines(s2write)
        outFi.flush()

        # region read confs (These are the randomly selected authors. stored in confs.txt)
        with open('confs.txt', 'r') as confFile:
            randomConfs = confFile.readlines()
        # enregion

        ### We read the data twice, once to tokenize/mask and ... 
        FLAGS.mask_digits = True
        datasets = loadMyData(FLAGS)
        datasets_raw = [(datasets[0][0], datasets[0][1], datasets[0][2]),
                        (datasets[0][3], datasets[0][4], datasets[0][5]),
                        (datasets[1][0], datasets[1][1], datasets[1][2]),
                        (datasets[2][0], datasets[2][1], datasets[2][2])]

        ### ... another one to extract stylometric features -- in case we combine two types of features 
        FLAGS.mask_digits = False
        datasets = loadMyData(FLAGS)
        datasets_stylo = [(extract_all_feats(datasets[0][0]), datasets[0][1], datasets[0][2]),
                        (extract_all_feats(datasets[0][3]), datasets[0][4], datasets[0][5]),
                        (extract_all_feats(datasets[1][0]), datasets[1][1], datasets[1][2]),
                        (extract_all_feats(datasets[2][0]), datasets[2][1], datasets[2][2])]

        FLAGS.k = 5000
        external_source = myDatasets.getDS(FLAGS, external=True)
        presList = external_source[0][0].split(" ")

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

            group_1_x_stylo, group_1_y, group_1_t = datasets_stylo[idx[0]]
            group_2_x_stylo, group_2_y, group_2_t = datasets_stylo[idx[1]]
            group_3_x_stylo, group_3_y, group_3_t = datasets_stylo[idx[2]]
            group_4_x_stylo, group_4_y, group_4_t = datasets_stylo[idx[3]]
            # endregion

            # region split data into groups:
            random_authors = np.array([int(x) for x in randomConfs[epoch][1:-2].split('] [')[1].split()])
            train_x_raw, train_x_stylo_raw, y_train, train_t, \
            valid_x_raw, valid_x_stylo_raw, y_valid, valid_t, \
            tests_x_raw, tests_x_stylo_raw, y_tests, tests_t = [], [], [], [], [], [], [], [], [], [], [], []

            for i in range(len(group_1_x_raw)):
                x_raw, x_stylo, y, t = group_1_x_raw[i], group_1_x_stylo[i], group_1_y[i], group_1_t[i]

                if y in random_authors[:6]:
                    train_x_raw.append(x_raw)
                    train_x_stylo_raw.append(x_stylo)
                    y_train.append(y)
                    train_t.append(t)

                elif y in random_authors[6:]:
                    tests_x_raw.append(x_raw)
                    tests_x_stylo_raw.append(x_stylo)
                    y_tests.append(y)
                    tests_t.append(t)

            for i in range(len(group_2_x_raw)):
                x_raw, x_stylo, y, t = group_2_x_raw[i], group_2_x_stylo[i], group_2_y[i], group_2_t[i]

                if y in random_authors[:6]:
                    tests_x_raw.append(x_raw)
                    tests_x_stylo_raw.append(x_stylo)
                    y_tests.append(y)
                    tests_t.append(t)

                elif y in random_authors[6:]:
                    train_x_raw.append(x_raw)
                    train_x_stylo_raw.append(x_stylo)
                    y_train.append(y)
                    train_t.append(t)

            for i in range(len(group_3_x_raw)):
                x_raw, x_stylo, y, t = group_3_x_raw[i], group_3_x_stylo[i], group_3_y[i], group_3_t[i]
                valid_x_raw.append(x_raw)
                valid_x_stylo_raw.append(x_stylo)
                y_valid.append(y)
                valid_t.append(t)

            for i in range(len(group_4_x_raw)):
                x_raw, x_stylo, y, t = group_4_x_raw[i], group_4_x_stylo[i], group_4_y[i], group_4_t[i]
                valid_x_raw.append(x_raw)
                valid_x_stylo_raw.append(x_stylo)
                y_valid.append(y)
                valid_t.append(t)
            # endregion


            """
            now we start trying different methods using the train/valid/test splits. 
            Each sub region contains two parts, training/fine tuning the method and then using the best hyperparameters to 
            test the method on the test set, specifically in the topic confusion set.
            """ 
            
            # region 1. tune
            ## region 1.a. char n-gram
            Ns = [3, 4, 5, 6, 7, 8]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            best_n, best_ft, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_char_ngrams(Ns, Fts, train_x_raw, y_train, valid_x_raw, y_valid)

            train_x_char_ngrams, tests_x_char_ngrams = tune_char_ngrams([best_n], [best_ft],
                                                                        train_x_raw, y_train, tests_x_raw, y_tests,
                                                                        return_best=True)
            info2print = [epoch, 'char', best_n, best_ft, '-', best_valid_acc]
            s2write = fitter10Times(info2print, train_x_char_ngrams, tests_x_char_ngrams, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion

            ## region 1.b. word n-gram
            Ns = [1, 2, 3]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            best_n, best_ft, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_word_ngrams(Ns, Fts, train_x_raw, y_train, valid_x_raw, y_valid)

            train_x_word_ngrams, tests_x_word_ngrams = tune_word_ngrams([best_n], [best_ft],
                                                                        train_x_raw, y_train, tests_x_raw, y_tests,
                                                                        return_best=True)
            info2print = [epoch, 'word', best_n, best_ft, '-', best_valid_acc]
            s2write = fitter10Times(info2print, train_x_word_ngrams, tests_x_word_ngrams, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion

            ## region 1.c. POS n-gram
            Ns = [1, 2, 3]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            best_n, best_ft, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_pos_ngrams(Ns, Fts, train_x_stylo_raw, y_train, valid_x_stylo_raw, y_valid)

            train_x_POS_ngrams, tests_x_POS_ngrams = tune_pos_ngrams([best_n], [best_ft],
                                                                    train_x_stylo_raw, y_train, tests_x_stylo_raw,
                                                                    y_tests,
                                                                    return_best=True)
            info2print = [epoch, 'POS', best_n, best_ft, '-', best_valid_acc]
            s2write = fitter10Times(info2print, train_x_POS_ngrams, tests_x_POS_ngrams, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion
            # endregion

            # region 2. combine
            ## region 2.a get stylometrics
            train_x_stylo = get_stylo_feats(train_x_stylo_raw)
            valid_x_stylo = get_stylo_feats(valid_x_stylo_raw)
            tests_x_stylo = get_stylo_feats(tests_x_stylo_raw)

            SS2 = StandardScaler(copy=False)
            train_x_stylo = SS2.fit_transform(train_x_stylo)
            valid_x_stylo = SS2.transform(valid_x_stylo)
            tests_x_stylo = SS2.transform(tests_x_stylo)

            info2print = [epoch, 'stylo', '-', '-', '-', 0.0]
            s2write = fitter10Times(info2print, train_x_stylo, tests_x_stylo, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion

            ## region 2.b stylo + char
            Ns = [3, 4, 5, 6, 7, 8]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            best_n, best_ft, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_char_ngrams(Ns, Fts,
                                    train_x_raw, y_train, valid_x_raw, y_valid,
                                    train_x_stylo_raw, valid_x_stylo_raw, with_stylo=True)

            train_x_stylo_char, tests_x_stylo_char = tune_char_ngrams([best_n], [best_ft],
                                                                        train_x_raw, y_train, tests_x_raw, y_tests,
                                                                        train_x_stylo_raw, tests_x_stylo_raw,
                                                                        with_stylo=True,
                                                                        return_best=True)

            info2print = [epoch, 'stylo_char', best_n, best_ft, '-', best_valid_acc]
            s2write = fitter10Times(info2print, train_x_stylo_char, tests_x_stylo_char, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion

            ## region 2.c stylo + word
            Ns = [1, 2, 3]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            best_n, best_ft, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_word_ngrams(Ns, Fts,
                                    train_x_raw, y_train, valid_x_raw, y_valid,
                                    train_x_stylo_raw, valid_x_stylo_raw, with_stylo=True)

            train_x_stylo_word, tests_x_stylo_word = tune_word_ngrams([best_n], [best_ft],
                                                                        train_x_raw, y_train, tests_x_raw, y_tests,
                                                                        train_x_stylo_raw, tests_x_stylo_raw,
                                                                        with_stylo=True,
                                                                        return_best=True)

            info2print = [epoch, 'stylo_word', best_n, best_ft, '-', best_valid_acc]
            s2write = fitter10Times(info2print, train_x_stylo_word, tests_x_stylo_word, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion

            ## region 2.d stylo + pos
            Ns = [1, 2, 3]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            best_n, best_ft, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_pos_ngrams(Ns, Fts, train_x_stylo_raw, y_train, valid_x_stylo_raw, y_valid, with_stylo=True)

            train_x_stylo_POS, tests_x_stylo_POS = tune_pos_ngrams([best_n], [best_ft],
                                                                    train_x_stylo_raw, y_train, tests_x_stylo_raw,
                                                                    y_tests,
                                                                    with_stylo=True,
                                                                    return_best=True)

            info2print = [epoch, 'stylo_pos', best_n, best_ft, '-', best_valid_acc]
            s2write = fitter10Times(info2print, train_x_stylo_POS, tests_x_stylo_POS, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion
            # endregion

            # region 3. all
            ## region 3.a stylo + pos + char
            Ns = [3, 4, 5, 6, 7, 8]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            best_n, best_ft, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_char_ngrams(Ns, Fts,
                                    train_x_raw, y_train, valid_x_raw, y_valid,
                                    train_x_stylo_raw, valid_x_stylo_raw, with_stylo=True, with_pos=True)

            train_x_all_char, tests_x_all_char = tune_char_ngrams([best_n], [best_ft],
                                                                    train_x_raw, y_train, tests_x_raw, y_tests,
                                                                    train_x_stylo_raw, tests_x_stylo_raw,
                                                                    with_stylo=True, with_pos=True,
                                                                    return_best=True)

            info2print = [epoch, 'all_char', best_n, best_ft, '-', best_valid_acc]
            s2write = fitter10Times(info2print, train_x_all_char, tests_x_all_char, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion

            ## region 3.a stylo + pos + word
            Ns = [1, 2, 3]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            best_n, best_ft, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_word_ngrams(Ns, Fts,
                                    train_x_raw, y_train, valid_x_raw, y_valid,
                                    train_x_stylo_raw, valid_x_stylo_raw, with_stylo=True, with_pos=True)

            train_x_all_word, tests_x_all_word = tune_word_ngrams([best_n], [best_ft],
                                                                    train_x_raw, y_train, tests_x_raw, y_tests,
                                                                    train_x_stylo_raw, tests_x_stylo_raw,
                                                                    with_stylo=True, with_pos=True,
                                                                    return_best=True)

            info2print = [epoch, 'all_word', best_n, best_ft, '-', best_valid_acc]
            s2write = fitter10Times(info2print, train_x_all_word, tests_x_all_word, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion
            # endregion


            # region 4. Masked
            ## region 4.a Masked char
            Ns = [3, 4, 5, 6, 7, 8]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            Ks = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
            best_n, best_ft, best_k, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_masked_char(Ns, Fts, Ks, presList, train_x_raw, y_train, valid_x_raw, y_valid)

            train_x_masked_char, tests_x_masked_char = tune_masked_char([best_n], [best_ft], [best_k], presList,
                                                                        train_x_raw, y_train, tests_x_raw, y_tests,
                                                                        return_best=True)

            info2print = [epoch, 'masked_char', best_n, best_ft, best_k, best_valid_acc]
            s2write = fitter10Times(info2print, train_x_masked_char, tests_x_masked_char, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion

            ## region 4.b Masked word
            Ns = [1, 2, 3]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            Ks = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
            best_n, best_ft, best_k, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_masked_word(Ns, Fts, Ks, train_x_raw, y_train, valid_x_raw, y_valid,
                                    train_x_stylo_raw, valid_x_stylo_raw, with_stylo=True, with_pos=True)

            train_x_masked_word, tests_x_masked_word = tune_masked_word([best_n], [best_ft], [best_k],
                                                                        train_x_raw, y_train, tests_x_raw, y_tests,
                                                                        train_x_stylo_raw, valid_x_stylo_raw,
                                                                        with_stylo=True, with_pos=True,
                                                                        return_best=True)

            info2print = [epoch, 'masked_word', best_n, best_ft, best_k, best_valid_acc]
            s2write = fitter10Times(info2print, train_x_masked_word, tests_x_masked_word, y_train, y_tests,
                                    random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion

            ## region 4.c all Masked char
            Ns = [3, 4, 5, 6, 7, 8]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            Ks = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
            best_n, best_ft, best_k, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_masked_char(Ns, Fts, Ks, presList, train_x_raw, y_train, valid_x_raw, y_valid,
                                    train_x_stylo_raw, valid_x_stylo_raw, with_stylo=True, with_pos=True)

            train_x_all_masked_char, tests_x_all_masked_char = tune_masked_char([best_n], [best_ft], [best_k], presList,
                                                                        train_x_raw, y_train, tests_x_raw, y_tests,
                                                                        train_x_stylo_raw, tests_x_stylo_raw,
                                                                        with_stylo=True, with_pos=True,
                                                                        return_best=True)

            info2print = [epoch, 'all_masked_char', best_n, best_ft, best_k, best_valid_acc]
            s2write = fitter10Times(info2print, train_x_all_masked_char, tests_x_all_masked_char, y_train, y_tests, random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion

            ## region 4.d all Masked word
            Ns = [1, 2, 3]
            Fts = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
            Ks = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
            best_n, best_ft, best_k, best_num_feats, best_valid_acc, preds_train, preds_tests \
                = tune_masked_word(Ns, Fts, Ks, train_x_raw, y_train, valid_x_raw, y_valid,
                                    train_x_stylo_raw, valid_x_stylo_raw, with_stylo=True, with_pos=True)

            train_x_all_masked_word, tests_x_all_masked_word = tune_masked_word([best_n], [best_ft], [best_k],
                                                                        train_x_raw, y_train, tests_x_raw, y_tests,
                                                                        train_x_stylo_raw, tests_x_stylo_raw,
                                                                        with_stylo=True, with_pos=True,
                                                                        return_best=True)

            info2print = [epoch, 'all_masked_word', best_n, best_ft, best_k, best_valid_acc]
            s2write = fitter10Times(info2print, train_x_all_masked_word, tests_x_all_masked_word, y_train, y_tests,
                                    random_authors)
            print(s2write)
            s2write += '\r'
            outFi.writelines(s2write)
            outFi.flush()
            ## endregion
            # endregion                        

    return 0

if __name__ == "__main__":
    app.run(main)
