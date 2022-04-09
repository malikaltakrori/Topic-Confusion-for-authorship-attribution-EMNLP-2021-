import numpy as np
import nltk
import re
import string
import collections


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


class CharFeats(object):
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
                                   "#<>%|{}[]/\@~+-*=$&^()_`"])  # 24 feats

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


class ProcessedSample(object):
    def __init__(self, name='ProcessedSample'):
        self.ngrams_w = None
        self.ngrams_ch = None
        self.stylo = None
        self.stylo_pos = None
        self.masked = None
        self.all_feat = None