import pickle

from utilitarian import QuickDataFrame, Progresser
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

NGRAM_PATH = 'E:/ngrams/'


class NgramExcavator:
    def __init__(self, processed_files=[]):
        # read all nouns and verbs and adjectives
        raw_nouns = set(QuickDataFrame.read_csv('./data/Nouns_ Brysbaert_2014.csv', header=False,
                                                columns=['term', 'a', 'b'])['term'])
        self.Lem = WordNetLemmatizer()
        # check the nouns with wWordNet and also use the singular forms only
        self.nouns = set()
        not_in_wordnet = 0
        for w in raw_nouns:
            if len(wordnet.synsets(w)) == 0:
                not_in_wordnet += 1
                continue
            s = self.Lem.lemmatize(w)
            self.nouns.add(s)
            # if s != w:
            #     print(w, '->', s)
            #     a += 1
        print(len(raw_nouns), '->', len(self.nouns), ' | ', not_in_wordnet, 'words not in WordNet.')

        self.adjectives = set(QuickDataFrame.read_csv('./data/Adjectives_Williams_1976_appendix.csv', header=False,
                                                      columns=['term', 'a'])['term'])

        verbs_r = QuickDataFrame.read_csv('./data/Verbs_Levin_1991_ch30-edited.csv', header=False,
                                          columns=['0', '1', '2', '3', '4', '5'])
        self.verbs = dict()
        for i in range(len(verbs_r)):
            for j in range(5):
                self.verbs[verbs_r[str(j)][i]] = verbs_r['0'][i]

        self.start = 1801
        self.end = 2001
        self.processed_files = processed_files
        if not processed_files:
            self.adj_nn = QuickDataFrame(columns=['year'])
            self.vrb_nn = QuickDataFrame(columns=['year'])
            for i in range(self.start, self.end):
                self.adj_nn.append([i])
                self.vrb_nn.append([i])
            self.adj_nn.set_index(self.adj_nn['year'], unique=True)
            self.vrb_nn.set_index(self.adj_nn['year'], unique=True)

            self.adj_nn_list = dict()
            self.vrb_nn_list = dict()

        else:
            # for adjectives
            self.adj_nn = QuickDataFrame.read_csv('./results/adjective_noun.csv')
            for c in self.adj_nn.cols:
                for i in range(len(self.adj_nn)):
                    self.adj_nn[c][i] = int(self.adj_nn[c][i])
            self.adj_nn.set_index(self.adj_nn['year'], unique=True)

            jjnn_ngrams = QuickDataFrame.read_csv('./results/adjective_noun_syntactic_ngrams.csv')
            self.adj_nn_list = dict()
            for i in range(len(jjnn_ngrams)):
                self.adj_nn_list[jjnn_ngrams[i]['query']] = eval(jjnn_ngrams[i]['ngrams'])
            # for verbs
            self.vrb_nn = QuickDataFrame.read_csv('./results/verb_noun.csv')
            for c in self.vrb_nn.cols:
                for i in range(len(self.vrb_nn)):
                    self.vrb_nn[c][i] = int(self.vrb_nn[c][i])
            self.vrb_nn.set_index(self.vrb_nn['year'], unique=True)

            vnn_ngrams = QuickDataFrame.read_csv('./results/verb_noun_syntactic_ngrams.csv')
            self.vrb_nn_list = dict()
            for i in range(len(vnn_ngrams)):
                self.vrb_nn_list[vnn_ngrams[i]['query']] = eval(vnn_ngrams[i]['ngrams'])
        print('init done!')

    def read_them_all(self, target='adj', arc_type='arcs'):
        # open each arc file and search for the combinations
        prog = Progresser(100 - len(self.processed_files))
        for index in range(99):
            if index in self.processed_files:
                continue
            num = str(index)
            if index < 10:
                num = '0' + num
            if target == 'adj':
                with open(NGRAM_PATH + arc_type + '.' + num + '-of-99', encoding='utf-8') as infile:
                    for line in infile:
                        try:
                            self._adj_liner(line)
                        except Exception as e:
                            print(e, ':', line)
            elif target == 'vrb':
                with open(NGRAM_PATH + arc_type + '.' + num + '-of-99', encoding='utf-8') as infile:
                    for line in infile:
                        try:
                            self._vrb_liner(line)
                        except Exception as e:
                            print(e, ':', line)

            self.processed_files.append(index)
            # save every 5 files
            if index % 5 == 0:
                self._save_results(target, index, arc_type)
            prog.count()
        self._save_results(target, 99, arc_type)

    def _save_results(self, target='adj', index=99, arc_type='arcs'):
        if target == 'adj':
            self.adj_nn.to_csv('./results/adjective_noun-' + arc_type + '.csv')
            jjnn_ngrams = QuickDataFrame(['query', 'ngrams'])
            for qry, ngm in self.adj_nn_list.items():
                jjnn_ngrams.append([qry, ngm])
            jjnn_ngrams.to_csv('./results/adjective_noun_syntactic_ngrams-' + arc_type + '.csv')
        elif target == 'vrb':
            self.vrb_nn.to_csv('./results/verb_noun-' + arc_type + '.csv')
            vnn_ngrams = QuickDataFrame(['query', 'ngrams'])
            for qry, ngm in self.vrb_nn_list.items():
                vnn_ngrams.append([qry, ngm])
            vnn_ngrams.to_csv('./results/verb_noun_syntactic_ngrams-' + arc_type + '.csv')

        with open('./results/' + target + '-processed_files.txt', 'w') as outfile:
            outfile.write(str(self.processed_files))
        print('\nsaved until', index)

    def _adj_liner(self, line):
        # each line is head_word<TAB>syntactic-ngram<TAB>total_count<TAB>counts_by_year
        root, t, rest = line.partition('\t')
        if root not in self.nouns:
            return
        s_ngram, t, rest = rest.partition('\t')

        s_ngram_tokens = s_ngram.split(' ')
        # iterate on the tokens to find an adjective and a noun that comes after it
        for i in range(len(s_ngram_tokens)):
            token = s_ngram_tokens[i]
            # each syntactic-ngram is word/pos-tag/dep-label/head-index
            try:
                word1, pos1, dep1, head1 = token.split('/')
            except:
                continue
            # look for an adjective in our list
            if 'JJ' in pos1 and word1 in self.adjectives:
                query = word1 + ' '
                noun_index = int(head1) - 1
                if noun_index < i:  # if the noun is before the adjective
                    continue
                try:
                    word2, pos2, dep2, head2 = s_ngram_tokens[noun_index].split('/')
                except:
                    continue
                if 'NN' in pos2:
                    # if its plural, make it singular
                    if 'S' in pos2:
                        word2 = self.Lem.lemmatize(word2)
                    if word2 in self.nouns:
                        query += word2
                        # if found both adjective and noun add the frequencies to the QDF
                        self._add_to_adj_nn(query, rest, s_ngram)

    def _vrb_liner(self, line):
        # each line is head_word<TAB>syntactic-ngram<TAB>total_count<TAB>counts_by_year
        root, t, rest = line.partition('\t')
        if root not in self.nouns:
            return
        s_ngram, t, rest = rest.partition('\t')

        s_ngram_tokens = s_ngram.split(' ')
        seen_verb = False
        # iterate on the tokens to find a verb and a noun that comes after it
        for i in range(len(s_ngram_tokens)):
            token = s_ngram_tokens[i]
            # each syntactic-ngram is word/pos-tag/dep-label/head-index
            try:
                word, pos, dep, head = token.split('/')
            except:
                continue
            if not seen_verb:
                # look for a verb in our list
                if 'V' in pos and word in self.verbs:
                    query = self.verbs[word] + ' '
                    seen_verb = True
            else:
                if 'NN' in pos:
                    # if its plural, make it singular
                    if 'S' in pos:
                        word = self.Lem.lemmatize(word)
                    if word in self.nouns:
                        query += word
                        # if found both adjective and noun add the frequencies to the QDF
                        self._add_to_vrb_nn(query, rest, s_ngram)
                        return

    def _add_to_adj_nn(self, query, all_counts, s_ngram):
        # adding the column for the query if it's new
        if query not in self.adj_nn_list:
            self.adj_nn_list[query] = set()
            self.adj_nn.add_column(name=query, value=0)
        self.adj_nn_list[query].add(s_ngram)
        total_count, t, counts = all_counts.partition('\t')
        # counts_by_year is a tab-separated list of year<comma>count items.
        for year_count in counts.split('\t'):
            year, t, count = year_count.partition(',')
            if year in self.adj_nn.index:
                self.adj_nn[query, year] = self.adj_nn[query, year] + int(count)

    def _add_to_vrb_nn(self, query, all_counts, s_ngram):
        # adding the column for the query if it's new
        if query not in self.vrb_nn_list:
            self.vrb_nn_list[query] = set()
            self.vrb_nn.add_column(name=query, value=0)
        self.vrb_nn_list[query].add(s_ngram)
        total_count, t, counts = all_counts.partition('\t')
        # counts_by_year is a tab-separated list of year<comma>count items.
        for year_count in counts.split('\t'):
            year, t, count = year_count.partition(',')
            if year in self.vrb_nn.index:
                self.vrb_nn[query, year] = self.vrb_nn[query, year] + int(count)


ne = NgramExcavator(processed_files=[])
ne.read_them_all(target='vrb', arc_type='arcs')
