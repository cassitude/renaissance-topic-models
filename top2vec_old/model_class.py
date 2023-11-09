from top2vec import Top2Vec
import os, time, multiprocessing
from os import listdir
import numpy as np 
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt 
from typing import List, Dict





# -----------------------------------------------------
# ------------- Model Object ---------------------------
# ----------------------------------------------------

class Top2VecModel(Top2Vec): 
    def __init__(self, tp_start: int, tp_end: int, 
                 path_metadata: str, path_files: str,
                 path_models: str, n_threads: int): 

        self.tp_start = tp_start 
        self.tp_end = tp_end 
        self.path_metadata = path_metadata 
        self.path_files = path_files
        self.n_threads = n_threads 

        print('Loading Files (' + str(tp_start) + '-' + str(tp_end) + ')')
        self._filenames() # -> self.filenames 
        self._texts() # -> self.texts, self.n_docs

        print('Training Model (' + str(tp_start) + '-' + str(tp_end) + ')')
        self._model_train() # -> self.model 
        self._model_save() 

    def _filenames(self):
        '''Get filenames associated with a particular timeperiod.'''

        # read metadata
        metadata_original = pd.read_csv(self.path_metadata) 
        numeric_filter = metadata_original['Date'].apply(lambda x: (x[0:4].isnumeric()))
        metadata = metadata_original.loc[numeric_filter] 

        # filter for files in timeperiod 
        metadata.loc[:, 'Date'] = metadata.Date.apply(lambda x: int(x[0:4]))
        metadata = metadata.loc[(metadata.Date >= self.tp_start) & (metadata.Date <= self.tp_end)]
        filenames = metadata.TCP.apply(lambda x: str(x) + '.headed.txt').tolist() 

        # filter for existing files
        filenames_exist = [] 
        for filename in filenames: 
            if os.path.exists(self.path_files+filename): 
                filenames_exist.append(filename)

        print('# of docs after filter:' + str(len(filenames_exist)))
        self.filenames = filenames_exist

    def _texts(self): 
        '''Extract text given filenames.'''

        self.texts = list() 
        for i in range(len(self.filenames)): 
            with open(self.path_files + self.filenames[i], 'r') as file: 
                self.texts.append(file.read())
        self.n_docs = len(self.texts)

    def _model_train(self):  
        '''Train Top2Vec model.'''

        self.train_speed = 'deep-learn' # check out on docs
        self.model = super().__init__(
            documents=self.texts, 
            speed=self.train_speed, 
            workers=self.n_threads
        ) 
    
    def _model_save(self): 
        '''Save Top2Vec model.'''

        name = self.path_models + str(self.tp_start) + '_' + str(self.tp_end) + '_' + self.train_speed
        self.model.save(name)





# ----------------------------------------------------
# ---------- Collection of Models --------------------
# ---------------------------------------------------

class LanguageOverTime: 
    def __init__(self, path_metadata: str, 
                 path_files='../../alltexts/',
                 path_models='models/'): 

        # initialize dictionary of models 
        self.models = dict() 
        self.path_metadata = path_metadata
        self.path_files = path_files 
        self.path_models = path_models

        # processing setup
        pool = multiprocessing.Pool() 
        n_avail_threads = pool._processes 
        pct_threads2use = 1 
        self.n_threads = int(n_avail_threads*pct_threads2use) 
        print('# of threads to use:', self.n_threads)

    def _generate_model(self, tp_start: int, tp_end: int):
        '''Create Top2Vec model trained on files from a particular timeperiod.''' 

        # keys based on start and end year
        if not tp_start in self.models: 
            self.models[tp_start] = dict() 

        if not tp_end in self.models[tp_start]: 

            # load saved models
            model_name = self.path_models + str(tp_start) + '_' + str(tp_end) + '_' + 'deep-learn'
            if os.path.exists(model_name): 
                print('Loading Model ('+ str(tp_start) + '-' + str(tp_end) + ')')
                self.models[tp_start][tp_end] = Top2Vec.load(model_name)

            # create model
            else: 
                print('Generating Model (' + str(tp_start) + '-' + str(tp_end) + ')')
                self.models[tp_start][tp_end] = Top2VecModel(
                    tp_start, tp_end,
                    self.path_metadata, self.path_files, 
                    self.path_models, self.n_threads 
                )
            

    def txt_similar_words(self, 
                          tp_start: int, tp_end: int, 
                          search_words: List[str], n: int): 
        '''Get "top-n" similar words for each word in a list.'''

        self._generate_model(tp_start, tp_end)

        # output file 
        filename = self._summaries_filename('similar_words', tp_start, tp_end)
        file = open(filename, 'w') 

        # write similar words and scores 
        for search_word in search_words: 
            file.write('Similar words to: '+ search_word + '\n') 
            close_words, scores = self.models[tp_start][tp_end].model.similar_words(
                keywords=[search_word], 
                keywords_neg=[], 
                num_words=n
            )
            for close_word, score in zip(close_words, scores): 
                file.write(close_word + ' ' + str(score) + '\n') 
            file.write('\n')

    def txt_model_summary(self, 
                          tp_start: int, tp_end: int,
                          n_words_per_topic: int, 
                          search_word: str, search_topic: int, max_docs: int, 
                          n_words_per_word: int,
                          keywords: List[str]):

        # create model if non-existent 
        self._generate_model(tp_start, tp_end) 

        # initialize file 
        filename = self._summaries_filename('model_summary', tp_start, tp_end)
        file = open(filename, 'w')
        file.write('Summary of the ' + str(tp_start) + '-' + str(tp_end) + ' Top2Vec model.\n\n')
        file.write('-'*30 + '\n') 

        # topic data 
        n_topics = self.models[tp_start][tp_end].model.get_num_topics() 
        topic_sizes = self.models[tp_start][tp_end].model.get_topic_sizes() 
        topics = self.models[tp_start][tp_end].model.get_topics() 

        # ------------------------
        # Documents/Topics Metrics 
        # ------------------------
        file.write('Documents/Topics Metrics\n') 
        file.write('-'*30 + '\n') 
        file.write('\n')

        # total topics 
        file.write('# of topics from documents: ' + str(n_topics) + '\n')
        file.write('\n')

        # docs per topic 
        file.write('# of docs assigned to each topic: \n') 
        for i in range(n_topics): 
            file.write('Topic ' + str(i) + ': ' + str(topic_sizes[0][i]) + '\n')
        file.write('\n')

        # top words per topic 
        file.write('Top words assigned to each topic: \n')
        for i in range(n_topics): 
            file.write('Topic ' + str(i) + ': ' + str(n_words_per_topic) + ' words to define the topic are:\n')
            for j in range(n_words_per_topic): 
                file.write('\t' + topics[0][i][j] + ' ' + str(topics[1][i][j]) + '\n')
        file.write('\n')

        # -------
        # Queries 
        # -------
        file.write('-'*30 + '\n')
        file.write('Queries\n') 
        file.write('-'*30 + '\n')
        file.write('\n')

        # word to topics 
        file.write('Word: ' + search_word + '\n') 
        word2topics = self.models[tp_start][tp_end].model.search_topics(keywords=[search_word], num_topics=n_topics)
        for i in range(n_topics): 
            file.write('\tTopic ' + str(word2topics[3][i]) + ': ' + str(word2topics[2][i]) + ' similarity\n')
        file.write('\n')

        # topic to docs 
        file.write('Topic:' + str(search_topic) + '\n')
        docs, doc_scores, doc_ids = self.models[tp_start][tp_end].model.search_documents_by_topic(
            topic_num=search_topic, 
            num_docs=max_docs
        )
        for i in range(max_docs): 
            file.write('\tDocument ' + str(doc_ids[i]) + ': ' + str(doc_scores[i]) + ' similarity\n')
            # file.write('\"' + docs[i][:500] + '...\"')
        file.write('\n')

        # word to docs 
        file.write('Word: ' + search_word + '\n')
        docs, doc_scores, doc_ids = self.models[tp_start][tp_end].model.search_documents_by_keywords(
            keywords=[search_word],
            num_docs=max_docs
        )
        for i in range(max_docs): 
            file.write('\tDocument: ' + str(doc_ids[i]) + ': ' + str(doc_scores[i]) + ' similarity\n')
            # file.write('\"' + docs[i][:500] + '...\"')
        file.write('\n')

        # word to words 
        file.write('Word: ' + search_word + '\n')
        words, scores = self.models[tp_start][tp_end].model.similar_words(
            keywords=[search_word], 
            keyword_neg=[], 
            num_words=n_words_per_word
        )
        for w, s in zip(words, scores): 
            file.write('\tWord: ' + w + ': ' + str(s) + ' similarity\n') 
        file.write('\n')

        # keywords to topics
        keywords_topic_query = self.models[tp_start][tp_end].model.search_topics(keywords, num_topics=3) 
        file.write('Keywords: ' + str(keywords) + '\n')
        for i in range(3): 
            file.write('\tTopic: ' + str(keywords_topic_query[3][i]) + ' (' + str(keywords_topic_query[0][i][:5]) + ')' + ': ' + str(keywords_topic_query[2][i]) + ' similarity\n')

    def _summaries_filename(self, prefix: str, 
                            tp_start: int, tp_end: int): 
        return 'summaries/' + prefix + '_' + str(tp_start) + '_' + str(tp_end) + '.txt'

# other possibilities:
# -generate word clouds, collapse topics into smaller set, search for topic or document by a document/sentence, search using multiple
# -keywords, adjust parameters on UMAP dimension reduction and HDB scan for clusteirng