"""
Create word2vec embeddings from a text corpus
"""
import os
import time
import random
from collections import Counter
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from utils import preprocess, create_lookup_tables

class Word2Vec:
    """ Encapsulates word2vec embeddings """
    def __init__(self):
        self.words = None
        self.embeddings = None
        self.vocab_to_int = None
        self.int_to_vocab = None
        self.int_words = None
        self.drop_count = 0
        self.total_count = 0
        self.vocab = None
        self.vocab_size = 0
        self.embedding_size = 0

        # TensorFlow graph members
        self.graph = tf.Graph()
        self.inputs = None
        self.labels = None
        self.embedding = None
        self.embed = None
        self.softmax_w = None
        self.softmax_b = None
        self.loss = None
        self.cost = None
        self.optimizer = None

        # Embedding validation members
        self.valid_size = None
        self.valid_examples = None
        self.valid_dataset = None

        # We use the cosine distance:
        self.norm = None
        self.normalized_embedding = None
        self.valid_embedding = None
        self.similarity = None

        # Subsampling members
        self.subsample_threshold = None
        self.train_words = None
        self.drop_counter = None

        # Saving and loading
        self.save_folder = None
        self.save_file = None

    def prepare_text(self, text, subsample_threshold=1e-3):
        """ Prepare text for training embeddings """
        self.words = preprocess(text)
        self.vocab_to_int, self.int_to_vocab = create_lookup_tables(self.words)
        self.int_words = [self.vocab_to_int[word] for word in self.words]
        self.subsample(subsample_threshold)

    def subsample(self, subsample_threshold=1e-3):
        """ subsample the text corpus to remove infrequent words """
        print("Sumsampling...")
        self.subsample_threshold = subsample_threshold
        word_counts = Counter(self.int_words)
        self.total_count = len(self.int_words)
        freqs = {word: count / self.total_count for word, count in word_counts.items()}
        p_drop = {word: 1 - np.sqrt(self.subsample_threshold / freqs[word]) for word in word_counts}
        self.train_words = []
        dropped_words = []
        for word in tqdm(self.int_words):
            rnd = random.random()
            if rnd > p_drop[word]:
                self.train_words.append(word)
            else:
                dropped_words.append(word)
        # self.train_words = [word for word in self.int_words if random.random() > p_drop[word]]
        self.drop_counter = Counter(dropped_words)
        self.drop_count = self.total_count - len(self.train_words)
        self.vocab = set(self.train_words)
        self.vocab_size = len(self.vocab)

    def print_sample_words(self, start=0, end=20):
        """ print out a range of sample words """
        print("words from {} to {}".format(start, end))
        print(self.words[start:end])

    def print_top_dropped_words(self, count=20):
        """ print out the counts of the top dropped words """
        print("Top {} dropped words from subsampling".format(count))
        dropped = self.drop_counter.most_common(count)
        for word in dropped:
            print("word: {}, count {}, {:.2f}% of total".format(self.int_to_vocab[word[0]], \
                word[1], word[1]/self.total_count))

    def print_data_stats(self):
        """ prints out some standard statistics on Word2Vec """
        print("--- Before Subsampling ---")
        print("Total words: {}".format(len(self.words)))
        print("Unique words: {}".format(len(set(self.words))))
        print("--- After subsampling ---")
        print("Total words: {}".format(len(self.train_words)))
        print("Unique words: {}".format(len(set(self.train_words))))
        print('Subsampling dropped {} words out of {} words ({:.2f}%)'.format(self.drop_count, \
            self.total_count, 100 * self.drop_count / self.total_count))


    def get_target(self, sentence, idx, window_size=5):
        """ Get a list of words in a window around an index. """

        window = random.randint(1, window_size + 1)
        start = idx - window if (idx - window) > 0 else 0
        stop = idx + window
        target_words = set(sentence[start:idx] + sentence[idx+1:stop+1])

        return list(target_words)

    def get_batches(self, batch_size, window_size=5):
        ''' Create a generator of word batches as a tuple (inputs, targets) '''

        n_batches = len(self.int_words)//batch_size

        # only full batches
        words = self.int_words[:n_batches*batch_size]

        for idx in range(0, len(words), batch_size):
            features, labels = [], []
            batch = words[idx:idx + batch_size]
            for batch_idx, batch_features in enumerate(batch):
                batch_labels = self.get_target(batch, batch_idx, window_size)
                labels.extend(batch_labels)
                features.extend([batch_features] * len(batch_labels))
            yield features, labels

    def make_embedding_graph(self, embedding_size=200, sample_size=100, valid_size=16, \
        valid_window=100):
        """ Make the TensorFlow graph that trains the network to produce the embedding """
        self.embedding_size = embedding_size
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.int32, [None], name='inputs')
            self.labels = tf.placeholder(tf.int32, [None, None], name='labels')

            self.embedding = tf.Variable(tf.random_uniform((self.vocab_size, \
                self.embedding_size), -1, 1))
            self.embed = tf.nn.embedding_lookup(self.embedding, self.inputs)

            self.softmax_w = tf.Variable(tf.truncated_normal((self.vocab_size, \
                self.embedding_size), stddev=0.1))
            self.softmax_b = tf.Variable(tf.zeros(self.vocab_size))

            # Calculate the loss using negative sampling
            self.loss = tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, \
                self.labels, self.embed, sample_size, self.vocab_size)

            self.cost = tf.reduce_mean(self.loss)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

            self.valid_size = valid_size
            self.valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
            self.valid_examples = np.append(self.valid_examples, random.sample(range(1000, 1000 + \
                valid_window), valid_size//2))
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # We use the cosine distance:
            self.norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding), 1, keep_dims=True))
            self.normalized_embedding = self.embedding / self.norm
            self.valid_embedding = tf.nn.embedding_lookup(self.normalized_embedding, \
                self.valid_dataset)
            self.similarity = tf.matmul(self.valid_embedding, tf.transpose(\
                self.normalized_embedding))

    def train_embedding(self, save_folder, save_file, epochs, batch_size, window_size=5):
        """ Train the word embedding """
        with self.graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            iteration = 1
            loss = 0
            sess.run(tf.global_variables_initializer())

            for epoch in range(1, epochs + 1):
                batches = self.get_batches(batch_size, window_size)
                start = time.time()
                for features, labels in batches:

                    feed = {self.inputs: features,
                            self.labels: np.array(labels)[:, None]}
                    train_loss, _ = sess.run([self.cost, self.optimizer], feed_dict=feed)

                    loss += train_loss

                    if iteration % 100 == 0:
                        end = time.time()
                        print("Epoch {}/{}".format(epoch, epochs), \
                            "Iteration: {}".format(iteration), \
                            "Avg. Training loss: {:.4f}".format(loss/100), \
                            "{:.4f} sec/batch".format((end-start)/100))
                        loss = 0
                        start = time.time()

                    if iteration % 1000 == 0:
                        # note that this is expensive (~20% slowdown if computed every 500 steps)
                        sim = self.similarity.eval()
                        for i in range(self.valid_size):
                            valid_word = self.int_to_vocab[self.valid_examples[i]]
                            top_k = 8 # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k+1]
                            log = 'Nearest to %s:' % valid_word
                            for k in range(top_k):
                                close_word = self.int_to_vocab[nearest[k]]
                                log = '%s %s,' % (log, close_word)
                            print(log)

                    iteration += 1
            self.save_folder = save_folder
            self.save_file = save_file
            save_path = os.path.join(save_folder, save_file)
            saver.save(sess, save_path)

    def load_embeddings(self, save_folder):
        """ Load saved embeddings """
        self.save_folder = save_folder
        with self.graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(self.save_folder))
            embed_mat = sess.run(self.embedding)
            return embed_mat
