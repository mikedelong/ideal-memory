# adapted from
# https://github.com/tejank10/Spam-or-Ham/blob/master/spam_ham.ipynb
from math import log

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def process_message(message, lower_case=True, stem=True, stop_words=True, gram=2):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words


class SpamClassifier(object):
    def __init__(self, train_data, method='tf-idf'):
        self.mails, self.labels = train_data['message'], train_data['label']
        self.method = method
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.prob_spam_mail = 0
        self.prob_ham_mail = 0
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        self.total_mails = 0
        self.spam_mails = 0
        self.ham_mails = 0
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0

    def train(self):
        self.calc_tf_and_idf()
        if self.method == 'tf-idf':
            self.calc_tf_idf()
        else:
            self.calc_prob()

    def calc_prob(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word] + 1) / (self.spam_words + len(list(self.tf_spam.keys())))
        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word] + 1) / (self.ham_words + len(list(self.tf_ham.keys())))
        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails

    def calc_tf_and_idf(self):
        message_count = self.mails.shape[0]
        self.spam_mails, self.ham_mails = self.labels.value_counts()[1], self.labels.value_counts()[0]
        self.total_mails = self.spam_mails + self.ham_mails
        self.spam_words = 0
        self.ham_words = 0
        self.tf_spam = dict()
        self.tf_ham = dict()
        self.idf_spam = dict()
        self.idf_ham = dict()
        for i in range(message_count):
            message_processed = process_message(self.mails[i])
            count = list()  # To keep track of whether the word has ocured in the message or not.
            # For IDF
            for word in message_processed:
                if self.labels[i]:
                    self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
                    self.spam_words += 1
                else:
                    self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
                    self.ham_words += 1
                if word not in count:
                    count += [word]
            for word in count:
                if self.labels[i]:
                    self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
                else:
                    self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

    def calc_tf_idf(self):
        self.prob_spam = dict()
        self.prob_ham = dict()
        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0
        for word in self.tf_spam:
            self.prob_spam[word] = (self.tf_spam[word]) * log(
                (self.spam_mails + self.ham_mails) / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
            self.sum_tf_idf_spam += self.prob_spam[word]
        for word in self.tf_spam:
            self.prob_spam[word] = (self.prob_spam[word] + 1) / (
                    self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))

        for word in self.tf_ham:
            self.prob_ham[word] = (self.tf_ham[word]) * log(
                (self.spam_mails + self.ham_mails) / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
            self.sum_tf_idf_ham += self.prob_ham[word]
        for word in self.tf_ham:
            self.prob_ham[word] = (self.prob_ham[word] + 1) / (self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))

        self.prob_spam_mail, self.prob_ham_mail = self.spam_mails / self.total_mails, self.ham_mails / self.total_mails

    def classify(self, processed_message):
        p_spam, p_ham = 0, 0
        for word in processed_message:
            if word in self.prob_spam:
                p_spam += log(self.prob_spam[word])
            else:
                if self.method == 'tf-idf':
                    p_spam -= log(self.sum_tf_idf_spam + len(list(self.prob_spam.keys())))
                else:
                    p_spam -= log(self.spam_words + len(list(self.prob_spam.keys())))
            if word in self.prob_ham:
                p_ham += log(self.prob_ham[word])
            else:
                if self.method == 'tf-idf':
                    p_ham -= log(self.sum_tf_idf_ham + len(list(self.prob_ham.keys())))
                else:
                    p_ham -= log(self.ham_words + len(list(self.prob_ham.keys())))
            p_spam += log(self.prob_spam_mail)
            p_ham += log(self.prob_ham_mail)
        return p_spam >= p_ham

    def predict(self, test_data):
        result = dict()
        for (i, message) in enumerate(test_data):
            processed_message = process_message(message)
            result[i] = int(self.classify(processed_message))
        return result
