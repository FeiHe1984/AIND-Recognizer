import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from numpy import asarray


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_bic_score = float("inf")
        best_model = None

        for n_component in range(self.min_n_components, self.max_n_components+1):
            try:
                model = self.base_model(n_component)
                # Get the logL (the likelihood of the fitted model)
                logL = model.score(self.X, self.lengths)
                # Get the p (number of parameters)
                p = n_component ** 2 + 2 * n_component * model.n_features - 1
                # Caculate BIC score
                bic_score = (-2 * logL) + (p * np.log(self.X.shape[0]))

                if bic_score < best_bic_score:
                    best_bic_score = bic_score
                    best_model = model

            except:
                pass

        return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_dic_score = float("-inf")
        best_model = None

        # The aginst words
        aginst_words = [word for word in self.words if word != self.this_word]

        for n_component in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_component)
                # Get the logL (the likelihood of the fitted model)
                logL = model.score(self.X, self.lengths)
                sum_other_logL = 0.0

                for word in aginst_words:
                    # X, lengths of this word
                    other_x, other_lengths = self.hwords[word]
                    # sum of other score
                    sum_other_logL += model.score(other_x, other_lengths)

                # Calculate DIC Score
                dic_score = logL - sum_other_logL / (len(self.words)-1)

                if dic_score > best_dic_score:
                    best_dic_score = dic_score
                    best_model = model

            except:
                pass

        return best_model       
                 

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float("-inf")
        best_model = None


        for n_component in range(self.min_n_components, self.max_n_components + 1):
            
            score = 0
            iter_num = 0
            last_model = None

            try:
                # Sequence can't be fold
                if len(self.sequences) <= 1:
                    last_model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    score = model.score(self.X, self.lengths)
                else:
                    # Sequence can be fold
                    # Save 'KFold' as variable
                    num_splits = min(len(self.sequences),3)
                    split_method = KFold(num_splits)

                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # Get the train and test set
                        X_train, length_train = combine_sequences(cv_train_idx, self.sequences)
                        X_test, length_test = combine_sequences(cv_test_idx, self.sequences)
                        # Fit model with train data
                        model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, length_train)
                        score += model.score(X_test,length_test)
                        iter_num += 1
                    # Caculate mean score
                    score = score / iter_num

                # Keep best score and best model
                if best_score < score:
                    best_score = score
                    if last_model is None:
                        last_model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000,
                                       random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    best_model = last_model

            except:
                pass

        return best_model