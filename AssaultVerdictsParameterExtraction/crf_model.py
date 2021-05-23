import nltk
import sklearn_crfsuite
import eli5

nltk.corpus.conll2002.fileids()
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
print(train_sents[0])

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

print(sent2features(train_sents[0])[0])
print(X_train[0][1])

# import torch
#
#
# def neg_log_likelihood(self, rolls, states):
#     """Compute neg log-likelihood for a given sequence.
#
#     Input:
#         rolls: numpy array, dim [1, n_rolls]. Integer 0-5 showing value on dice.
#         states: numpy array, dim [1, n_rolls]. Integer 0, 1. 0 if dice is fair.
#     """
#     loglikelihoods = self._data_to_likelihood(rolls)
#     states = torch.LongTensor(states)
#
#     sequence_loglik = self._compute_likelihood_numerator(loglikelihoods, states)
#     denominator = self._compute_likelihood_denominator(loglikelihoods)
#     return denominator - sequence_loglik
#
#
# def _data_to_likelihood(self, rolls):
#     """Converts a numpy array of rolls (integers) to log-likelihood.
#     self.loglikeligood is a matrix of 6 x 2 in our case.
#     Input is one [1, n_rolls]
#     """
#     log_likelihoods = self.loglikelihood[rolls]
#     return Variable(torch.FloatTensor(log_likelihoods), requires_grad=False)
#
#
# def _compute_likelihood_numerator(self, loglikelihoods, states):
#     """Computes numerator of likelihood function for a given sequence.
#
#     We'll iterate over the sequence of states and compute the sum
#     of the relevant transition cost with the log likelihood of the observed
#     roll.
#     Input:
#         loglikelihoods: torch Variable. Matrix of n_obs x n_states.
#                         i,j entry is loglikelihood of observing roll i given state j
#         states: sequence of labels
#     Output:
#         score: torch Variable. Score of assignment.
#     """
#     prev_state = self.n_states
#     score = Variable(torch.Tensor([0]))
#     for index, state in enumerate(states):
#         score += self.transition[state, prev_state] + loglikelihoods[index, state]
#         prev_state = state
#     return score
#
#
# def _compute_likelihood_denominator(self, loglikelihoods):
#     """Implements the forward pass of the forward-backward algorithm.
#
#     We loop over all possible states efficiently using the recursive
#     relationship: alpha_t(j) = \sum_i alpha_{t-1}(i) * L(x_t | y_t) * C(y_t | y{t-1} = i)
#     Input:
#         loglikelihoods: torch Variable. Same input as _compute_likelihood_numerator.
#                         This algorithm efficiently loops over all possible state sequences
#                         so no other imput is needed.
#     Output:
#         torch Variable.
#     """
#
#     # Stores the current value of alpha at timestep t
#     prev_alpha = self.transition[:, self.n_states] + loglikelihoods[0].view(1, -1)
#
#     for roll in loglikelihoods[1:]:
#         alpha_t = []
#
#         # Loop over all possible states
#         for next_state in range(self.n_states):
#
#             # Compute all possible costs of transitioning to next_state
#             feature_function = self.transition[next_state,:self.n_states].view(1, -1) + \
#                                roll[next_state].view(1, -1).expand(1, self.n_states)
#
#             alpha_t_next_state = prev_alpha + feature_function
#             alpha_t.append(self.log_sum_exp(alpha_t_next_state))
#         prev_alpha = torch.cat(alpha_t).view(1, -1)
#     return self.log_sum_exp(prev_alpha)
#
#
# def _viterbi_algorithm(self, loglikelihoods):
#     """Implements Viterbi algorithm for finding most likely sequence of labels.
#
#     Very similar to _compute_likelihood_denominator but now we take the maximum
#     over the previous states as opposed to the sum.
#     Input:
#         loglikelihoods: torch Variable. Same input as _compute_likelihood_denominator.
#     Output:
#         tuple. First entry is the most likely sequence of labels. Second is
#                the loglikelihood of this sequence.
#     """
#
#     argmaxes = []
#
#     # prev_delta will store the current score of the sequence for each state
#     prev_delta = self.transition[:, self.n_states].view(1, -1) + \
#                  loglikelihoods[0].view(1, -1)
#
#     for roll in loglikelihoods[1:]:
#         local_argmaxes = []
#         next_delta = []
#         for next_state in range(self.n_states):
#             feature_function = self.transition[next_state,:self.n_states].view(1, -1) + \
#                                roll.view(1, -1) + \
#                                prev_delta
#             most_likely_state = self.argmax(feature_function)
#             score = feature_function[0][most_likely_state]
#             next_delta.append(score)
#             local_argmaxes.append(most_likely_state)
#         prev_delta = torch.cat(next_delta).view(1, -1)
#         argmaxes.append(local_argmaxes)
#
#     final_state = self.argmax(prev_delta)
#     final_score = prev_delta[0][final_state]
#     path_list = [final_state]
#
#     # Backtrack through the argmaxes to find most likely state
#     for states in reversed(argmaxes):
#         final_state = states[final_state]
#         path_list.append(final_state)
#
#     return np.array(path_list), final_score
#
#
#
# if __name__ == "__main__":
#     _viterbi_algorithm()