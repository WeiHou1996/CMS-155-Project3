import numpy as np


class HiddenMarkovModel:
    """
    Class implementation of Hidden Markov Models.
    """

    def __init__(self, A, O):
        """
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.
        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.
        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        """

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1.0 / self.L for _ in range(self.L)]

    def viterbi(self, x):
        """
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        """

        M = len(x)  # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0.0 for _ in range(self.L)] for _ in range(M)]
        seqs = [["" for _ in range(self.L)] for _ in range(M)]

        max_seq = ""

        for i in range(self.L):
            probs[0][i] = self.A_start[i] * self.O[i][x[0]]
            seqs[0][i] = str(i)

        for j in range(1, M):  # each obervation
            xj = x[j]
            for i in range(self.L):  # each state
                tmp = [0.0 for _ in range(self.L)]
                for s in range(self.L):
                    tmp[s] = probs[j - 1][s] * self.A[s][i] * self.O[i][xj]
                probs[j][i] = max(tmp)
                idx = tmp.index(max(tmp))
                seqs[j][i] = seqs[j - 1][idx] + str(i)
        max_prob_idx = (probs[-1]).index(max(probs[-1]))
        max_seq = seqs[-1][max_prob_idx]

        return max_seq

    def forward(self, x, normalize=False):
        """
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            alphas:     Vector of alphas.
                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.
                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        """

        M = len(x)  # Length of sequence.
        alphas = [[0.0 for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            alphas[0][i] = 1
            alphas[1][i] = self.A_start[i] * self.O[i][x[0]]

        for j in range(2, M + 1):  # each obervation
            xj = x[j - 1]
            for i in range(self.L):  # each state
                tmp = 0
                for a in range(self.L):
                    tmp += alphas[j - 1][a] * self.A[a][i]
                alphas[j][i] = self.O[i][xj] * tmp

        if normalize:
            for j in range(M + 1):
                sum_j = sum(alphas[j])
                for i in range(self.L):
                    if sum_j != 0:
                        alphas[j][i] = alphas[j][i] / sum_j

        return alphas

    def backward(self, x, normalize=False):
        """
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.
        Returns:
            betas:      Vector of betas.
                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.
                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        """

        M = len(x)  # Length of sequence.
        betas = [[0.0 for _ in range(self.L)] for _ in range(M + 1)]

        for z in range(self.L):
            betas[-1][z] = 1

        for i in range(M - 1, -1, -1):  # each obervation
            xi = x[i]
            for z in range(self.L):  # each state
                for j in range(self.L):
                    betas[i][z] += betas[i + 1][j] * self.A[z][j] * self.O[j][xi]

        if normalize:
            for j in range(M + 1):
                sum_j = sum(betas[j])
                for z in range(self.L):
                    if sum_j != 0:
                        betas[j][z] = betas[j][z] / sum_j

        return betas

    def unsupervised_learning(self, X, N_iters):
        """
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of variable-length lists, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            N_iters:    The number of iterations to train on.
        """
        L = len(self.A)
        D = len(self.O[0])

        for it in range(N_iters):
            A_new = [[0.0 for _ in range(L)] for _ in range(L)]
            O_new = [[0.0 for _ in range(D)] for _ in range(L)]
            for xi in range(len(X)):
                x = X[xi]
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)
                sum_a = [
                    sum(
                        sum(
                            alphas[j - 1][l]
                            * self.O[k][x[j]]
                            * self.A[l][k]
                            * betas[j][k]
                            for l in range(L)
                        )
                        for k in range(L)
                    )
                    for j in range(1, len(x))
                ]
                sum_o = [
                    sum(alphas[j][l] * betas[j][l] for l in range(L))
                    for j in range(len(x))
                ]
                for a in range(L):
                    for j, xj in enumerate(x):
                        if sum_o[j] != 0:
                            O_new[a][xj] += alphas[j][a] * betas[j][a] / sum_o[j]
                        if j != 0:
                            for b in range(L):
                                xj1 = x[j]
                                if sum_a[j - 1] != 0:
                                    A_new[a][b] += (
                                        alphas[j - 1][a]
                                        * self.O[b][xj1]
                                        * self.A[a][b]
                                        * betas[j][b]
                                        / sum_a[j - 1]
                                    )

            for i in range(len(O_new)):
                norm = sum(O_new[i])
                for j in range(len(O_new[i])):
                    if norm != 0:
                        O_new[i][j] /= norm

            for i in range(len(A_new)):
                norm = sum(A_new[i])
                for j in range(len(A_new[i])):
                    if norm != 0:
                        A_new[i][j] /= norm

            self.O = O_new
            self.A = A_new

    def generate_emission(self, M, seed=None):
        """
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        """

        emission = []
        states = []

        rng = np.random.default_rng(seed=seed)
        rints = rng.random(size=(M, 2))

        y = int((rints[0, 0]) // (1 / self.L))

        for m in range(M):
            states.append(y)

            x_bin = np.zeros((self.D - 1,))
            xo = self.O[y]
            for i in range(self.D - 1):
                x_bin[i] = xo[i] + x_bin[i - 1]
            xm = rints[m, 1]
            x = len(np.where(xm > x_bin)[0])
            emission.append(x)

            if m != M - 1:
                y_bin = np.zeros((self.L - 1,))
                yo = self.A[y]
                for i in range(self.L - 1):
                    y_bin[i] = yo[i] + y_bin[i - 1]
                ym = rints[m + 1, 0]
                y = len(np.where(ym > y_bin)[0])

        return emission, states


def unsupervised_HMM(X, n_states, N_iters, seed=None):
    """
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.
    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.
        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
        rng:        The random number generator for reproducible result.
                    Default to RandomState(1).
    """
    # Initialize random number generator
    rng = np.random.default_rng(seed=seed)

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[rng.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[rng.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM


def sample_sentence(hmm, obs_map_r, syllable, n_words=100, seed=None):
    # Get reverse map.

    # Sample and convert sentence.
    emission, states = hmm.generate_emission(n_words, seed=seed)
    sentence = [obs_map_r[i] for i in emission]

    syl_count = 0
    for i, s in enumerate(sentence):
        if i != (len(s) - 1):
            syl_count += int(syllable[s][0])
        else:
            syl_count += int(syllable[s][1])

    print(syl_count)

    return " ".join(sentence).capitalize()
