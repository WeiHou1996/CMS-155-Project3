import numpy as np
import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
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
        '''
        #A = [[0.7,0.3],[0.4,0.6]]
        #O = [[0.5,0.4,0.1],[0.1,0.3,0.6]]
        #self.A_start = [0.6,0.4]
        
        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]
        


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''
        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M+1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M+1)]

        for ydx in range(self.L):
            probs[0][ydx] = self.A_start[ydx]
            probs[1][ydx] = self.A_start[ydx] * self.O[ydx][x[0]]
            seqs[1][ydx] = str(ydx)
        
        for xdx in range(1,M): # iterate through x
            for ydx in range(self.L): # iterate through next Y
                p_xGy = self.O[ydx][x[xdx]]
                thisMax = 0
                idxMax = None
                for pdx in range(self.L): # iterate through previous Y
                    p_yGx = probs[xdx][pdx] # previous table values
                    p_yGy = self.A[pdx][ydx] # p of pdx to jdx
                    thisP = p_xGy * p_yGy * p_yGx
                    if thisP  > thisMax:
                        thisMax = thisP
                        idxMax = pdx
                        
                probs[xdx+1][ydx] = thisMax
                if idxMax is None:
                    seqs[xdx+1][ydx] = ""
                else:
                    thisStr = seqs[xdx][idxMax] + str(ydx)
                    seqs[xdx+1][ydx] = thisStr
        
        # find MAP sequence
        thisMax = 0
        idxMax = None
        for ydx in range(self.L):
            thisProbs = probs[M][ydx]
            if thisProbs > thisMax:
                idxMax = ydx
                thisMax = thisProbs
        max_seq = seqs[M][idxMax]
        return max_seq


    def forward(self, x, normalize=False):
        '''
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
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            alphas[1][i] = self.A_start[i] * self.O[i][x[0]]

        for d in range(2, M + 1):

            for curr_state in range(self.L):
                prob = 0
                for prev_state in range(self.L):
                    prob += (self.O[curr_state][x[d-1]] * (alphas[d-1][prev_state] * self.A[prev_state][curr_state]))

                alphas[d][curr_state] = prob

            if normalize:
                denom = np.sum(alphas[d])
                alphas[d] = [alpha/denom for alpha in alphas[d]]


        return alphas


    def backward(self, x, normalize=False):
        '''
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
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for i in range(self.L):
            betas[M][i] = 1

        for d in range(M - 1, -1, -1):

            for curr_state in range(self.L):
                prob = 0
                for next_state in range(self.L):
                    if d == 0:
                        prob += (betas[d+1][next_state] * self.A_start[next_state] * self.O[next_state][x[d]])
                    else:
                        prob += (betas[d+1][next_state] * self.A[curr_state][next_state] * self.O[next_state][x[d]])

                betas[d][curr_state] = prob

            if normalize:
                denom = np.sum(betas[d])
                betas[d] = [beta/denom for beta in betas[d]]

        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.
                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        for adx in range(self.L):
            for bdx in range(self.L):
                cN = 0
                cD = 0
                for ldx in range(len(X)):
                    for jdx in range(1,len(X[ldx])):

                        # handle data in current time period
                        if Y[ldx][jdx-1] == bdx:
                            cD += 1
                            if Y[ldx][jdx] == adx:
                                cN += 1
                                                    
                # store values
                self.A[bdx][adx] = cN / cD

        # verify A
        for adx in range(self.L):
            thisErr = abs(sum(self.A[adx])-1)
            if thisErr > 1e-12:
                print("Error in Evolution Matrix: ",thisErr)

        # Calculate each element of O using the M-step formulas.
        for wdx in range(self.D):
            for zdx in range(self.L):
                cN = 0
                cD = 0
                for ldx in range(len(X)):
                    for jdx in range(len(X[ldx])):
                        if Y[ldx][jdx] == zdx:
                            cD += 1
                            if X[ldx][jdx] == wdx:
                                cN += 1
                self.O[zdx][wdx] = cN / cD
        
        # verify O
        for zdx in range(self.L):
            thisErr = abs(sum(self.O[zdx])-1)
            if thisErr > 1e-12:
                print("Error in Observation Matrix")
        
        pass


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.
        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of variable-length lists, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.
            N_iters:    The number of iterations to train on.
        '''
        oTol = 1e-3
        aTol = 1e-3
        for i in range(N_iters):
            A_numer = np.zeros((self.L, self.L))
            A_denom = np.zeros((self.L, self.L))
            O_numer = np.zeros((self.L, self.D))
            O_denom = np.zeros((self.L, self.D))

            for x in X:
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)
                M = len(x)

                for d in range(1, M + 1):
                    prob_OAd = np.array([alphas[d][curr_state] * betas[d][curr_state] for curr_state in range(self.L)])
                    prob_OAd /= np.sum(prob_OAd)

                    for curr_state in range(self.L):
                        O_numer[curr_state][x[d-1]] += prob_OAd[curr_state]
                        O_denom[curr_state] += prob_OAd[curr_state]
                        if d != M:
                            A_denom[curr_state] += prob_OAd[curr_state]

                for d in range(1, M):
                    prob_An = np.array([[alphas[d][curr_state] \
                                    * self.O[next_state][x[d]] \
                                    * self.A[curr_state][next_state] \
                                    * betas[d+1][next_state] \
                                    for next_state in range(self.L)] \
                                    for curr_state in range(self.L)])
                    prob_An /= np.sum(prob_An)

                    for curr_state in range(self.L):
                        for next_state in range(self.L):
                            A_numer[curr_state][next_state] += prob_An[curr_state][next_state]
            thisA = A_numer / A_denom
            thisO = O_numer / O_denom
            diffA = np.linalg.norm(thisA-self.A)
            diffO = np.linalg.norm(thisO-self.O)
            if diffO < oTol and diffA < aTol:
                print("HMM Training terminates after {:d} iterations".format(i+1))
                break
            else:
                print("HMM Training iteration {:d}".format(i+1))
                print("A error {:3e}, O error {:3e}".format(diffA,diffO))
                self.A = A_numer / A_denom
                self.O = O_numer / O_denom
                    
        pass


    def generate_emission(self, M, rng=None, e0 = None, s0 = None):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        '''

        # (Re-)Initialize random number generator
        if rng is None:
            rng = np.random.default_rng()

        if e0 is None and s0 is None:
            emission = []
            states = []
            r0 = range(M)
        else:
            emission = [e0]
            states = [s0]
            M += 1
            r0 = range(1,M)

        # get array of random numbs
        randArr = rng.random((M,2))
        
        for mdx in r0:
            
            # get probability of y given y
            if mdx == 0:
                pyy = np.ones(self.L) / self.L
            else:
                pyy = self.A[states[mdx-1]]
            pyy_sum1 = np.cumsum(pyy)
            pyy_sum = np.zeros(self.L+1)
            pyy_sum[1:] = pyy_sum1

            # get states
            for ldx in range(self.L):
                if randArr[mdx,0] > pyy_sum[ldx] and randArr[mdx,0] <= pyy_sum[ldx+1]:
                    states.append(ldx)
            
            # get probabiliy of x given y
            pxy = self.O[states[mdx]]
            pxy_sum1 = np.cumsum(pxy)
            pxy_sum = np.zeros(self.D+1)
            pxy_sum[1:] = pxy_sum1

            # get observation
            for ddx in range(self.D):
                if randArr[mdx,1] > pxy_sum[ddx] and randArr[mdx,1] <= pxy_sum[ddx+1]:
                    emission.append(ddx)

        return emission, states
    
    def generate_emission_r(self, M, lastObs, lastState = None, rng=None):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        '''

        # (Re-)Initialize random number generator
        if rng is None:
            rng = np.random.default_rng()

        # intialize list
        emission = [np.nan for _ in range(M+1)]
        states = [np.nan for _ in range(M+1)]

        # assume a uniform prior
        py = np.ones(self.L) / self.L

        # generate emission
        emission[-1] = lastObs
        if lastState is None:
            # get p(y|x)
            pyGx = []
            for idx in range(self.L):
                pyGx.append(self.O[idx][lastObs] * py[idx])
            pyGx_sum1 = np.cumsum(pyGx)
            pyGx_sum = np.zeros(self.L+1)
            pyGx_sum[1:] = pyGx_sum1 / np.sum(pyGx)
            
            # get states
            thisR = rng.random(1)
            for ldx in range(self.L):
                if thisR > pyGx_sum[ldx] and thisR <= pyGx_sum[ldx+1]:
                    states[-1] = ldx
        else:
            states[-1] = lastState
        
        for mdx in range(M,0,-1):
            
            # get probability of y given next y
            pyGy = []
            for idx in range(self.L):
                pyGy.append(self.A[idx][states[mdx]])
            pyGy_sum1 = np.cumsum(pyGy)
            pyGy_sum = np.zeros(self.L+1)
            pyGy_sum[1:] = pyGy_sum1 / np.sum(pyGy)

            # get states
            thisR = rng.random(1)
            for ldx in range(self.L):
                if thisR > pyGy_sum[ldx] and thisR <= pyGy_sum[ldx+1]:
                    states[mdx-1] = ldx
            
            # get probabiliy of x given y
            pxGy = self.O[states[mdx-1]]
            pxGy_sum1 = np.cumsum(pxGy)
            pxGy_sum = np.zeros(self.D+1)
            pxGy_sum[1:] = pxGy_sum1

            # get observation
            thisR = rng.random(1)
            for ddx in range(self.D):
                if thisR > pxGy_sum[ddx] and thisR <= pxGy_sum[ddx+1]:
                    emission[mdx-1] = ddx

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob