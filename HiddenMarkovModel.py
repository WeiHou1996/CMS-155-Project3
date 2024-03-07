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
        alphas = [[0. for _ in range(self.L)] for _ in range(M+1)]

        # get first set of alphas
        for zdx in range(self.L):
            alphas[0][zdx] = 1;
            alphas[1][zdx] = self.O[zdx][x[0]] * self.A_start[zdx]

        if normalize:
            for idx in range(2):
                thisSum = sum(alphas[idx])
                for jdx in range (self.L):
                    alphas[idx][jdx] = alphas[idx][jdx] / thisSum

        # get subsequent set of alphas
        for idx in range(2,M+1):
            for zdx in range(self.L):
                thisSum = 0
                for jdx in range(self.L):
                    thisSum += alphas[idx-1][jdx] * self.A[jdx][zdx]
                
                alphas[idx][zdx] = self.O[zdx][x[idx-1]] * thisSum
            
            if normalize:
                thisSum = sum(alphas[idx])
                for jdx in range(self.L):
                    alphas[idx][jdx] = alphas[idx][jdx] / thisSum
                    

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

        # handle final row
        for zdx in range(self.L):
            if normalize:
                betas[M][zdx] = 1 / self.L
            else:
                betas[M][zdx] = 1
        
        for idx in range(M,0,-1):
            for zdx in range(self.L):
                thisSum = 0
                for jdx in range(self.L):
                    thisSum += betas[idx][jdx] * self.A[zdx][jdx] * self.O[jdx][x[idx-1]]
                
                betas[idx-1][zdx] = thisSum
            
            # normalize
            if normalize:
                thisSum = sum(betas[idx-1])
                for jdx in range(self.L):
                    betas[idx-1][jdx] = betas[idx-1][jdx] / thisSum
        
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
        # perform N_iters iterations
        for ndx in range(N_iters):
            
            # print iteration
            print("Iteration: ",ndx+1)

            # update alpha and beta
            alphas = []
            betas = []
            for ldx in range(len(X)):
                alphas.append(self.forward(X[ldx],normalize=True))
                betas.append(self.backward(X[ldx],normalize=True))

            # make a copy of A
            thisA = []
            for idx in range(self.L):
                thisA.append(self.A[idx].copy())
            
            # make list of dInSum
            dInSumList = []
            for ldx in range(len(X)):
                thisList = []
                for idx in range(len(X[ldx])):
                    dInSum = 0
                    for pdx in range(self.L):
                        dInSum += alphas[ldx][idx][pdx] * betas[ldx][idx][pdx]
                    thisList.append(dInSum)
                dInSumList.append(thisList.copy())
            
            # make list of aInSum
            aInSumList = []
            for ldx in range(len(X)):
                thisList = []
                for idx in range(len(X[ldx])-1):
                    aInSum = 0
                    for pdx in range(self.L):
                        for odx in range(self.L):
                            aInSum += alphas[ldx][idx][pdx] * thisA[pdx][odx] * self.O[odx][X[ldx][idx+1]] * betas[ldx][idx+1][odx]
                    thisList.append(aInSum)
                aInSumList.append(thisList.copy())
            
            # make list of alpha * beta / dIn
            abList = []
            for ldx in range(len(X)):
                listL = []
                for idx in range(len(X[ldx])):
                    listI = []
                    for adx in range(self.L):
                        listI.append(alphas[ldx][idx][adx] * betas[ldx][idx][adx] / dInSumList[ldx][idx])
                    listL.append(listI.copy())
                abList.append(listL)
            
            # sum ab over idx
            abSumList = []
            for ldx in range(len(X)):
                thisList = []
                for adx in range(self.L):
                    thisSum = 0
                    for idx in range(len(X[ldx])):
                        thisSum += abList[ldx][idx][adx]
                    thisList.append(thisSum)
                abSumList.append(thisList.copy())
            
            # update A
            for adx in range(self.L):
                for bdx in range(self.L):
                    
                    # sum entries for A
                    aSum = 0

                    # iterate through sequences
                    for ldx in range(len(X)):
                        
                        # compute numerator p(y2=b,y1=a|x)
                        for idx in range(0,len(X[ldx])-1):
                            thisProb = alphas[ldx][idx][adx] * thisA[adx][bdx] * self.O[bdx][X[ldx][idx+1]] * betas[ldx][idx+1][bdx]
                            aSum += thisProb / aInSumList[ldx][idx]
                    
                    # update A
                    self.A[adx][bdx] = aSum / abSumList[ldx][adx]

            # update O
            for wdx in range(self.D):
                for zdx in range(self.L):
                    
                    # sum entries for O
                    nSum = 0

                    # iterate through sequences
                    for ldx in range(len(X)):
                        
                        # iterate through X
                        for idx in range(len(X[ldx])):
                            if X[ldx][idx] == wdx:
                                nSum += abList[ldx][idx][zdx]
                    
                    self.O[zdx][wdx] = nSum / abSumList[ldx][zdx]
            
            # normalize A
            for idx in range(self.L):
                thisSum = sum(self.A[idx])
                for jdx in range(self.L):
                    self.A[idx][jdx] /= thisSum

            # normalize O
            for idx in range(self.L):
                thisSum = sum(self.O[idx])
                for jdx in range(self.D):
                    self.O[idx][jdx] /= thisSum
                    
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
    
    def generate_emission_r(self, M, lastObs, rng=None):
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

        emission = [np.nan for _ in range(M)]
        states = [np.nan for _ in range(M)]

        # get array of random numbs
        randArr = rng.random((M,2))

        # get emission
        emission[-1] = lastObs

        # assume a uniform prior
        py = np.ones(self.L) / self.L

        # get p(y|x)
        pyGx = []
        for idx in range(self.L):
            pyGx.append(self.O[idx][lastObs] * py[idx])
        pyGx_sum1 = np.cumsum(pyGx)
        pyGx_sum = np.zeros(self.L+1)
        pyGx_sum[1:] = pyGx_sum1 / np.sum(pyGx)
        
        # get states
        for ldx in range(self.L):
            if randArr[-1,0] > pyGx_sum[ldx] and randArr[-1,0] <= pyGx_sum[ldx+1]:
                states[-1] = ldx

        for mdx in range(M-1,0,-1):
            
            # get probability of y given next y
            pyGy = []
            for idx in range(self.L):
                pyGy.append(self.A[idx][states[mdx]])
            pyGy_sum1 = np.cumsum(pyGy)
            pyGy_sum = np.zeros(self.L+1)
            pyGy_sum[1:] = pyGy_sum1 / np.sum(pyGy)

            # get states
            for ldx in range(self.L):
                if randArr[mdx-1,0] > pyGy_sum[ldx] and randArr[mdx-1,0] <= pyGy_sum[ldx+1]:
                    states[mdx-1] = ldx
            
            # get probabiliy of x given y
            pxGy = self.O[states[mdx-1]]
            pxGy_sum1 = np.cumsum(pxGy)
            pxGy_sum = np.zeros(self.D+1)
            pxGy_sum[1:] = pxGy_sum1

            # get observation
            for ddx in range(self.D):
                if randArr[mdx-1,1] > pxGy_sum[ddx] and randArr[mdx-1,1] <= pxGy_sum[ddx+1]:
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