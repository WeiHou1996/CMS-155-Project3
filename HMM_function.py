import numpy as np
from wordcloud import WordCloud
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import urllib.request
from find_rhyme import find_rhymes


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

            normA = 0
            for i in range(self.L):
                for j in range(self.L):
                    normA += abs(self.A[i][j] - A_new[i][j])

            normA /= self.L * self.L

            normO = 0
            for i in range(self.L):
                for j in range(self.D):
                    normO += abs(self.O[i][j] - O_new[i][j])

            normO /= self.L * self.D

            # print(normO, normA)

            if (normA < 1e-7) and (normO < 1e-7):
                print(f"Stopped after {it} iterations")
                return

            self.O = O_new
            self.A = A_new

    def generate_emission(self, M, endings, obs_map, obs_map_r, syllable, seed=None):
        """
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        """

        lines = []

        for line in range(2):
            syl = False
            while syl is False:
                emission = []
                states = []
                rng = np.random.default_rng()
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
                    if m == 0:
                        if line == 0:
                            num_rhymes = len(find_rhymes(obs_map_r[x], endings))
                            while num_rhymes < 1:
                                xm = rng.random()
                                x = len(np.where(xm > x_bin)[0])
                                num_rhymes = len(find_rhymes(obs_map_r[x], endings))
                            x_start = x
                        elif line == 1:
                            rhymes = find_rhymes(obs_map_r[x_start], endings)
                            # print(rhymes, obs_map_r[x_start])
                            x_bin_new = np.zeros((len(rhymes),))
                            for j, i in enumerate(rhymes):
                                idx = obs_map[i]
                                x_bin_new[j] = xo[idx]
                            x_bin_new /= sum(x_bin_new)
                            x_bin_new_total = np.zeros((len(rhymes) - 1,))
                            for i in range(len(rhymes) - 1):
                                x_bin_new_total[i] = (
                                    x_bin_new[i] + x_bin_new_total[i - 1]
                                )
                            x_idx = len(np.where(xm > x_bin_new_total)[0])
                            for j, i in enumerate(rhymes):
                                if j == x_idx:
                                    x = obs_map[i]
                    emission.append(x)

                    if m != M - 1:
                        y_bin = np.zeros((self.L - 1,))
                        yo = self.A[y]
                        for i in range(self.L - 1):
                            y_bin[i] = yo[i] + y_bin[i - 1]
                        ym = rints[m + 1, 0]
                        y = len(np.where(ym > y_bin)[0])

                syl_count = 0
                sentence = []
                for i, e in enumerate(emission):
                    if syl_count == 10:
                        lines.append(sentence)
                        syl = True
                        break
                    if i != 0:
                        syl_count += int(syllable[obs_map_r[e]][0])
                        sentence.append(e)
                    else:
                        syl_count += int(syllable[obs_map_r[e]][1])
                        sentence.append(e)

        return lines, states


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


def sample_sentence(
    hmm, obs_map, obs_map_r, syllable, endings, n_syl=10, n_lines=14, seed=None
):
    sonnet = []

    for n in range(n_lines):
        # Sample and convert sentence.
        emissions, states = hmm.generate_emission(
            n_syl, endings, obs_map, obs_map_r, syllable, seed=seed
        )

        for emission in emissions:
            sentence = [obs_map_r[i] for i in reversed(emission)]
            sonnet.append(sentence)

    return sonnet


####################
# WORDCLOUD FUNCTIONS
####################


def mask():
    # Parameters.
    r = 128
    d = 2 * r + 1

    # Get points in a circle.
    y, x = np.ogrid[-r : d - r, -r : d - r]
    circle = x**2 + y**2 <= r**2

    # Create mask.
    mask = 255 * np.ones((d, d), dtype=np.uint8)
    mask[circle] = 0

    return mask


def text_to_wordcloud(text, max_words=50, title="", show=True):
    plt.close("all")

    # Generate a wordcloud image.
    wordcloud = WordCloud(
        random_state=0, max_words=max_words, background_color="white", mask=mask()
    ).generate(text)

    # Show the image.
    if show:
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(title, fontsize=24)
        plt.show()

    return wordcloud


def states_to_wordclouds(
    hmm,
    obs_map,
    obs_map_r,
    syllable,
    endings,
    n_syl=10,
    n_lines=14,
    max_words=50,
    show=True,
    seed=None,
):
    # Initialize.
    n_states = len(hmm.A)
    wordclouds = []

    sonnet = []

    for n in range(int(n_lines / 2)):
        # Sample and convert sentence.
        emissions, states = hmm.generate_emission(
            n_syl, endings, obs_map, obs_map_r, syllable, seed=seed
        )

        for emission in emissions:
            for i in reversed(emission):
                sonnet.append(i)

    emission = flatten(sonnet)

    # For each state, get a list of observations that have been emitted
    # from that state.
    obs_count = []
    for i in range(n_states):
        obs_lst = np.array(emission)[np.where(np.array(states) == i)[0]]
        obs_count.append(obs_lst)

    # For each state, convert it into a wordcloud.
    for i in range(n_states):
        obs_lst = obs_count[i]
        sentence = [obs_map_r[j] for j in obs_lst]
        sentence_str = " ".join(sentence)

        try:
            wordclouds.append(
                text_to_wordcloud(
                    sentence_str, max_words=max_words, title="State %d" % i, show=show
                )
            )
        except:
            pass

    return wordclouds


####################
# HMM VISUALIZATION FUNCTIONS
####################


def visualize_sparsities(hmm, O_max_cols=50000, O_vmax=0.1):
    plt.close("all")
    plt.set_cmap("viridis")

    # Visualize sparsity of A.
    plt.imshow(hmm.A, vmax=1.0)
    plt.colorbar()
    plt.title("Sparsity of A matrix")
    plt.show()

    O_vmax = max(max(hmm.O))

    # Visualize parsity of O.
    plt.imshow(np.array(hmm.O)[:O_max_cols], vmax=O_vmax, aspect="auto")
    plt.colorbar()
    plt.title("Sparsity of O matrix")
    plt.show()


####################
# HMM ANIMATION FUNCTIONS
####################


def animate_emission(
    hmm, obs_map, obs_map_r, M=8, height=12, width=12, delay=1, seed=None
):
    # Parameters.
    lim = 1200
    text_x_offset = 40
    text_y_offset = 80
    x_offset = 580
    y_offset = 520
    R = 420
    r = 100
    arrow_size = 20
    arrow_p1 = 0.03
    arrow_p2 = 0.02
    arrow_p3 = 0.06

    # Initialize.
    n_states = len(hmm.A)
    wordclouds = states_to_wordclouds(hmm, obs_map, max_words=20, show=False)

    # Initialize plot.
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.grid("off")
    plt.axis("off")
    ax.set_xlim([0, lim])
    ax.set_ylim([0, lim])

    # Plot each wordcloud.
    for i, wordcloud in enumerate(wordclouds):
        x = x_offset + int(R * np.cos(np.pi * 2 * i / n_states))
        y = y_offset + int(R * np.sin(np.pi * 2 * i / n_states))
        ax.imshow(
            wordcloud.to_array(),
            extent=(x - r, x + r, y - r, y + r),
            aspect="auto",
            zorder=-1,
        )

    # Initialize text.
    text = ax.text(text_x_offset, lim - text_y_offset, "", fontsize=24)

    # Make the arrows.
    zorder_mult = n_states**2 * 100
    arrows = []
    for i in range(n_states):
        row = []
        for j in range(n_states):
            # Arrow coordinates.
            x_i = x_offset + R * np.cos(np.pi * 2 * i / n_states)
            y_i = y_offset + R * np.sin(np.pi * 2 * i / n_states)
            x_j = x_offset + R * np.cos(np.pi * 2 * j / n_states)
            y_j = y_offset + R * np.sin(np.pi * 2 * j / n_states)

            dx = x_j - x_i
            dy = y_j - y_i
            d = np.sqrt(dx**2 + dy**2)

            if i != j:
                arrow = ax.arrow(
                    x_i + (r / d + arrow_p1) * dx + arrow_p2 * dy,
                    y_i + (r / d + arrow_p1) * dy + arrow_p2 * dx,
                    (1 - 2 * r / d - arrow_p3) * dx,
                    (1 - 2 * r / d - arrow_p3) * dy,
                    color=(1 - hmm.A[i][j],) * 3,
                    head_width=arrow_size,
                    head_length=arrow_size,
                    zorder=int(hmm.A[i][j] * zorder_mult),
                )
            else:
                arrow = ax.arrow(
                    x_i,
                    y_i,
                    0,
                    0,
                    color=(1 - hmm.A[i][j],) * 3,
                    head_width=arrow_size,
                    head_length=arrow_size,
                    zorder=int(hmm.A[i][j] * zorder_mult),
                )

            row.append(arrow)
        arrows.append(row)

    emission, states = hmm.generate_emission(M, seed=seed)

    def animate(i):
        if i >= delay:
            i -= delay

            if i == 0:
                arrows[states[0]][states[0]].set_color("red")
            elif i == 1:
                arrows[states[0]][states[0]].set_color(
                    (1 - hmm.A[states[0]][states[0]],) * 3
                )
                arrows[states[i - 1]][states[i]].set_color("red")
            else:
                arrows[states[i - 2]][states[i - 1]].set_color(
                    (1 - hmm.A[states[i - 2]][states[i - 1]],) * 3
                )
                arrows[states[i - 1]][states[i]].set_color("red")

            # Set text.
            text.set_text(
                " ".join([obs_map_r[e] for e in emission][: i + 1]).capitalize()
            )

            return arrows + [text]

    # Animate!
    print("\nAnimating...")
    anim = FuncAnimation(fig, animate, frames=M + delay, interval=1000)

    return anim


class Utility:
    """
    Utility for the problem files.
    """

    def __init__():
        pass

    @staticmethod
    def load_sequence(n):
        """
        Load the file 'sequence_data<n>.txt' for a given n.
        Arguments:
            n:          Sequence index.
        Returns:
            A:          The transition matrix.
            O:          The observation matrix.
            seqs:       Input sequences.
        """
        A = []
        O = []
        seqs = []

        # For each file:
        with urllib.request.urlopen(
            f"https://caltech-cs155.s3.us-east-2.amazonaws.com/sets/set6/data/sequence_data{n}.txt"
        ) as f:
            # Read the parameters.
            L, D = [int(x) for x in f.readline().decode("utf-8").strip().split("\t")]

            # Read the transition matrix.
            for i in range(L):
                A.append(
                    [float(x) for x in f.readline().decode("utf-8").strip().split("\t")]
                )

            # Read the observation matrix.
            for i in range(L):
                O.append(
                    [float(x) for x in f.readline().decode("utf-8").strip().split("\t")]
                )

            # The rest of the file consists of sequences.
            while True:
                seq = f.readline().decode("utf-8").strip()
                if seq == "":
                    break
                seqs.append([int(x) for x in seq])

        return A, O, seqs

    @staticmethod
    def load_ron():
        """
        Loads the file 'ron.txt'.
        Returns:
            moods:      Sequnces of states, i.e. a list of lists.
                        Each sequence represents half a year of data.
            mood_map:   A hash map that maps each state to an integer.
            genres:     Sequences of observations, i.e. a list of lists.
                        Each sequence represents half a year of data.
            genre_map:  A hash map that maps each observation to an integer.
        """
        moods = []
        mood_map = {}
        genres = []
        genre_map = {}
        mood_counter = 0
        genre_counter = 0

        with urllib.request.urlopen(
            "https://caltech-cs155.s3.us-east-2.amazonaws.com/sets/set6/data/ron.txt"
        ) as f:
            mood_seq = []
            genre_seq = []

            while True:
                line = f.readline().decode("utf-8").strip()

                if line == "" or line == "-":
                    # A half year has passed. Add the current sequence to
                    # the list of sequences.
                    moods.append(mood_seq)
                    genres.append(genre_seq)
                    # Start new sequences.
                    mood_seq = []
                    genre_seq = []

                if line == "":
                    break
                elif line == "-":
                    continue

                mood, genre = line.split()

                # Add new moods to the mood state hash map.
                if mood not in mood_map:
                    mood_map[mood] = mood_counter
                    mood_counter += 1

                mood_seq.append(mood_map[mood])

                # Add new genres to the genre observation hash map.
                if genre not in genre_map:
                    genre_map[genre] = genre_counter
                    genre_counter += 1

                # Convert the genre into an integer.
                genre_seq.append(genre_map[genre])

        return moods, mood_map, genres, genre_map

    @staticmethod
    def load_ron_hidden():
        """
        Loads the file 'ron.txt' and hides the states.
        Returns:
            genres:     The observations.
            genre_map:  A hash map that maps each observation to an integer.
        """
        moods, mood_map, genres, genre_map = Utility.load_ron()

        return genres, genre_map


def flatten(lst: any) -> any:
    flattened_list = []
    for item in lst:
        flattened_list.append(item)
    return flattened_list


def state_top_words(hmm, obs_map_r, n_words=10):
    for state, O_row in enumerate(hmm.O):
        O_row = np.array(O_row)
        top_words = np.argpartition(O_row, -n_words)[-n_words:]
        s = []
        for w in top_words:
            s.append(obs_map_r[w])
        print(f"State: {state}")
        print(", ".join(s).capitalize())
