# Author: Kyle Kastner
# License: BSD 3-Clause
# built out of code from Gabriele Barbieri
# https://github.com/gabrielebarbieri/markovchain
# All mistakes my own
from collections import defaultdict
import copy
import nltk
import pandas as pd
import os
import numpy as np

NLTK_PACKAGES = ['punkt', 'word2vec_sample', 'cmudict']
START_SYMBOL = '<s>'
END_SYMBOL = '</s>'

_WORD2VEC = None
_CMU_DICT = None

BLACKLIST = ['"', '``', "''"]

DYLAN_DATA = 'data/Dylan'
DYLAN_POPULARITY = 'data/dylan_popularity.csv'

def get_dylan_most_popular_songs(n=5):
    df = pd.read_csv(DYLAN_POPULARITY)
    return [os.path.join(DYLAN_DATA, f) for f in df.file.head(n)]


def get_rhymes(word):
    try:
        d = get_cmu_dict()
        return set(get_rhyme_from_pronunciation(p) for p in d[word] if p)
    except KeyError:
        return []


def get_rhyme_from_pronunciation(pronunciation):

    stresses = []
    for p in pronunciation:
        try:
            stresses.append(int(p[-1]))
        except ValueError:
            pass
    stress = str(max(stresses))

    # the reversed is needed to deal with the "because" case
    for i, e in enumerate(reversed(pronunciation)):
        if e.endswith(stress):
            return ','.join(pronunciation[len(pronunciation) - 1 - i:])


def process_word(word, replace_dict=None):
    processed_word = word.lower()
    if replace_dict is not None:
        for k, v in replace_dict.items():
            if k in processed_word:
                processed_word = processed_word.replace(k, v)
    return processed_word


def tokenize(string, replace_dict=None):
    words = [process_word(token, replace_dict) for token in nltk.word_tokenize(string) if token not in BLACKLIST]
    # strip nonrhyming words, which includes stuff like punctuation
    return [START_SYMBOL] + words + [END_SYMBOL]


def tokenize_corpus(sources, replace_dict=None):
    sentences = []
    for file_name in sources:
        try:
            with open(file_name) as f:
                sentences += [tokenize(line, replace_dict) for line in f if line.strip()]
        except IOError:
            pass
    return sentences


def get_coeffs(transitions):
    return {prefix: sum(probabilities.values()) for prefix, probabilities in transitions.items()}


def normalize_it(values, coeff):
    return {suffix: value / float(coeff) for suffix, value in values.items()}


def normalize(transitions, coeffs=None):
    # normalization coeffs are optional
    if coeffs is None:
        coeffs = get_coeffs(transitions)
    res = empty_transitions()
    for prefix, probabilities in transitions.items():
        res[prefix] = normalize_it(probabilities, coeffs[prefix])
    return res


def empty_transitions():
    m = defaultdict(lambda: defaultdict(float))
    return m


def markov_transitions(sequences, order):
    # order None returns the dict structure directly
    m = empty_transitions()
    for seq in sequences:
        for n_gram in zip(*(seq[i:] for i in range(order + 1))):
            prefix = n_gram[:-1]
            suffix = n_gram[-1]
            m[prefix][suffix] += 1.
    return normalize(m)


def make_markov_corpus(sequences, order_upper):
    return {order: markov_transitions(sequences, order) for order in range(order_upper + 1)}


def filter_(transitions, values):
    if values is None:
        return transitions
    if hasattr(values, "keys"):
        res = copy.deepcopy(transitions)
        res_keys = res.keys()
        # make the partial keys up front... annoying
        r = [[(rk, rk[-i:]) for idx, rk in enumerate(res_keys)] for i in range(1, len(res_keys[0]) + 1)]
        res_k = [[rii[0] for rii in ri] for ri in r]
        res_m = [[rii[1] for rii in ri] for ri in r]
        for prefix, suffix in values.items():
            i = len(prefix) - 1
            p_m = res_m[i]
            p_k = res_k[i]
            if prefix in p_m:
                # find indices
                ii = [n for n, _ in enumerate(p_m) if prefix == _]
                # delete full matches
                for di in ii:
                    # looks ugly due to checks
                    if p_k[di] in res:
                        for su in suffix:
                            if su in res[p_k[di]]:
                                del res[p_k[di]][su]
        return res
    else:
        res = {}
        for prefix, probs in transitions.items():
            filtered = {suffix: probs[suffix] for suffix in probs.keys() if suffix in values}
            if filtered:
                res[prefix] = filtered
        return res


def propagate_(constrained_markov, coeffs):
    if coeffs is None:
        return constrained_markov

    res = {}
    for prefix, probs in constrained_markov.items():
        transitions = {}
        for suffix, value in probs.items():
            index = prefix[1:] + (suffix,)
            if index in coeffs:
                transitions[suffix] = value * coeffs[index]
            else:
                # for lower order ones
                index = prefix + (suffix,)
                if index in coeffs:
                    transitions[suffix] = value * coeffs[index]
                # if it's not in the coeffs, just pass
        if transitions:
            res[prefix] = transitions
    return res


def make_constrained_markov(markovs, constraints):
    # Section 4.3 of https://www.ijcai.org/Proceedings/11/Papers/113.pdf
    # Finite-Length Markov Processes with Constraints
    # Pachet, Roy, Barbieri
    # constraints are hard requirements for the sequence
    # dict rules are transitions which should be disallowed
    coeffs = None
    orders = markovs.keys()
    max_order = max(orders)
    markov_process = []
    for index, values in reversed(list(enumerate(constraints))):
        transitions = markovs[min(index, max_order)]
        filtered = filter_(transitions, values)
        filtered = propagate_(filtered, coeffs)
        if not filtered:
            raise RuntimeError('The constraints satisfaction problem has no solution. '
                               'Try to relax your constraints')
        coeffs = get_coeffs(filtered)
        # prepend because of reverse
        markov_process.insert(0, normalize(filtered, coeffs))
    return markov_process


def generate_from_constrained_markov_process(constrained_markov_process, random_state):
    max_order = len(constrained_markov_process[-1].keys()[0])
    sequence = []
    for index, markov in enumerate(constrained_markov_process):
        prefix = tuple(sequence[-min(index, max_order):])
        probs = markov[prefix]
        value = random_state.choice(probs.keys(), p=probs.values())
        sequence.append(value)
    return sequence


class SuffixNode(object):
    # borrowed from Gabriele Barbieri's code
    def __init__(self, value=None):
        self.value = value
        self.exts = {}

    def create_ext(self, value):
        if value in self.exts:
            return self.exts[value]
        else:
            ext = SuffixNode(value)
            self.exts[value] = ext
            return ext

    def parse_sub_sequence(self, sequence):
        node = self
        for element in sequence:
            node = node.create_ext(element)

    def parse(self, sequence):
        for i, _ in enumerate(sequence):
            self.parse_sub_sequence(sequence[i:])

    def get_order(self, sequence):
        if not sequence:
            return 0
        node = self
        i = 0
        for i, e in enumerate(sequence):
            if e in node.exts:
                node = node.exts[e]
            else:
                return i
        return i + 1

    def get_all_orders(self, sequence):
        return [self.get_order(sequence[i:]) for i, _ in enumerate(sequence)]

    def get_max_order(self, sequence):
        return max(self.get_all_orders(sequence))


def get_suffix_tree(sequences):
    tree = SuffixNode()
    for seq in sequences:
        tree.parse(seq)
    return tree


def test_markov_process():
    # constraints are listed PER STEP
    # rule constraints are list of list, e.g. [[prefix1, prefix2],[suffix]]
    # this one will match tutorial
    order = 1
    # mc should exactly match end of 4.4
    #c = [None, None, None, ["D"]]
    # this one checks hard unary constraints that *SHOULD* happen
    # c = [["E"], ["C"], ["C"], ["D"]]
    # can have multiple unary constraints - output should be in this set
    #c = [["E", "C"], ["E", "C"], ["E", "C"], ["D"]]
    # this one checks pairwise transitions that shouldn't happen
    c = [None, None, {("E",): ["D","C"], ("C",): ["D"]}, ["D"]]

    # can also do higher order
    #order = 2
    #c = [None, None, ["E"]]
    # binary constraints up to markov order
    #c = [None, None, {("C", "D"): ["E"]}]
    # can accept constraints that are shorter, for partial match
    #c = [None, None, {"E": ["E"]}]
    corpus = [["E", "C", "D", "E", "C", "C"],
              ["C", "C", "E", "E", "D", "C"]]
    # turn it into words
    ms = make_markov_corpus(corpus, order)
    mc = make_constrained_markov(ms, c)
    import numpy as np
    random_state = np.random.RandomState(100)
    for i in range(5):
        print(generate_from_constrained_markov_process(mc, random_state))
        # can also seed the generation, thus bypassing the prior
        #print(generate_from_constrained_markov_process(mc, random_state, starting_seed=["C"]))
        # also for higher order, seed with order length
        #print(generate_from_constrained_markov_process(mc, random_state, starting_seed=["E", "C"]))

if __name__ == "__main__":
    #test_markov_process()
    sources = get_dylan_most_popular_songs(40)
    order = 2
    corpus = tokenize_corpus(sources)
    ms = make_markov_corpus(corpus, order)
    # add not constraints
    # like !D, !C etc
    # can we iteratively create the max-order constraints?
    # using suffix tree, writing to c and recompiling mc?
    length = 10
    c = [["<s>"]] + [None] * (length + 1) + [["</s>"]]
    # remove you
    #c[1] = {("<s>",): ["you"]}
    #c[2] = {("<s>", "you"): ["'ve"]}
    #c[5] = {("been", "through"): ["all"]}
    # check partial match
    #c[5] = {("through",): ["all"]}
    mc = make_constrained_markov(ms, c)
    random_state = np.random.RandomState(100)
    for i in range(10):
        #print(generate_from_constrained_markov_process(mc, random_state, starting_seed=["<s>"]))
        print(generate_from_constrained_markov_process(mc, random_state))
    from IPython import embed; embed(); raise ValueError()
