# Author: Kyle Kastner
# License: BSD 3-Clause
# built out of code from Gabriele Barbieri
# https://github.com/gabrielebarbieri/markovchain
# All mistakes my own
from collections import defaultdict
import copy


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


def propagate_coeffs(constrained_markov, coeffs):
    for prefix, v in constrained_markov.items():
        transitions = {}
        for suffix, prob in v.items():
            # handle prior
            if prefix == "<s>":
                index = (suffix,)
                if index in coeffs:
                    transitions[suffix] = prob * coeffs[index]
            else:
                index = prefix[1:] + (suffix,)
                if index in coeffs:
                    transitions[suffix] = prob * coeffs[index]

            if len(transitions.keys()) > 0:
                constrained_markov[prefix] = transitions
    return constrained_markov


def markov_transitions(sequences, order):
    # order None returns the dict structure directly
    m = empty_transitions()
    for seq in sequences:
        for n_gram in zip(*(seq[i:] for i in range(order + 1))):
            prefix = n_gram[:-1]
            suffix = n_gram[-1]
            m[prefix][suffix] += 1.
    return normalize(m)


def make_markov_corpus(sequences, order):
    # start = "<s>"
    # default to 2 grams
    # set order_upper and order_lower to the same value
    # to only do 1 markov type
    # range is inclusive
    # internally add 1
    p_order = max(order - 1, 0)
    prior = markov_transitions(sequences, p_order)
    if p_order == 0:
        tmp = prior[()]
        del prior[()]
        prior["<s>"] = tmp
    else:
        tmp = {}
        tmp["<s>"] = {}
        for k in prior.keys():
            for vi, vv in prior[k].items():
                tmp["<s>"][tuple(list(k) + [vi,])] = vv
        prior = normalize(tmp)
    d = {o: markov_transitions(sequences, o) for o in [order]}
    d[0] = prior
    return d


def make_constrained_markov(markovs, constraints):
    # Section 4.3 of https://www.ijcai.org/Proceedings/11/Papers/113.pdf
    # Finite-Length Markov Processes with Constraints
    # Pachet, Roy, Barbieri
    # constraints is hard requirements for the sequence
    # rules are transitions which should be disallowed
    alphas = None
    orders = markovs.keys()
    # initialized the constrained markovs to the original matrices
    # make sure we have independent copies for manipulation
    constrained_markovs = [copy.deepcopy({k: m for k, m in markovs.items() if k > 0}) for i in range(len(constraints))]
    # put the prior in front
    constrained_markovs.insert(0, {0: markovs[0]})

    for index, values in list(enumerate(constraints)):
        # None is a bypass
        if values is None:
            continue
        if hasattr(values, "keys"):
            # pairwise or greater, replace entry with 0 / skip it
            for o in constrained_markovs[index + 1].keys():
                tt = {}
                for k in constrained_markovs[index + 1][o].keys():
                    new_v = {}
                    # fix annoying case with string inputs
                    for vik, viv in values.items():
                        if isinstance(vik, basestring):
                            new_v[tuple([viik for viik in vik])] = viv
                        else:
                            new_v[vik] = viv
                    tim = {dk: dv for dk, dv in constrained_markovs[index + 1][o][k].items() if k not in new_v.keys() or (k in new_v.keys() and dk not in new_v[k])}
                    tt[k] = tim
                constrained_markovs[index + 1][o] = copy.deepcopy(tt)
        else:
            for v in values:
                # for unary constraints, replace whole column with 0
                # index + 1 to get correct step, since prior is at 0
                for o in constrained_markovs[index + 1].keys():
                    tt = {k: {dk: dv
                        for dk, dv in constrained_markovs[index + 1][o][k].items() if dk == v}
                    for k in constrained_markovs[index + 1][o].keys()}
                    constrained_markovs[index + 1][o] = copy.deepcopy(tt)
    # do the alpha updates in reverse time order, prune empty dicts
    coeffs = None
    for index in reversed(range(len(constrained_markovs))):
        # any way to handle multiple orders?
        current_order = max(constrained_markovs[index].keys())
        curr = constrained_markovs[index][current_order]
        # remove empty probs
        curr = {k: v for k, v in curr.items() if len(v.keys()) > 0}
        if coeffs is not None:
            curr = propagate_coeffs(curr, coeffs)
            curr = {k: v for k, v in curr.items() if len(v.keys()) > 0}
        coeffs = get_coeffs(curr)
        out = normalize(curr)
        constrained_markovs[index][current_order] = out
    return constrained_markovs


def generate_from_constrained_markov_process(constrained_markov_process, random_state, starting_seed=None, error_type="print"):
    # error_type can be "print" or "raise"
   
    if starting_seed is None:
        out = []
        prefix = "<s>"
        curr = constrained_markov_process[0][0][prefix]
        j = [(k, v) for k, v in curr.items()]
        elems = [n for n, ji in enumerate(j)]
        probs = [ji[1] for ji in j]
        choose_n = random_state.choice(elems, p=probs)
        choose = j[choose_n][0]
        if len(choose) > 1:
           out.extend(choose)
        else:
            out.append(choose)
    else:
        # check starting seed?
        if len(starting_seed) < max(constrained_markov_process[1].keys()):
            raise ValueError("starting_seed must be be a list as long as the markov order")

        out = starting_seed
    for index, constrained_markovs in enumerate(constrained_markov_process[1:]):
        current_order = max(constrained_markovs.keys())
        # get prefix
        prefix = tuple(out[-current_order:])
        if prefix not in constrained_markovs[current_order]:
            msg = "WARNING: terminated early at index {}, consider looser constraints or larger corpus".format(index)
            if error_type == "print":
                print(msg)
            else:
                raise ValueError(msg)
            return out
        curr = constrained_markovs[current_order][prefix]
        j = [(k, v) for k, v in curr.items()]
        elems = [ji[0] for ji in j]
        probs = [ji[1] for ji in j]
        choose = random_state.choice(elems, p=probs)
        out.append(choose)
    return out


if __name__ == "__main__":
    # constraints are listed PER STEP
    # rule constraints are list of list, e.g. [[prefix1, prefix2],[suffix]]
    # this one will match tutorial
    # mc should exactly match end of 4.4
    #c = [None, None, ["D"]]
    # this one checks hard unary constraints that *SHOULD* happen
    #c = [["E"], ["C"], ["C"], ["D"]]
    # this one checks pairwise transitions that shouldn't happen?
    #c = [None, None, {"E": ["D","C"], "C": ["D"]}]
    # can mix them
    #c = [None, None, {"E": ["E"]}]
    # can also do higher order such as order = 2
    c = [None, None, None, ["D"]]
    # currently, binary constraints must match marokov order
    # TODO: generalize this a bit to accept shorter transition constraints than order
    #c = [None, None, {("E", "C"): ["D"]}]
    corpus = ["ECDECC", "CCEEDC"]
    order = 2
    ms = make_markov_corpus(corpus, order)
    mc = make_constrained_markov(ms, c)
    import numpy as np
    random_state = np.random.RandomState(100)
    for i in range(5):
        print(generate_from_constrained_markov_process(mc, random_state))
        # can also seed the generation, thus bypassing the prior
        #print(generate_from_constrained_markov_process(mc, random_state, starting_seed=["C"]))
        #print(generate_from_constrained_markov_process(mc, random_state, starting_seed=["E", "C"]))
