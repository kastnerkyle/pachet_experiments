#!/usr/bin/env python
import numpy as np
from scipy.cluster.vq import vq
import os
import cPickle as pickle
import copy
import collections
from collections import defaultdict
from maxent import SparseMaxEnt
import cPickle as pickle
import time

from pthbldr.datasets import fetch_bach_chorales_music21
from pthbldr.datasets import quantized_to_pretty_midi

default_quarter_length = 55
voice_type = "woodwinds"

def pitch_and_duration_to_piano_roll(list_of_pitch_voices, list_of_duration_voices, min_dur):
    def expand(pitch, dur, min_dur):
        assert len(pitch) == len(dur)
        expanded = [int(d // min_dur) for d in dur]
        check = [d / min_dur for d in dur]
        assert all([e == c for e, c in zip(expanded, check)])
        stretch = [[p] * e for p, e in zip(pitch, expanded)]
        # flatten out to 1 voice
        return [pi for p in stretch for pi in p]

    res = []
    for lpv, ldv in zip(list_of_pitch_voices, list_of_duration_voices):
        qi = expand(lpv, ldv, min_dur)
        res.append(qi)

    min_len = min([len(ri) for ri in res])
    res = [ri[:min_len] for ri in res]
    piano_roll = np.array(res).transpose()
    return piano_roll


def get_data(offset=88, shuffle=True):
    if os.path.exists("tmp_data.pkl"):
        print("Found existing data storage {}, loading...".format("tmp_data.pkl"))
        with open("tmp_data.pkl", "r") as f:
            r = pickle.load(f)
        return r

    mu = fetch_bach_chorales_music21()
    order = len(mu["list_of_data_pitch"][0])

    random_state = np.random.RandomState(1999)
    key = "major"

    lp = mu["list_of_data_pitch"]
    lt = mu["list_of_data_time"]
    ltd = mu["list_of_data_time_delta"]
    lql = mu["list_of_data_quarter_length"]
    fnames = mu["filename_list"]

    if key != None:
        keep_lp = []
        keep_lt = []
        keep_ltd = []
        keep_lql = []
        keep_fnames = []
        lk = mu["list_of_data_key"]
        for n in range(len(lp)):
            if key in lk[n]:
                keep_lp.append(lp[n])
                keep_lt.append(lt[n])
                keep_ltd.append(ltd[n])
                keep_lql.append(lql[n])
                keep_fnames.append(fnames[n])
        lp = copy.deepcopy(keep_lp)
        lt = copy.deepcopy(keep_lt)
        ltd = copy.deepcopy(keep_ltd)
        lql = copy.deepcopy(keep_lql)
        fnames = copy.deepcopy(keep_fnames)

    all_pr = []
    for ii in range(len(lp)):
        pr = pitch_and_duration_to_piano_roll(lp[ii], ltd[ii], .125)
        all_pr.append(pr)

    note_set = set()
    for n, pr in enumerate(all_pr):
        uq = set(tuple(np.unique(pr)))
        note_set = note_set | uq
    note_set = sorted(list(set(note_set)))

    """
    name_tag = "actual_{}.mid"
    save_dir = "samples/samples"

    quantized_to_pretty_midi(all_pr[:10], .125,
                             save_dir=save_dir,
                             name_tag=name_tag,
                             default_quarter_length=80,
                             voice_params="nylon")
    """

    lut = {}
    rlut = {}
    i = 0
    for n in sorted(list(note_set)):
        lut[n] = i
        rlut[i] = n
        i += 1
    return all_pr, lut, rlut

offset = 88
h_context = 3

all_pieces, lut, rlut = get_data(offset)
dataset = np.concatenate(all_pieces, axis=0)

n_classes = len(lut.keys())
n_classes = offset
n_features_per = offset

# -h_context to h_context, ignore self = 2 * 3
# 3 for vertical minus self
# 3 for prev diag
# 3 for future diag
n_features = n_features_per * (2 * 3 + 3 + 3 + 3)
dataset = np.ascontiguousarray(dataset)

def feature_fn(X, i):
    which_voice = wv = i
    features = []
    notes = X
    for ii in range(h_context, len(notes) - h_context):
        tot = 0
        nv = [n for n in [0, 1, 2, 3] if n != which_voice]

        h_span = list(range(ii - h_context, ii + h_context + 1))
        h_span = [h for h in h_span if h != ii]
        h_n = []
        for hi in h_span:
            h_ni = lut[int(notes[hi, wv].ravel())] + tot * offset
            tot += 1
            h_n.append(h_ni)
        h_n = np.array(h_n).ravel()

        vm1_n = notes[ii - 1, nv].ravel()
        for nn in range(len(vm1_n)):
            vm1_n[nn] = lut[int(vm1_n[nn])] + tot * offset
            tot += 1

        v_n = notes[ii, nv].ravel()
        for nn in range(len(v_n)):
            v_n[nn] = lut[int(v_n[nn])] + tot * offset
            tot += 1

        vp1_n = notes[ii + 1, nv].ravel()
        for nn in range(len(v_n)):
            vp1_n[nn] = lut[int(vp1_n[nn])] + tot * offset
            tot += 1
        features_i = np.concatenate((h_n, vm1_n, v_n, vp1_n))
        features.append(features_i)
    return [None] * h_context + features + [None] * h_context

def feature_fn0(X):
    return feature_fn(X, 0)

def feature_fn1(X):
    return feature_fn(X, 1)

def feature_fn2(X):
    return feature_fn(X, 2)

def feature_fn3(X):
    return feature_fn(X, 3)

feature_fns = [feature_fn0, feature_fn1, feature_fn2, feature_fn3]

labels = {}
for which_voice in [0, 1, 2, 3]:
    labels[which_voice] = [lut[d] for d in dataset[:, which_voice]]

def get_models(dataset, labels):
    models = []
    random_state = np.random.RandomState(2100)
    for which_voice in [0, 1, 2, 3]:
        if not os.path.exists("saved_sme_{}.pkl".format(which_voice)):
            model = SparseMaxEnt(feature_fns[which_voice], n_features=n_features, n_classes=n_classes,
                                 random_state=random_state)
            start_time = time.time()
            l1 = 3E-5
            model.fit(dataset, labels[which_voice], l1)
            stop_time = time.time()
            print("Total training time {}".format(stop_time - start_time))
            with open("saved_sme_{}.pkl".format(which_voice), "w") as f:
                pickle.dump(model, f)
        else:
            print("Found saved model saved_sme_{}.pkl, loading...".format(which_voice))
            with open("saved_sme_{}.pkl".format(which_voice), "r") as f:
                model = pickle.load(f)
        models.append(model)
    return models

models = get_models(dataset, labels)

random_state = np.random.RandomState(2147)
generated = copy.copy(dataset[:100])
random_state.shuffle(generated)

def save_midi(generated, itr):
    print("Saving, iteration {}".format(itr))
    name_tag = "generated_{}".format(itr) + "_{}.mid"
    save_dir = "samples/samples"

    quantized_to_pretty_midi([generated[h_context:-h_context]], .125,
                             save_dir=save_dir,
                             name_tag=name_tag,
                             default_quarter_length=80,
                             voice_params="nylon")

total_itr = 100
for n in range(total_itr):
    print("Iteration {}".format(n))
    if n % 10 == 0 or n == (total_itr - 1):
        save_midi(generated, n)

    # all voices, over the comb range
    # metropolized gibbs comb?
    # OrMachine
    moves = list(range(generated.shape[1])) * (2 * h_context + 1)
    random_state.shuffle(moves)

    comb_offset = random_state.randint(10000) % (2 * h_context + 1)
    for m in moves:
        j = m
        ls = models[j].likelihoods(generated, generated[:, j])

        poss = [vi for vi in rlut.values() if vi > 0 and vi < 88]
        #rvv = random_state.choice(generated[h_context:-h_context, j], len(generated[h_context:-h_context]), replace=True)
        rvv = random_state.choice(poss, len(generated[h_context:-h_context]), replace=True)

        l = models[j].predict_proba(generated)
        valid_sub = l[h_context:-h_context].argmax(axis=1)
        cvv = np.array([rlut[vs] for vs in valid_sub])

        #cvv = generated[h_context - 1:-h_context - 1, j]

        # flip coin to choose between random and copy
        choose = np.array(np.random.rand(len(cvv)) > .85).astype("int16")

        vv = cvv #choose * rvv + (1 - choose) * cvv

        """
        # proba + shuffled randomly as proposal dist
        l = models[j].predict_proba(generated)
        valid_sub = l[h_context:-h_context].argmax(axis=1)
        vv = np.array([rlut[vs] for vs in valid_sub])
        random_state.shuffle(vv)
        """

        new_generated = copy.copy(generated)
        new_generated[h_context:-h_context, j] = vv
        new_ls = models[j].likelihoods(new_generated, new_generated[:, j])

        accept_score = new_ls[h_context:-h_context] / ls[h_context:-h_context]
        accept_roll = random_state.rand(len(vv))
        accept_ind = np.array([1 if (viv % (2 * h_context + 1)) == comb_offset else 0 for viv in range(len(vv))])

        # if new_ls small, chance is low
        #accept = np.array((accept_ind * accept_roll) < accept_score).astype("int32")
        accept = accept_ind
        comb_offset += 1
        if comb_offset >= 2 * h_context + 1:
            comb_offset = comb_offset % (2 * h_context + 1)
        generated[h_context:-h_context, j] = accept * vv + (1 - accept) * generated[h_context:-h_context, j]
