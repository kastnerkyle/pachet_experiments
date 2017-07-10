#!/usr/bin/env python
import numpy as np
from scipy.cluster.vq import vq
import os
import cPickle as pickle
import copy
import collections
from collections import defaultdict

from pthbldr.datasets import pitches_and_durations_to_pretty_midi
from pthbldr.datasets import quantized_to_pretty_midi
from pthbldr.datasets import list_of_array_iterator
from pthbldr.datasets import fetch_bach_chorales_music21
from pthbldr.utils import minibatch_kmedians, beamsearch

#mu = fetch_lakh_midi_music21(subset="pop")
#mu = fetch_haralick_midi_music21(subset="mozart_piano")
#mu = fetch_symbtr_music21()
#mu = fetch_wikifonia_music21()

#n_epochs = 500
#n_epochs = 2350
#n_epochs = 3000

mu = fetch_bach_chorales_music21()
order = len(mu["list_of_data_pitch"][0])

random_state = np.random.RandomState(1999)
key = "minor"

lp = mu["list_of_data_pitch"]
lt = mu["list_of_data_time"]
ltd = mu["list_of_data_time_delta"]
lql = mu["list_of_data_quarter_length"]

if key != None:
    keep_lp = []
    keep_lt = []
    keep_ltd = []
    keep_lql = []
    lk = mu["list_of_data_key"]
    for n in range(len(lp)):
        if key in lk[n]:
            keep_lp.append(lp[n])
            keep_lt.append(lt[n])
            keep_ltd.append(ltd[n])
            keep_lql.append(lql[n])
    lp = copy.deepcopy(keep_lp)
    lt = copy.deepcopy(keep_lt)
    ltd = copy.deepcopy(keep_ltd)
    lql = copy.deepcopy(keep_lql)


default_quarter_length = 55
voice_type = "woodwinds"
split = 20
clip_gen = 20
which_voice = 0

# https://csl.sony.fr/downloads/papers/uploads/pachet-02f.pdf
# https://stackoverflow.com/questions/11015320/how-to-create-a-trie-in-python
class Continuator:
    def __init__(self, random_state):
        self.root = dict()
        self.index = dict()
        # 0 indexed
        self.continuation_offset = 0
        self.random_state = random_state
        # use this to reduce the complexity of queries
        self.max_seq_len_seen = 0


    def insert(self, list_of_symbol, continuation_offset=None):
        if isinstance(list_of_symbol, (str, unicode)):
            raise AttributeError("list of symbol must not be string")
        word = list_of_symbol
        if continuation_offset is None:
            for n, wi in enumerate(word):
                # 1 indexed to match the paper
                self.index[n + self.continuation_offset + 1] = wi
            self.continuation_offset += len(word)
            continuation_offset = self.continuation_offset
            self.max_seq_len_seen = max(len(word), self.max_seq_len_seen)
        co = continuation_offset
        root = self.root
        current = root
        word_slice = word[:-1]
        for letter in word_slice[::-1]:
            if letter not in current:
                current[letter] = [co, {}]
            else:
                current[letter].insert(len(current[letter]) - 1, co)
            current = current[letter][-1]
        current["_end"] = None
        if len(word) > 1:
            self.insert(word[:-1], co - 1)


    def _prefix_search(self, prefix):
        root = self.root
        current = root
        subword = prefix[::-1]
        continuations = []
        for letter in subword:
            if letter in current and "_end" in current[letter][-1].keys():
                continuations += current[letter][:-1]
                return continuations
            elif letter not in current:
                # node not found
                return []
            current = current[letter][-1]
        # short sequence traversed to partial point of tree ("BC case from paper")
        continuations = []
        for k in current.keys():
            continuations += current[k][:-1]
        return continuations


    def _index_lookup(self, indices):
        return [self.index[i] for i in indices]


    def _next(self, prefix):
        ci = self._prefix_search(prefix)
        if len(ci) > 0:
            possibles = self._index_lookup(ci)
        else:
            sub_prefix = prefix[-self.max_seq_len_seen + 1:]
            possibles = None
            for i in range(len(sub_prefix)):
                ci = self._prefix_search(sub_prefix[i:])
                if len(ci) > 0:
                    possibles = self._index_lookup(ci)
                    break
        if possibles is not None:
            # choose one of possibles
            irange = np.arange(len(possibles))
            i = self.random_state.choice(irange)
            p = possibles[i]
        else:
            p = ""
        return [p]

    def continuate(self, seq, max_steps=-1):
        if isinstance(seq, (str, unicode)):
            raise AttributeError("prefix must list of symbols, not string")
        res = None
        i = 0
        new_seq = []
        while res != [""]:
            if max_steps > 0 and i > max_steps:
                break
            if res is not None:
                new_seq = new_seq + res
            res = t._next(seq)
            i += 1
        return new_seq

'''
# tests from
# https://csl.sony.fr/downloads/papers/uploads/pachet-02f.pdf
random_state = np.random.RandomState(1999)
t = Continuator(random_state)
t.insert(["A", "B", "C", "D"])
t.insert(["A", "B", "B", "C"])
ret = t.continuate(["A", "B"])
# should be ["B", "B", "C", "D"]

# Test the duration / tuple case
random_state = np.random.RandomState(1999)
t = Continuator(random_state)
t.insert([("A", 1), ("B", 1), ("C", 1), ("D", 1)])
t.insert([("A", 1), ("B", 1), ("B", 1), ("C", 1)])
ret = t.continuate([("A", 1), ("B", 1)])
# should be [("B", 1), ("B", 1), ("C", 1), ("D", 1)]
'''

random_state = np.random.RandomState(1999)
t = Continuator(random_state)

inds = range(len(lp))
for ii in inds:
    pii = lp[ii][0]
    tdii = ltd[ii][0]
    if len(pii) % split != 0:
        offset = split * (len(pii) // split)
        pii = pii[:offset]
        tdii = tdii[:offset]

    if len(tdii) < split or len(pii) < split:
        continue

    tdr = np.array(tdii).reshape(len(tdii) // split, -1)
    pr = np.array(pii).reshape(len(pii) // split, -1)

    for i in range(len(tdr)):
        tdri = tdr[i]
        pri = pr[i]
        comb = [(pi, tdi) for pi, tdi in zip(pri, tdri)]
        t.insert(comb)

"""
tri = tr[0][:20]
pri = pr[0][:20]
comb_q = pitch_and_time_to_piano_roll(pri, tri, min_div)
ret = t.continuate(comb_q, 100)
ret = [ret]
name_tag = "generated_{}.mid"
quantized_to_pretty_midi(ret, min_div,
                         save_dir="samples/samples",
                         name_tag=name_tag,
                         #list_of_quarter_length=[int(.5 * qpm) for qpm in qpms],
                         default_quarter_length=default_quarter_length,
                         voice_params=voice_type)
"""

tri = tdr[which_voice]
pri = pr[which_voice]
comb = [(pi, tdi) for pi, tdi in zip(pri, tdri)]
ret = t.continuate(comb, clip_gen)

name_tag = "generated_{}.mid"
pitches = [[r[0] for r in ret]]
durations = [[r[1] for r in ret]]
pitches_and_durations_to_pretty_midi(pitches, durations,
                                     save_dir="samples/samples",
                                     name_tag=name_tag,
                                     #list_of_quarter_length=[int(.5 * qpm) for qpm in qpms],
                                     default_quarter_length=default_quarter_length,
                                     voice_params=voice_type)

pii = [lp[0][which_voice]]
tdii = [ltd[0][which_voice]]
name_tag = "sample_{}.mid"
pitches_and_durations_to_pretty_midi(pii, tdii,
                         save_dir="samples/samples",
                         name_tag=name_tag,
                         #list_of_quarter_length=[int(.5 * qpm) for qpm in qpms],
                         default_quarter_length=default_quarter_length,
                         voice_params=voice_type)
"""
qii = pitch_and_time_to_piano_roll(pii, tii, min_div)
name_tag = "sample_{}.mid"
qii = [qii]
quantized_to_pretty_midi(qii, min_div,
                         save_dir="samples/samples",
                         name_tag=name_tag,
                         #list_of_quarter_length=[int(.5 * qpm) for qpm in qpms],
                         default_quarter_length=default_quarter_length,
                         voice_params=voice_type)
"""
