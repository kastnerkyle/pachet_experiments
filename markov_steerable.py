#!/usr/bin/env python
import numpy as np
from scipy.cluster.vq import vq
import os
import cPickle as pickle
import copy
import collections
from collections import defaultdict
import music21
from pthbldr.datasets import pitches_and_durations_to_pretty_midi

class Trie(object):
    def __init__(self):
        self.root = collections.defaultdict()
        self._end = "_end"
        self.orders = []

    def insert(self, list_of_items):
        current = self.root
        for item in list_of_items:
            current = current.setdefault(item, {})
        current.setdefault(self._end)
        self.orders = sorted(list(set(self.orders + [len(list_of_items)])))

    def order_insert(self, order, list_of_items):
        s = 0
        e = order
        while e < len(list_of_items):
            e = s + order
            self.insert(list_of_items[s:e])
            s += 1

    def search(self, list_of_items):
        # items of the list should be hashable
        # returns True if item in Trie, else False
        if len(list_of_items) not in self.orders:
            raise ValueError("item {} has invalid length {} for search, only {} supported".format(list_of_items, len(list_of_items), self.orders))
        current = self.root
        for item in list_of_items:
            if item not in current:
                return False
            current = current[item]
        if self._end in current:
            return True
        return False

    def partial(self, prefix_list):
        # items of the list should be hashable
        # Returns valid keys for continuation
        if len(prefix_list) + 1 not in self.orders:
            raise ValueError("item {} has invalid length {} for partial search, only {} supported".format(prefix_list, len(prefix_list), [o - 1 for o in self.orders]))
        current = self.root
        for p in prefix_list:
            if p not in current:
                return []
            current = current[p]
        return [c for c in current.keys() if c != self._end]


class MaxOrder(object):
    def __init__(self, max_order, ptype="avg"):
        self.bad = Trie()
        self.max_order = max_order
        self.order = max_order - 1
        self.goods = [Trie() for i in range(2, self.order)]
        self.ptype = ptype

    def insert(self, list_of_items):
        self.bad.order_insert(self.max_order, list_of_items)
        for i in list(range(2, self.order)):
            self.goods[i - 2].order_insert(i, list_of_items)

    def partial(self, prefix_list):
        # returns dict of item: prob
        if len(prefix_list) != self.order:
            raise ValueError("item {} has invalid length {} for partial search, only {} supported".format(prefix_list, len(prefix_list), self.order))
        all_p = []
        all_gp = []
        # smoothed probs for all chains
        for i in list(range(2, self.order))[::-1]:
            gp = self.goods[i - 2].partial(prefix_list[-i + 1:])
            bp = self.bad.partial(prefix_list)
            p = list(set(gp) - set(bp))
            if len(p) > 0:
                all_p += p
                all_gp += gp
                if self.ptype == "max":
                    break
        ps = list(set(all_p))
        gps = np.array(all_gp)
        d = {psi: sum(gps == psi) for psi in ps}
        s = sum(d.values())
        d = {k: float(v) / s for k, v in d.items()}
        return d

    def generate(self, seed_list, length, random_state):
        if len(seed_list) != self.order:
            raise ValueError("item {} has invalid length {} for seed, only {} supported".format(seed_list, len(seed_list), self.order))
        res = [s for s in seed_list]
        for i in range(length):
            nxt = m.partial(res[-self.order:])
            if len(nxt) > 0:
                r = sorted([(k, v) for k, v in nxt.items()])
                el = [ri[0] for ri in r]
                pp = [ri[1] for ri in r]
                res += [random_state.choice(el, p=pp)]
            else:
                return res
        return res

def realize_chord(chordstring, numofpitch=3, baseoctave=4, direction="ascending"):
    """
    given a chordstring like Am7, return a list of numofpitch pitches, starting in octave baseoctave, and ascending
    if direction == "descending", reverse the list of pitches before returning them
    """
    # https://github.com/shimpe/canon-generator
    # http://web.mit.edu/music21/doc/moduleReference/moduleHarmony.html
    try:
        pitches = music21.harmony.ChordSymbol(chordstring).pitches
    except ValueError:
        # enharmonic equivalents
        orig_chordstring = chordstring
        if "halfDim" in chordstring:
            chordstring = chordstring.replace("halfDim", "/o7")
        if chordstring[:2] == "Eb":
            chordstring = "D#" + chordstring[2:]

        try:
            pitches = music21.harmony.ChordSymbol(chordstring).pitches
        except ValueError:
            from IPython import embed; embed(); raise ValueError()

    num_iter = numofpitch / len(pitches) + 1
    octave_correction = baseoctave - pitches[0].octave
    result = []
    actual_pitches = 0
    for i in range(num_iter):
        for p in pitches:
            if actual_pitches < numofpitch:
                newp = copy.deepcopy(p)
                newp.octave = newp.octave + octave_correction
                result.append(newp)
                actual_pitches += 1
            else:
                if direction == "ascending":
                    return result
                else:
                    result.reverse()
                    return result
        octave_correction += 1

    if direction == "ascending":
        return result
    else:
        result.reverse()
        return result


# hardcode the data for now
with open("12BarBluesOmnibook.txt", "r") as f:
   r = f.readlines()
names = r[::2]
bars = r[1::2]
names = [n.strip() for n in names]
bars = [b.strip() for b in bars]

pairs = zip(names, bars)
new_bars = []
for n, b in pairs:
    bb = [bi.split("/") for bi in b.split("|")]
    bb = [bbii for bbi in bb for bbii in bbi]
    new_bars.append(bb)
pairs = zip(names, new_bars)

random_state = np.random.RandomState(1999)
max_order = 8

m = MaxOrder(max_order)
for p in pairs:
    m.insert(p[1])

r = []
for p in pairs:
    ri = m.generate(p[1][:max_order - 1], 20, random_state)
    r.append(ri)

midi_p = []
for ri in r:
    rch = [realize_chord(rii, 4) for rii in ri]
    rt = []
    for rchi in rch:
        rt.append([rchi[idx].midi for idx in range(len(rchi))])
    midi_p.append(rt)

midi_d = [[[2 for midi_ppii in midi_ppi] for midi_ppi in midi_pi] for midi_pi in midi_p]

midi_p = [np.array(midi_pi) for midi_pi in midi_p]
midi_d = [np.array(midi_di) for midi_di in midi_d]

name_tag = "sample_{}.mid"
pitches_and_durations_to_pretty_midi(midi_p, midi_d,
                                     save_dir="samples/samples",
                                     name_tag=name_tag,
                                     default_quarter_length=90,
                                     voice_params="piano")

"""
#for rii in r[0]:
#    print(rii)
#    realize_chord(rii, 3)

#for p in pairs:
#    r = m.generate(p[1][:max_order - 1], 20, random_state)
#    print(r)

#raise ValueError()
#from IPython import embed; embed(); raise ValueError()

good_t = Trie()
good_t.order_insert(4, pairs[0][1])

bad_t = Trie()
bad_t.order_insert(5, pairs[0][1])
"""


'''
# basic tests
test = Trie()
test.insert('helloworld')
test.insert('ilikeapple')
test.insert('helloz')

print(test.search('hello'))
print(test.partial('hello'))
print(test.search('ilikeapple'))
print(test.partial('helloz'))
print(test.partial('hellow'))
raise ValueError()
'''
