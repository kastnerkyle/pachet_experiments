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
            # + 1 due to numpy slicing
            e = s + order + 1
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


class Node(object):
    def __init__(self, level, proposed_note, log_prob, previous_notes):
        self.level = level
        self.proposed_note = proposed_note
        self.previous_notes = previous_notes
        self.log_prob = log_prob

    def __repr__(self):
        return str((self.level, self.proposed_note, self.log_prob, self.previous_notes))


class CMP(object):
    """ Constrained Markov Process

    Implements tools/ideas from the following papers:

    The Continuator: Musical Interaction with Style
    F. Pachet
    https://www.csl.sony.fr/downloads/papers/uploads/pachet-02f.pdf

    Finite-Length Markov Processes With Constraints
    F. Pachet, P. Roy, G. Barbieri
    https://www.csl.sony.fr/downloads/papers/2011/pachet-11b.pdf

    Markov Constraints: Steerable Generation of Markov Sequences
    F. Pachet, P. Roy
    https://www.csl.sony.fr/downloads/papers/2011/pachet-09c.pdf

    Avoiding Plagiarism in Markov Sequence Generation
    A. Papadopolous, P. Roy, F. Pachet
    https://www.csl.sony.fr/downloads/papers/2014/papadopoulos-14a.pdf

    Enforcing Meter in Finite-Length Markov Sequences
    P. Roy, F. Pachet
    https://www.csl.sony.fr/downloads/papers/2013/roy-13a.pdf

    Non-Conformant Harmonization: The Real Book in the Style of Take 6
    F. Pachet, P. Roy
    https://www.csl.sony.fr/downloads/papers/2014/pachet-14a.pdf
    """
    def __init__(self, order, max_order=None, ptype="max", named_constraints={},
                 verbose=False):

        self.order = order
        self.goods = [Trie() for i in range(0, self.order)]
        self.max_order = max_order
        constraint_types = ["end", "start", "position", "alldiff"]
        # need to flesh out API
        # position is dict of dict of list
        # alldiff key indicates window size
        assert all([k in constraint_types for k in named_constraints.keys()])
        self.named_constraints = named_constraints
        self.bad = Trie()
        self.ptype = ptype
        assert ptype in ["fixed", "max", "avg"]
        self.verbose = verbose

    def insert(self, list_of_items):
        if self.max_order is not None:
            self.bad.order_insert(self.max_order, list_of_items)
        for i in list(range(0, self.order)):
            self.goods[i].order_insert(i + 1, list_of_items)

    def partial(self, prefix_list):
        # returns dict of item: prob
        if len(prefix_list) >= self.order:
            if self.verbose:
                print("item {} has invalid length {} for partial search, using last {}".format(prefix_list, len(prefix_list), self.order))
            partial_prefix_list = prefix_list[-self.order:]
        else:
            if len(prefix_list) < self.order:
                raise ValueError("Prefix list for partial match too short")

            #raise ValueError("item {} has invalid length {} for partial search, only {} supported".format(prefix_list, len(prefix_list), self.order))
        all_p = []
        all_gp = []
        for i in list(range(0, self.order))[::-1]:
            gp = self.goods[i].partial(partial_prefix_list[-(i + 1):])
            # use old_prefix_list since that is the full data
            if self.max_order is not None and len(prefix_list) > self.max_order:
                bp = self.bad.partial(prefix_list[-self.max_order:])
            else:
                bp = []
            p = list(set(gp) - set(bp))
            if self.ptype == "fixed":
                all_p += p
                all_gp += gp

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

    def check_constraint(self, node, sequence, depth_index, max_length):
        generated = sequence[-(depth_index + 1):]
        if "alldiff" in self.named_constraints:
            # windowed alldiff?
            if len(set(generated)) != len(generated):
                return False

        if "start" in self.named_constraints:
            valid_start = self.named_constraints["start"]
            if generated[0] not in valid_start:
                return False

        if "end" in self.named_constraints:
            valid_end = self.named_constraints["end"]
            if depth_index == (max_length - 1) and sequence[-1] not in valid_end:
                return False

        if "position" in self.named_constraints:
            position_checks = self.named_constraints["position"]
            for k, v in position_checks.items():
                if len(generated) > k and generated[k] not in v:
                    return False
        return True


    def branch(self, seed_list, length):
        res = [s for s in seed_list]

        options = self.partial(res)

        el = []
        def push(i, p=None):
            el.append(i)

        def pop():
            return el.pop()

        for k, v in options.items():
            n = Node(0, k, np.log(v), tuple(res))
            push(n)

        soln = {}
        break_while = False
        while len(el) > 0 and break_while is False:
            current = pop()
            index = current.level
            cur_note = current.proposed_note
            cur_seq = current.previous_notes
            cur_log_prob = current.log_prob
            new_seq = cur_seq + (cur_note,)
            if index >= length:
                if cur_seq not in soln:
                    # soln: log_prob
                    soln[cur_seq] = cur_log_prob
            else:
                if self.check_constraint(current, new_seq, index, length):
                    options = self.partial(new_seq)
                    for k, v in options.items():
                        n = Node(index + 1, k, cur_log_prob + np.log(v), new_seq)
                        push(n)

        res = sorted([(v, k[len(seed_list):]) for k, v in soln.items()])[::-1]
        return res

    def constrained_greedy(self, seed_list, length, random_state):
        res = [s for s in seed_list]
        for i in range(length):
            nxt = m.partial(res)
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


def render_chords(list_of_chord_lists, name_tag, dur=2, tempo=110, voices=4,
                  voice_type="piano", save_dir="samples/samples"):
        r = list_of_chord_lists
        midi_p = []
        for ri in r:
            rch = [realize_chord(rii, voices) for rii in ri]
            rt = []
            for rchi in rch:
                rt.append([rchi[idx].midi for idx in range(len(rchi))])
            midi_p.append(rt)

        midi_d = [[[dur for midi_ppii in midi_ppi] for midi_ppi in midi_pi] for midi_pi in midi_p]

        midi_p = [np.array(midi_pi) for midi_pi in midi_p]
        midi_d = [np.array(midi_di) for midi_di in midi_d]

        midi_pp = []
        midi_dd = []
        for p, d in zip(midi_p, midi_d):
            w = np.where((p[:, 3] - p[:, 2]) > 12)[0]
            p[w, 3] = 0.
            midi_pp.append(p)
            midi_dd.append(d)

        pitches_and_durations_to_pretty_midi(midi_pp, midi_dd,
                                             save_dir=save_dir,
                                             name_tag=name_tag,
                                             default_quarter_length=tempo,
                                             voice_params=voice_type)

def transpose(chord_seq):
    roots = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    roots2map = {k: v for v, k in enumerate(roots)}
    # 2 octaves for easier transpose
    oct_roots = roots + roots
    map2roots = {k: v for k, v in enumerate(oct_roots)}

    prototype = []
    for c in chord_seq:
        if c[:-1] in roots2map:
            prototype.append(roots2map[c[:-1]])
        elif c[:2] in roots2map:
            prototype.append(roots2map[c[:2]])
        elif c[0] in roots2map:
            prototype.append(roots2map[c[0]])
        else:
            print(c)
            from IPython import embed; embed(); raise ValueError()

    chord_types = ["m", "7", "halfDim"]
    chord_function = []
    for c in chord_seq:
        if "halfDim" in c:
            chord_function.append("halfDim")
            continue
        elif c[-1] not in ["m", "7"]:
            chord_function.append("")
            continue
        chord_function.append(c[-1])

    assert len(chord_function) == len(prototype)
    all_t = []
    for i in range(len(roots)):
        t = [map2roots[p + i] + cf for p, cf in zip(prototype, chord_function)]
        all_t.append(t)
    return all_t


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


"""
final_pairs = []
for p in pairs:
    t_p = transpose(p[1])
    final_pairs += [(p[0], ti_p) for ti_p in t_p]
pairs = final_pairs
"""

random_state = np.random.RandomState(1999)
max_order = 5
rrange = 5
dur = 2
tempo = 110

#m = CMP(1, max_order=None, ptype="fixed", named_constraints={"start": ["C7"], "end": ["G7"], "position": {4: ["F7"]}}, verbose=False)
m = CMP(1, max_order=None, ptype="fixed", named_constraints={"alldiff": True, "end": ["G7"]}, verbose=False)
for p in pairs:
    m.insert(p[1])
t = m.branch(["C7"], 8)
if len(t) == 0:
    raise ValueError("No solution found!")

res = t[0][1]
# repeat 2x
render_chords([res + res], "sample_branch_{}.mid", dur=dur, tempo=tempo)
raise ValueError()

m = CMP(4, max_order=5, ptype="max", named_constraints={"start": ["C7"], "end": ["G7"]}, verbose=False)
#t = m.branch(pairs[0][1], 8)
raise ValueError("Complete")

# greedy example
r = []
for n, p in enumerate(pairs):
    print("Running greedy pair {} of {}".format(n, len(pairs)))
    part = p[1][:max_order - 1]
    ri = m.constrained_greedy(part, int(4 * rrange), random_state)
    r.append(ri)


# branch example
r = []
for n, p in enumerate(pairs):
    print("Running branch pair {} of {}".format(n, len(pairs)))
    part = p[1][:max_order - 1]

    b = part
    for i in range(rrange):
        ri = m.branch(b[-(max_order - 1):], 4)
        if len(ri) == 1:
            break
        part = ri[0][1]
        b += part[-(max_order - 1):]
    r.append(b)

midi_p = []
for ri in r:
    rch = [realize_chord(rii, 3) for rii in ri]
    rt = []
    for rchi in rch:
        rt.append([rchi[idx].midi for idx in range(len(rchi))])
    midi_p.append(rt)

# all half note
midi_d = [[[dur for midi_ppii in midi_ppi] for midi_ppi in midi_pi] for midi_pi in midi_p]

midi_p = [np.array(midi_pi) for midi_pi in midi_p]
midi_d = [np.array(midi_di) for midi_di in midi_d]

name_tag = "sample_branch_{}.mid"
pitches_and_durations_to_pretty_midi(midi_p, midi_d,
                                     save_dir="samples/samples",
                                     name_tag=name_tag,
                                     default_quarter_length=tempo,
                                     voice_params="piano")
