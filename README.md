Usage
=====

`python markov_continuator.py`

or

`python markov_steerable.py`

or

`python exponential_families.py`


Info
====
This repo implements (parts of) several ideas from Sony CSL, led by Francois Pachet. See References for paper citations/links. Hear some example audio [at this link](https://badsamples.tumblr.com/post/164569290687/experiments-in-musical-textures-based-on-work).

Exponential families is in a semi-modifiable form (see constants at the top of the file)
but takes a while to run the first time.

Continuator experiment is easier to modify (constants near the top) but is not very exciting.

Using the constraint experiments (steerable) will generate chord progressions, but needs more know-how to use. Here the modifications should happen at the bottom of the file.

This is *not* meant to be easy to use on new data, or out-of-the box usable without knowledge of the papers.
The goal was for me to understand a series of interesting works,
and hopefully provide some code that could be read/hacked up by other motivated people to understand the core ideas in these papers.

That said, feel free to fork and add whatever you want :D


A Bit on the Constrained Markov Process
=======================================
An instance of a CMP object can be used by the .insert, and .branch methods. When creating these objects, one can specify the following:

order to calculate likelihoods over

max order, over which there *cannot* be any copying from the dataset (default None disables the check)

ptype, which is how the probability is calculated for longer history

constraints, passed as a dictionary

These constraints are *requirements* that each sequence returned from .branch
must satisfy. "start", "end", and "position" arguments all define that a given place in the sequence
must have one of the elements of the list in that position. The alldiff constraint specifies that every sequence element *must* be different.

Calling the .insert method with a sequence (list of tokens) will add it to the CMP object.

Finally when calling branch the two key arguments are the sequence to start from, and the length of the sequence to generate. This is the expensive call, .insert should be fast.

Constraints pt 2
================
For example, this snippet approximates the setup from "Markov Constraints: Steerable Generation of Markov Sequences"

```
order = 1
m = CMP(order,
        max_order=None,
        ptype="fixed",
        named_constraints={"not_contains": ["C7"],
                           "position": {8: ["F7"]},
                           "alldiff": True,
                           "end": ["G7"]})
m.insert(chord_seq1)
m.insert(chord_seq2)
m.insert(chord_seq3)
t = m.branch(["C7"], 15)
```

Adding constraints can greatly *decrease* runtime, due to reducing the search space.
ptype "fixed" is faster to evaluate than the other options ("max", "avg") but gives different results.
Changing the branch search type is not recommended, but could be useful in new problems


Tips
====
Best sounding results come from editing `/etc/timidity/timidity.cfg`
and using `source /etc/timidity/fluidr3_gm.cfg` instead of the default `source /etc/timidity/freepats.cfg`
This may require the fluid-soundfont package, which I installed with `sudo apt-get install fluidsynth`
To play midi files, do `timidity whatever_sample.mid`.
Use the helper function `timidifyit.sh myfile.mid` to convert mid files into wav


Requirements
============
numpy

scipy

music21

pretty\_midi

python 2.7 (3 may work, not tested)

timidity (optional)


References
==========
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

Style Imitation and Chord Invention in Polyphonic Music with Exponential Families
Gaetan Hadjeres, Jason Sakellariou, Francois Pachet
https://arxiv.org/abs/1609.05152

The omnibook data came from here
http://francoispachet.fr/texts/12BarBluesOmnibook.txt

All Bach data is pulled from Music21 (https://github.com/cuthbertLab/music21)
