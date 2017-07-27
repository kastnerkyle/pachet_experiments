import zipfile
from tokenizing import Tokenizer
from collections import Counter, defaultdict
from maxent import MaxEnt, SMaxEnt
import numpy as np

fname = "smsspamcollection.zip"
archive = zipfile.ZipFile(fname, 'r')
raw = archive.open(archive.infolist()[0]).readlines()
cats = [l.split("\t")[0] for l in raw]
labels = [0 if l == "ham" else 1 for l in cats]

data = [l.split("\t")[1].rstrip() for l in raw]
tok = Tokenizer(preserve_case=False)
tok_data = [tok.tokenize(s) for s in data]

vocab = Counter()
for tok_data_i in tok_data:
    vocab.update(tok_data_i)

lookup = defaultdict(lambda: 0)
vocab_size = 8000
n_classes = 2

# 0 is UNK
for n, (k, v) in enumerate(vocab.most_common(vocab_size - 1)):
    lookup[k] = n + 1

def feature_fn(s):
    si = tok.tokenize(s)
    indices = [lookup[sii] for sii in si]
    return indices

"""
tr = int(.8 * float(len(data)))
train_data = data[:tr]
train_labels = labels[:tr]

test_data = data[tr:]
test_labels = labels[tr:]

m1 = MaxEnt(feature_fn, n_features=vocab_size,
            n_classes=n_classes)
m2 = SMaxEnt(feature_fn, n_features=vocab_size,
             n_classes=n_classes)

nll1 = m1._cost(train_data, train_labels, 0.)
nll2 = m2._cost(train_data, train_labels, 0.)

grads1 = m1._grads(train_data, train_labels, 0.)
print(grads1.sum())
grads2 = m2._grads(train_data, train_labels, 0.)
print(grads2.sum())
'''
"""
model = SMaxEnt(feature_fn, n_features=vocab_size, n_classes=n_classes)
tr = int(.8 * float(len(data)))
train_data = data[:tr]
train_labels = labels[:tr]

test_data = data[tr:]
test_labels = labels[tr:]

model.fit(train_data, train_labels, 0.)
raise ValueError()
def do_eval(d):
    probs = model.predict_proba(d)
    preds = np.argmax(probs, axis=-1)
    hams = [dd for n, dd in enumerate(d) if n in np.where(preds == 0)[0]]
    spams = [dd for n, dd in enumerate(d) if n in np.where(preds == 1)[0]]
    return preds, probs, hams, spams

test_preds, test_probs, hams, spams = do_eval(test_data)
from IPython import embed; embed(); raise ValueError()
