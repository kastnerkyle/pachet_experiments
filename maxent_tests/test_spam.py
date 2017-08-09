import zipfile
from tokenizing import Tokenizer
from collections import Counter, defaultdict
from maxent import SparseMaxEnt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# based on https://stackoverflow.com/questions/23455728/scikit-learn-balanced-subsampling
def balanced_sample_maker(X, y, sample_size="max", random_state=None):
    y_o = y
    y = np.array(y)
    uniq_levels = np.unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}
    if sample_size == "max":
        sample_size = max([v for v in uniq_counts.values()])
    if random_state is None:
        raise ValueError("Must provide random state")

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx

    # oversampling on observations of each label
    balanced_copy_idx = []
    for gb_level, gb_idx in groupby_levels.iteritems():
        if len(gb_idx) >= sample_size:
            over_sample_idx = []
            over_sample_idx += gb_idx
        else:
            over_sample_idx = []
            # every minority datapoint at least once
            over_sample_idx += gb_idx
            ex_over_sample_idx = np.random.choice(gb_idx, size=sample_size - len(over_sample_idx), replace=True).tolist()
            over_sample_idx += ex_over_sample_idx
        balanced_copy_idx += over_sample_idx
    np.random.shuffle(balanced_copy_idx)
    X_bal = [X[idx] for idx in balanced_copy_idx]
    y_bal = [y_o[idx] for idx in balanced_copy_idx]
    return (X_bal, y_bal, balanced_copy_idx)


fname = "smsspamcollection.zip"
archive = zipfile.ZipFile(fname, 'r')
raw = archive.open(archive.infolist()[0]).readlines()
random_state = np.random.RandomState(1999)
random_state.shuffle(raw)
cats = [l.split("\t")[0] for l in raw]
labels = [0 if l == "ham" else 1 for l in cats]
data = [l.split("\t")[1].rstrip() for l in raw]

tr = int(.8 * float(len(data)))
train_data = data[:tr]
train_labels = labels[:tr]
#train_data, train_labels, _ = balanced_sample_maker(train_data, train_labels, random_state=random_state)

test_data = data[tr:]
test_labels = labels[tr:]
#test_data, test_labels, _ = balanced_sample_maker(test_data, test_labels, random_state=random_state)

tfidf = TfidfVectorizer()
tfidf.fit(train_data)
vocab_size = len(tfidf.get_feature_names())
n_classes = 2

def feature_fn(x):
    transformed = tfidf.transform([x])
    return transformed.nonzero()[1]

model = SparseMaxEnt(feature_fn, n_features=vocab_size, n_classes=n_classes,
                     random_state=random_state)

model.fit(train_data, train_labels, 0.)
def do_eval(d):
    probs = model.predict_proba(d)
    preds = np.argmax(probs, axis=-1)
    hams = [dd for n, dd in enumerate(d) if n in np.where(preds == 0)[0]]
    spams = [dd for n, dd in enumerate(d) if n in np.where(preds == 1)[0]]
    return preds, probs, hams, spams

train_preds, train_probs, train_hams, train_spams = do_eval(train_data)
train_score = np.mean([tl == tp for tl, tp in zip(train_labels, train_preds)])
print("Train accuracy {}".format(train_score))
test_preds, test_probs, test_hams, test_spams = do_eval(test_data)
test_score = np.mean([tl == tp for tl, tp in zip(test_labels, test_preds)])
print("Test accuracy {}".format(test_score))
from IPython import embed; embed(); raise ValueError()
