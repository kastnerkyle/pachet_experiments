import numpy as np
from collections import Counter, defaultdict
from minimize import minimize
import scipy as sp
import copy
import hashlib

class memoize(object):
    def __init__(self, func):
        self.func = func
        self.lu = {}

    def __call__(self, *args):
        try:
            ha = hash(args)
            return self.lu[args]
        # numpy array in there
        except TypeError:
            new_args = []
            for a in args:
                try:
                    hash(a)
                    new_args.append(a)
                except TypeError:
                    b = a.view(np.uint8)
                    b_as_str = hashlib.sha1(b).hexdigest()
                    new_args.append(b_as_str)
            ha = hash(tuple(new_args))
        if ha in self.lu:
            return self.lu[ha]
        else:
            r = self.func(*args)
            self.lu[ha] = r
            return r


def log_sum_exp(x, axis=-1):
    """Compute log(sum(exp(x))) in a numerically stable way.

       Use second argument to specify along which dimensions the logsumexp
       shall be computed. If -1 (which is also the default), logsumexp is
       computed along the last dimension.

       From R. Memisevic
    """
    if len(x.shape) < 2:  #only one possible dimension to sum over?
        x_max = x.max()
        return x_max + np.log(np.sum(np.exp(x - x_max)))
    else:
        if axis != -1:
            x = x.transpose(range(axis) + range(axis + 1, len(x.shape)) + [axis])
        last = len(x.shape) - 1
        x_max = x.max(last)
        return x_max + np.log(np.sum(np.exp(x - x_max[..., None]), last))


def softmax(x):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


class SparseMaxEnt(object):
    """ Also called a log-linear model, or logistic regression.
        Implementation using sparsity for discrete features"""
    def __init__(self, feature_function, n_features, n_classes,
                 random_state=None, shuffle=True, optimizer="lbfgs",
                 verbose=True):
        # feature function returns list of indices
        # features are only indicator
        # assume sparse setup
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = random_state
        self.shuffle = shuffle
        self.optimizer = optimizer
        if random_state == None:
            raise ValueError("Random state must not be None!")
        self.params = 0.02 * random_state.randn(self.n_classes * self.n_features + self.n_classes)
        #self.params = np.zeros((self.n_classes * self.n_features + self.n_classes,))
        self.weights = self.params[:self.n_classes * self.n_features].reshape(self.n_features, self.n_classes)
        self.biases = self.params[-self.n_classes:]
        # memoize it
        self.feature_function = feature_function
        self.mem_feature_function = memoize(feature_function)
        self.verbose = verbose

    def fit(self, data, labels, l1_weight_cost=0., l2_weight_cost=0.):
        if self.optimizer == "lbfgs":
            from scipy.optimize import minimize
            res = minimize(self.f_and_g, self.params.copy(),
                           (data, labels, l1_weight_cost, l2_weight_cost), method="L-BFGS-B", jac=True,
                           options={"ftol": 1E-4})
            p = res.x
        elif self.optimizer == "minimize_cg":
            max_n_line_search = np.inf
            p, g, n_line_searches = minimize(self.params.copy(),
                                             (data, labels, l1_weight_cost, l2_weight_cost),
                                             self.f_and_g,
                                             True,
                                             maxnumlinesearch=max_n_line_search,
                                             verbose=self.verbose)
        else:
            raise ValueError("Unknown optimizer setting {}".format(self.optimizer))

        if self.verbose:
            print("Training complete!")
            self.update_params(p)

    def _oh(self, x, max_classes=None):
        if max_classes == None:
            n_classes = self.n_classes
        else:
            n_classes = max_classes
        #list of list == lol
        # need to normalize...
        try:
            max_len = max([len(xi) for xi in x])
            empty = np.zeros((len(x), max_len)) - 1
            for n, xi in enumerate(x):
                empty[n, :len(xi)] = xi
        except TypeError:
            max_len = 1
            empty = np.zeros((len(x), max_len)) - 1
            for n, xi in enumerate(x):
                empty[n] = xi

        result = np.zeros([len(x)] + [n_classes], dtype="int")
        z = np.zeros(len(x)).astype("int64")
        for c in range(n_classes):
            z *= 0
            z[np.where(empty == c)[0]] = 1
            result[..., c] += z
        return result

    def _uh(self, oh_x):
        return oh_x.argmax(len(oh_x.shape)-1)

    def loglikelihoods(self, data, pseudolabels):
        # trim means return regardless of matching original data length
        active_idxs = self.feature_function(data)
        inds = [n for n in range(len(active_idxs)) if hasattr(active_idxs[n], "flatten") or active_idxs[n] != None]
        not_inds = [n for n in range(len(active_idxs)) if not hasattr(active_idxs[n], "flatten") and active_idxs[n] == None]

        active_idxs = [active_idxs[ii] for ii in inds]

        label_scores = np.zeros((len(active_idxs), self.n_classes))
        for n in range(len(active_idxs)):
            active_idx = active_idxs[n]
            active_weights = self.weights[active_idx, :]
            active_biases = self.biases
            sscores = active_weights.sum(axis=0) + active_biases
            label_scores[n] = sscores
        sprobs = softmax(label_scores)

        final_probs = []
        si = 0
        for ii in range(len(data)):
            if ii in inds:
                new = sprobs[si]
                final_probs.append(new)
                si += 1
            elif ii in not_inds:
               new = 0. * sprobs[0] - 1.
               final_probs.append(new)
            else:
               raise ValueError("This shouldnt happen")
        sprobs = np.array(final_probs)
        sub_idx = [l for l in list(range(len(data))) if l not in not_inds]
        lls = np.zeros_like(sprobs[:, 0]) - 1E8
        lls[sub_idx] = np.log(sprobs[list(range(len(data))), pseudolabels][sub_idx])
        return lls

    def predict_proba(self, data):
        # trim means return regardless of matching original data length
        active_idxs = self.feature_function(data)
        inds = [n for n in range(len(active_idxs)) if hasattr(active_idxs[n], "flatten") or active_idxs[n] != None]
        not_inds = [n for n in range(len(active_idxs)) if not hasattr(active_idxs[n], "flatten") and active_idxs[n] == None]

        active_idxs = [active_idxs[ii] for ii in inds]

        label_scores = np.zeros((len(active_idxs), self.n_classes))
        for n in range(len(active_idxs)):
            active_idx = active_idxs[n]
            active_weights = self.weights[active_idx, :]
            active_biases = self.biases
            sscores = active_weights.sum(axis=0) + active_biases
            label_scores[n] = sscores
        sprobs = softmax(label_scores)

        final_probs = []
        si = 0
        for ii in range(len(data)):
            if ii in inds:
                new = sprobs[si]
                final_probs.append(new)
                si += 1
            elif ii in not_inds:
               new = 0. * sprobs[0] - 1.
               final_probs.append(new)
            else:
               raise ValueError("This shouldnt happen")
        return np.array(final_probs)

    def _cost_and_grads(self, data, labels, l1_weight_cost, l2_weight_cost):
        assert len(data) == len(labels)

        # switch to block transform...
        # preparation for block transform
        active_idxs = self.mem_feature_function(data)
        if len(active_idxs) != len(labels):
            raise ValueError("feature_function should return same number of datapoints! Return None for entries to ignore in training")

        # short circuit OR to avoid issues with array compare
        inds = [n for n in range(len(active_idxs)) if hasattr(active_idxs[n], "flatten") or active_idxs[n] != None]

        if self.shuffle:
            self.random_state.shuffle(inds)

        active_idxs = [active_idxs[ii] for ii in inds]
        labels = [labels[ii] for ii in inds]

        label_scores = np.zeros((len(labels), self.n_classes))
        for n in range(len(active_idxs)):
            active_idx = active_idxs[n]
            active_weights = self.weights[active_idx, :]
            active_biases = self.biases
            sscores = active_weights.sum(axis=0) + active_biases
            label_scores[n] = sscores

        sprobs = softmax(label_scores)
        # https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models
        nll = -np.sum(np.log(sprobs)[list(range(len(labels))), labels])
        nll = nll / float(len(labels)) + l1_weight_cost * np.sum(np.abs(self.weights)).sum() + l2_weight_cost * np.sum(self.weights ** 2).sum()
        if self.verbose:
            print("nll {}".format(nll))

        # see non-sparse derivation http://cs231n.github.io/neural-networks-case-study/#loss
        dsprobs = sprobs
        dsprobs[list(range(len(labels))), labels] -= 1
        dsprobs /= float(len(labels))

        sgrad_w = np.zeros((self.n_features, self.n_classes))
        sgrad_b = np.zeros((self.n_classes,))
        # use cached active_idxs
        #for n, (x, y) in enumerate(zip(data, labels)):
        #    active_idx = sorted(list(set(self.feature_function(x))))
        #    if len(active_idx) == 0:
        #        continue
        for n in range(len(active_idxs)):
            active_idx = active_idxs[n]
            sgrad_w[active_idx] += dsprobs[n]
            sgrad_b += dsprobs[n]
        sgrad_w += l1_weight_cost * np.sign(self.weights)
        sgrad_w += l2_weight_cost * self.weights
        grads = np.hstack((sgrad_w.flatten(), sgrad_b))
        if self.verbose:
            print("grads_norm {}".format(np.sqrt((grads ** 2).sum())))
        return nll, grads

    def f_and_g(self, x, features, labels, l1_weight_cost, l2_weight_cost):
        xold = self.params.copy()
        self.update_params(x.copy())
        result = self._cost_and_grads(features, labels, l1_weight_cost, l2_weight_cost)
        self.update_params(xold.copy())
        return result

    def update_params(self, new_params):
        """ Update model parameters."""
        self.params[:] = new_params.copy()
