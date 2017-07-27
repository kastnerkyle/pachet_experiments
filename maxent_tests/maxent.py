import numpy as np
from collections import Counter, defaultdict
from minimize import minimize

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


class MaxEnt(object):
    """ Also called a log-linear model, or logistic regression. """
    def __init__(self, feature_function, n_features, n_classes, smoothing=0,
                 verbose=True):
        # feature function returns list of indices
        # features are only indicator
        # assume sparse setup
        self.n_features = n_features
        self.n_classes = n_classes
        self.params = np.zeros((self.n_classes * self.n_features + self.n_classes,))
        self.weights = self.params[:self.n_classes * self.n_features].reshape(self.n_features, self.n_classes)
        self.biases = self.params[-self.n_classes:]
        self.feature_function = feature_function
        self.class_feature_counters = [Counter() for c in range(n_classes)]
        self.class_priors = [0 for c in range(n_classes)]
        self.class_defaults = [defaultdict(lambda: smoothing) for c in range(n_classes)]
        self.verbose = verbose

    def fit(self, data, labels, weight_cost):
        max_n_line_search = np.inf
        p, g, n_line_searches = minimize(self.params.copy(),
                                         (data, labels, weight_cost),
                                         self.f,
                                         self.g,
                                         maxnumlinesearch=max_n_line_search,
                                         verbose=self.verbose)
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

    def predict_proba(self, data):
        transform_data = [sorted(list(set(self.feature_function(d)))) for d in data]
        max_classes = max([max(li) for li in transform_data]) + 1
        oh_data = self._oh(transform_data, max_classes)
        scores = np.dot(oh_data, self.weights) + self.biases[None, :]
        probs = np.exp(scores - log_sum_exp(scores, 0))
        return probs

    def _cost(self, data, labels, weight_cost):
        transform_data = [sorted(list(set(self.feature_function(d)))) for d in data]
        max_classes = max([max(li) for li in transform_data]) + 1
        oh_data = self._oh(transform_data, max_classes)

        max_labels = max(labels) + 1
        oh_labels = self._oh(labels, max_labels)
        scores = np.dot(oh_data, self.weights) + self.biases[None, :]
        nll = -((oh_labels * scores).sum() + log_sum_exp(scores, 0).sum())
        nll = nll / float(oh_data.shape[1]) + weight_cost * np.sum(self.weights ** 2).sum()
        if self.verbose:
            print("nll {}".format(nll))
        return nll

    def _grads(self, data, labels, weight_cost):
        grad_w = np.zeros((self.n_classes, self.n_features))
        grad_b = np.zeros((self.n_classes,))
        transform_data = [sorted(list(set(self.feature_function(d)))) for d in data]
        max_classes = max([max(li) for li in transform_data]) + 1
        oh_data = self._oh(transform_data, max_classes)

        max_labels = max(labels) + 1
        oh_labels = self._oh(labels, max_labels)
        scores = np.dot(oh_data, self.weights) + self.biases[None, :]
        probs = np.exp(scores - log_sum_exp(scores, 0))
        for c in range(self.n_classes):
            grad_w[c, :] = -np.sum((oh_labels[:, c] - probs[:, c])[:, None] * oh_data, 0)
            grad_b[c] = -np.sum(oh_labels[:, c] - probs[:, c])
        grad_w /= float(self.n_features)
        grad_b /= float(self.n_features)
        grad_w = grad_w.T + 2 * weight_cost * self.weights
        grads = np.hstack((grad_w.flatten(), grad_b))
        if self.verbose:
            print("grads {}".format(grads.sum()))
        return grads

    def f(self, x, features, labels, weight_cost):
        xold = self.params.copy()
        self.update_params(x.copy())
        result = self._cost(features, labels, weight_cost)
        self.update_params(xold.copy())
        return result

    def g(self, x, features, labels, weight_cost):
        """Wrapper function around gradient to check grads, etc."""
        xold = self.params.copy()
        self.update_params(x.copy())
        result = self._grads(features, labels, weight_cost).flatten()
        self.update_params(xold.copy())
        return result

    def update_params(self, new_params):
        """ Update model parameters."""
        self.params[:] = new_params.copy()


class SMaxEnt(object):
    """ Also called a log-linear model, or logistic regression.
        Implementation using sparsity for discrete features"""
    def __init__(self, feature_function, n_features, n_classes, smoothing=0,
                 verbose=True):
        # feature function returns list of indices
        # features are only indicator
        # assume sparse setup
        self.n_features = n_features
        self.n_classes = n_classes
        self.params = np.zeros((self.n_classes * self.n_features + self.n_classes,))
        self.weights = self.params[:self.n_classes * self.n_features].reshape(self.n_features, self.n_classes)
        self.biases = self.params[-self.n_classes:]
        self.feature_function = feature_function
        self.class_feature_counters = [Counter() for c in range(n_classes)]
        self.class_priors = [0 for c in range(n_classes)]
        self.class_defaults = [defaultdict(lambda: smoothing) for c in range(n_classes)]
        self.verbose = verbose

    def fit(self, data, labels, weight_cost):
        max_n_line_search = np.inf
        p, g, n_line_searches = minimize(self.params.copy(),
                                         (data, labels, weight_cost),
                                         self.f_and_g,
                                         True,
                                         maxnumlinesearch=max_n_line_search,
                                         verbose=self.verbose)
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

    def predict_proba(self, data):
        label_scores = np.zeros((len(data), self.n_classes))
        # combine fwd and bwd
        for n, x in enumerate(data):
            active_idx = sorted(list(set(self.feature_function(x))))
            if len(active_idx) == 0:
                continue
            active_weights = self.weights[active_idx, :]
            active_biases = self.biases[None, :]
            sscores = active_weights + active_biases
            label_scores[n] = sscores.sum(axis=0)
        sprobs = np.exp(label_scores - log_sum_exp(label_scores, 0))
        return sprobs

    def _ohgrads(self, data, labels, weight_cost):
        grad_w = np.zeros((self.n_classes, self.n_features))
        grad_b = np.zeros((self.n_classes,))
        transform_data = [sorted(list(set(self.feature_function(d)))) for d in data]
        max_classes = max([max(li) for li in transform_data]) + 1
        oh_data = self._oh(transform_data, max_classes)

        max_labels = max(labels) + 1
        oh_labels = self._oh(labels, max_labels)
        scores = np.dot(oh_data, self.weights) + self.biases[None, :]
        probs = np.exp(scores - log_sum_exp(scores, 0))
        for c in range(self.n_classes):
            grad_w[c, :] = -np.sum((oh_labels[:, c] - probs[:, c])[:, None] * oh_data, 0)
            grad_b[c] = -np.sum(oh_labels[:, c] - probs[:, c])
        grad_w /= float(self.n_features)
        grad_b /= float(self.n_features)
        grad_w = grad_w.T + 2 * weight_cost * self.weights
        grads = np.hstack((grad_w.flatten(), grad_b))
        if self.verbose:
            print("grads {}".format(grads.sum()))
        return grads

    def _ohcost(self, data, labels, weight_cost):
        transform_data = [sorted(list(set(self.feature_function(d)))) for d in data]
        max_classes = max([max(li) for li in transform_data]) + 1
        oh_data = self._oh(transform_data, max_classes)

        max_labels = max(labels) + 1
        oh_labels = self._oh(labels, max_labels)
        scores = np.dot(oh_data, self.weights) + self.biases[None, :]
        nll = -((oh_labels * scores).sum() + log_sum_exp(scores, 0).sum())
        nll = nll / float(oh_data.shape[1]) + weight_cost * np.sum(self.weights ** 2).sum()
        if self.verbose:
            print("nll {}".format(nll))
        return nll

    def _cost_and_grads(self, data, labels, weight_cost):
        assert len(data) == len(labels)
        max_labels = max(labels) + 1
        oh_labels = self._oh(labels, max_labels)
        label_scores = np.zeros((len(data), self.n_classes))
        for n, (x, y) in enumerate(zip(data, labels)):
            active_idx = sorted(list(set(self.feature_function(x))))
            if len(active_idx) == 0:
                continue
            active_weights = self.weights[active_idx, :]
            active_biases = self.biases
            sscores = active_weights.sum(axis=0) + active_biases
            label_scores[n] = sscores
        nll = -((oh_labels * label_scores).sum() + log_sum_exp(label_scores, 0).sum())
        nll = nll / float(self.n_features) + weight_cost * np.sum(self.weights ** 2).sum()
        if self.verbose:
            print("nll {}".format(nll))
        nll = self._ohcost(data, labels, weight_cost)

        sgrad_w = np.zeros((self.n_classes, self.n_features))
        sgrad_b = np.zeros((self.n_classes,))
        sprobs = np.exp(label_scores - log_sum_exp(label_scores, 0))
        for n, (x, y) in enumerate(zip(data, labels)):
            active_idx = sorted(list(set(self.feature_function(x))))
            if len(active_idx) == 0:
                continue
            for c in range(self.n_classes):
                sgrad_w[c, active_idx] += -(oh_labels[n, c] - sprobs[n, c])
                sgrad_b[c] += -(oh_labels[n, c] - sprobs[n, c])
        sgrad_w /= float(self.n_features)
        sgrad_b /= float(self.n_features)
        grads = np.hstack((sgrad_w.flatten(), sgrad_b))
        if self.verbose:
            print("grads_sum {}".format(grads.sum()))
        grads = self._ohgrads(data, labels, weight_cost)
        return nll, grads

    def f_and_g(self, x, features, labels, weight_cost):
        xold = self.params.copy()
        self.update_params(x.copy())
        result = self._cost_and_grads(features, labels, weight_cost)
        self.update_params(xold.copy())
        return result

    def update_params(self, new_params):
        """ Update model parameters."""
        self.params[:] = new_params.copy()
