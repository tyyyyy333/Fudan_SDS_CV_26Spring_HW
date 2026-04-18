import numpy as np


class BaseLayer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, x, training=True):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def zero_grad(self):
        for name, value in self.params.items():
            self.grads[name] = np.zeros_like(value)


class Linear(BaseLayer):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier initialization for stable MLP training.
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.params["weight"] = np.random.uniform(
            -limit, limit, size=(in_features, out_features)
        ).astype(np.float32)
        self.params["bias"] = np.zeros((1, out_features), dtype=np.float32)
        self.zero_grad()
        self.input = None

    def forward(self, x, training=True):
        self.input = x
        return x @ self.params["weight"] + self.params["bias"]

    def backward(self, grad):
        self.grads["weight"] = self.input.T @ grad
        self.grads["bias"] = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.params["weight"].T


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x, training=True):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * self.mask


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x, training=True):
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grad):
        return grad * self.output * (1.0 - self.output)


class Tanh(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x, training=True):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad):
        return grad * (1.0 - self.output ** 2)


class Softmax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x, training=True, axis=-1):
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(shifted)
        self.output = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        return self.output

    def backward(self, grad):
        dot_product = np.sum(grad * self.output, axis=-1, keepdims=True)
        return self.output * (grad - dot_product)


class Dropout(BaseLayer):
    def __init__(self, p=0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability p must be in [0, 1).")
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if training and self.p > 0.0:
            self.mask = (np.random.rand(*x.shape) >= self.p).astype(np.float32)
            self.mask /= (1.0 - self.p)
            return x * self.mask
        self.mask = None
        return x

    def backward(self, grad):
        if self.mask is None:
            return grad
        return grad * self.mask


class CrossEntropy:
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits, one_hot_targets):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.targets = one_hot_targets
        loss = -np.sum(self.targets * np.log(self.probs + 1e-12)) / logits.shape[0]
        return float(loss)

    def backward(self):
        return (self.probs - self.targets) / self.targets.shape[0]
