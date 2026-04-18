import numpy as np

try:
    import layer as layer_module
except ImportError:
    from . import layer as layer_module


class BaseModel:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)
        return self

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def named_parameters(self):
        named_params = []
        for layer_idx, layer in enumerate(self.layers):
            for name in layer.params.keys():
                named_params.append(
                    (f"layer{layer_idx}.{name}", layer.params[name], layer.grads[name])
                )
        return named_params

    def parameters(self):
        return [(param, grad) for _, param, grad in self.named_parameters()]

    def state_dict(self):
        state = {}
        for name, param, _ in self.named_parameters():
            state[name] = param.copy()
        return state

    def load_state_dict(self, state):
        for name, param, _ in self.named_parameters():
            if name not in state:
                raise KeyError(f"Missing parameter in state_dict: {name}")
            if param.shape != state[name].shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {param.shape}, got {state[name].shape}"
                )
            param[...] = state[name]

    def save(self, path):
        np.savez(path, **self.state_dict())

    def load(self, path):
        loaded = np.load(path)
        state = {k: loaded[k] for k in loaded.files}
        self.load_state_dict(state)


def _get_activation(name):
    name = name.lower()
    if name == "relu":
        return layer_module.ReLU
    if name == "sigmoid":
        return layer_module.Sigmoid
    if name == "tanh":
        return layer_module.Tanh
    raise ValueError(f"Unsupported activation: {name}")


class MLP(BaseModel):
    def __init__(self, size_list, activation="relu", dropout=0.0):
        super().__init__()
        if len(size_list) < 2:
            raise ValueError("size_list must contain at least input and output size.")

        activation_cls = _get_activation(activation)
        for i in range(len(size_list) - 2):
            self.add(layer_module.Linear(size_list[i], size_list[i + 1]))
            self.add(activation_cls())
            if dropout > 0:
                self.add(layer_module.Dropout(dropout))

        self.add(layer_module.Linear(size_list[-2], size_list[-1]))

    def predict_logits(self, x):
        return self.forward(x, training=False)

    def predict_proba(self, x):
        logits = self.predict_logits(x)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def get_first_layer_weights(self):
        for layer in self.layers:
            if isinstance(layer, layer_module.Linear):
                return layer.params["weight"]
        return None
