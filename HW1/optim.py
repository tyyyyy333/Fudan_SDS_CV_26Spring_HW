import numpy as np
class Optimizer:
    def __init__(self, model, lr=0.01, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def _params(self):
        return self.model.parameters()

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        self.model.zero_grad()


class SGD(Optimizer):
    def __init__(self, model, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(model, lr=lr, weight_decay=weight_decay)
        self.momentum = momentum
        self.velocity = [np.zeros_like(param) for param, _ in self._params()]

    def step(self):
        for i, (param, grad) in enumerate(self._params()):
            decay = self.weight_decay * param if param.ndim > 1 else 0.0
            update_grad = grad + decay
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] + update_grad
                update = self.velocity[i]
            else:
                update = update_grad
            param -= self.lr * update


class Adam(Optimizer):
    def __init__(
        self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0
    ):
        super().__init__(model, lr=lr, weight_decay=weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(param) for param, _ in self._params()]
        self.v = [np.zeros_like(param) for param, _ in self._params()]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(self._params()):
            decay = self.weight_decay * param if param.ndim > 1 else 0.0
            grad = grad + decay
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError


class ExponentialLR(Scheduler):
    def __init__(self, optimizer, gamma=0.95):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self.optimizer.lr *= self.gamma


class LambdaLR(Scheduler):
    def __init__(self, optimizer, lr_lambda):
        super().__init__(optimizer)
        self.lr_lambda = lr_lambda
        self.base_lr = optimizer.lr
        self.step_count = -1

    def step(self):
        self.step_count += 1
        self.optimizer.lr = self.base_lr * self.lr_lambda(self.step_count)
