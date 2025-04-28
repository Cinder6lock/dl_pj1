from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return np.matmul(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        self.grads['W'] = np.matmul(self.input.T, grad)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        return np.matmul(grad, self.W.T)
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initialize_method = initialize_method
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        # 权重：[out_channels, in_channels, K, K]
        self.W = self.initialize_method(size=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        # 偏置：[out_channels]
        self.b = np.zeros(self.out_channels)

        # 参数和梯度字典
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        no padding
        """
        self.input = X  # 保存用于 backward
        B, C_in, H, W = X.shape
        K = self.kernel_size
        s = self.stride

        H_out = (H - K) // s + 1
        W_out = (W - K) // s + 1

        Y = np.zeros((B, self.out_channels, H_out, W_out))

        for b in range(B):
            for c_out in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        for c_in in range(C_in):
                            # 取出一个 kernel 区块
                            patch = X[b, c_in, i*s:i*s+K, j*s:j*s+K]
                            Y[b, c_out, i, j] += np.sum(patch * self.W[c_out, c_in])
                        Y[b, c_out, i, j] += self.b[c_out]
        return Y

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        X = self.input
        B, C_in, H, W = X.shape
        K = self.kernel_size
        s = self.stride

        H_out = (H - K) // s + 1
        W_out = (W - K) // s + 1

        # 初始化梯度
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        dX = np.zeros_like(X)

        for b in range(B):
            for c_out in range(self.out_channels):
                for i in range(H_out):
                    for j in range(W_out):
                        dy = grads[b, c_out, i, j]
                        for c_in in range(C_in):
                            patch = X[b, c_in, i*s:i*s+K, j*s:j*s+K]
                            dW[c_out, c_in] += dy * patch
                            dX[b, c_in, i*s:i*s+K, j*s:j*s+K] += dy * self.W[c_out, c_in]
                        db[c_out] += dy

        if self.weight_decay:
            dW += self.weight_decay_lambda * self.W

        self.grads['W'] = dW
        self.grads['b'] = db
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.optimizable = False
        self.has_softmax = True
        self.model = model
        self.max_classes = max_classes

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        # / ---- your codes here ----/
        self.predicts = predicts
        self.labels = labels
        self.probs = softmax(self.predicts)
        correct_probs = self.probs[np.arange(predicts.shape[0]), labels]
        loss = -np.log(correct_probs + 1e-12)
        return np.mean(loss)
    
    def backward(self):
        # first compute the grads from the loss to the input
        # / ---- your codes here ----/
        grad = self.probs.copy()
        batch_size = self.predicts.shape[0]
        grad[np.arange(batch_size), self.labels] -= 1
        grad /= batch_size
        self.grads = grad
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.optimizable = False
        self.input_shape = None
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X) -> np.ndarray:
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)
    
    def backward(self, grads) -> np.ndarray:
        return grads.reshape(self.input_shape)
    
class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.optimizable = False
        self.input = None
        self.argmax_mask = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        B, C, H, W = X.shape
        k = self.kernel_size
        s = self.stride

        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        output = np.zeros((B, C, H_out, W_out))
        self.argmax_mask = np.zeros_like(X)

        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * s
                        w_start = j * s
                        window = X[b, c, h_start:h_start + k, w_start:w_start + k]
                        max_val = np.max(window)
                        output[b, c, i, j] = max_val

                        # 保存最大值位置用于反向传播
                        max_index = np.unravel_index(np.argmax(window), window.shape)
                        self.argmax_mask[b, c, h_start + max_index[0], w_start + max_index[1]] = 1
        return output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)
        k = self.kernel_size
        s = self.stride
        B, C, H_out, W_out = grad_output.shape

        for b in range(B):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * s
                        w_start = j * s
                        grad = grad_output[b, c, i, j]
                        mask = self.argmax_mask[b, c, h_start:h_start + k, w_start:w_start + k]
                        grad_input[b, c, h_start:h_start + k, w_start:w_start + k] += grad * mask
        return grad_input