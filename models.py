from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        self.layers = []
        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            layer.W = param_list[i + 2]['W']
            layer.b = param_list[i + 2]['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = param_list[i + 2]['weight_decay']
            layer.weight_decay_lambda = param_list[i+2]['lambda']
            if self.act_func == 'Logistic':
                raise NotImplementedError
            elif self.act_func == 'ReLU':
                layer_f = ReLU()
            self.layers.append(layer)
            if i < len(self.size_list) - 2:
                self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, in_channel=1, conv_channels=None, linear_dims=None,
                 kernel_size=3, lambda_list=None):
        super().__init__()
        assert conv_channels is not None and linear_dims is not None
        self.conv_channels = conv_channels
        self.linear_dims = linear_dims
        self.kernel_size = kernel_size

        self.layers = []
        in_ch = in_channel

        # 构造卷积 + 激活 + 池化层
        conv_layer_idx = 0
        for out_ch in conv_channels:
            conv = conv2D(in_ch, out_ch, kernel_size)
            if lambda_list is not None:
                conv.weight_decay = True
                conv.weight_decay_lambda = lambda_list[conv_layer_idx]
            self.layers.append(conv)
            self.layers.append(ReLU())
            self.layers.append(MaxPool2D(2, 2))
            in_ch = out_ch
            conv_layer_idx += 1

        self.layers.append(Flatten())
        dummy_input = np.zeros((1, in_channel, 28, 28))
        out = dummy_input
        for layer in self.layers:
            out = layer(out)
        flatten_dim = out.shape[1]

        # 构造全连接层
        in_dim = flatten_dim
        for i, out_dim in enumerate(linear_dims):
            layer = Linear(in_dim, out_dim)
            if lambda_list is not None:
                layer.weight_decay = True
                layer.weight_decay_lambda = lambda_list[conv_layer_idx + i]  # 注意加了 conv_layer_idx 偏移
            self.layers.append(layer)
            if i < len(linear_dims) - 1:
                self.layers.append(ReLU())
            in_dim = out_dim

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, loss_grad):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def load_model(self, param_path):
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        idx = 0
        for layer in self.layers:
            if layer.optimizable:
                layer.params['W'] = param_list[idx]['W']
                layer.params['b'] = param_list[idx]['b']
                layer.weight_decay = param_list[idx]['weight_decay']
                layer.weight_decay_lambda = param_list[idx]['lambda']
                idx += 1

    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                })
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)