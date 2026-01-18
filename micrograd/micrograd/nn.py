import random
from .engine import Value


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []


class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        # 根据当前layer的输入数目，确定每个neuron的 w参数 的个数。
        # random.uniform(-1,1) 会生成[-1,1]之间的一个随机的浮点数，用于初始化w参数。
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
        # i = 0, Layer(sz[0]=2, sz[1]=16, nonlin=0!=2=True), [Neuron(2, nonlin=True) for _ in range(16)]
        # i = 1, Layer(sz[1]=16, sz[2]=16, nonlin=1!=2=True), [Neuron(16, nonlin=True) for _ in range(16)]
        # i = 2, Layer(sz[2]=16, sz[3]=1, nonlin=2!=2=True), [Neuron(16, nonlin=False) for _ in range(1)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):

    def __init__(self, nin, nouts):
        # nin (int): num of inputs in input layer (layer0), nin=2表示输入样本有2个特征。Input layer实际不是由neuron组成的，只是输入的向量拍成一排。
        # nouts (list): num of neurons from layer1 to output layer。nouts=[16, 16, 1]表示layer1(hidden layer 1)有16个neuron，layer2(hidden layer 2)有16个neuron，layer3(output layer)有1个neuron。
        # nin:int = 2
        # nouts:list = [16, 16, 1]
        sz = [nin] + nouts  # 所有层的尺寸列表(sizes)  [2, 16, 16, 1]
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
        # self.layers:list = [Layer of [ReLUNeuron(2), *16], Layer of [ReLUNeuron(16), *16], Layer of [LinearNeuron(16)]]
        # self.layers:list = [Layer of [Neuron List], Layer of [Neuron List], Layer of [Neuron List]]
        # for i in [0, 1, 2], 执行 Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1)
        # i = 0, Layer(sz[0]=2, sz[1]=16, nonlin=0!=2=True)
        # i = 1, Layer(sz[1]=16, sz[2]=16, nonlin=1!=2=True)
        # i = 2, Layer(sz[2]=16, sz[3]=1, nonlin=2!=2=False)
        # 常见的网络设计模式: 隐藏层用非线性激活(e.g., ReLU)增加表达能力，输出层根据任务选择是否加激活。
        #    回归: 通常输出层不加激活(或加线性激活)。
        #    二分类: 通常用sigmoid。
        #    多分类: 通常用softmax。


    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
        # 等价于  
        # result = []
        # for layer in self.layers:
        #   for p in layer.parameters():
        #       result.append(p)
        # return result

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
