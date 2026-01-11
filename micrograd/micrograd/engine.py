# engine.py实现了自动微分(automatic differentiation)的核心引擎，是micrograd的基础。
# 微分: 数学概念，求导数。
# 自动微分: 用计算机自动计算导数的方法。
# 反向传播: 自动微分的一种实现方式，通过反向遍历计算图应用链式法则。


class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None  # 空函数
        # 每个Value对象初始化时都有一个_backward方法，默认为空操作。
        # 运算时会被重新赋值为实际的反向传播函数。
        # 反向传播时，调用v._backward(), 叶子节点执行空函数，运算节点执行实际的反向传播逻辑。
        self._prev = set(_children)
        # previous的缩写，存储产生当前节点的前驱节点(子节点)
        # set使用输入的元组_children构建一个集合(set)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc
        # operator的缩写，存储产生当前节点的操作。
        # 单下划线前缀变量(如_backward,_prev,_op)表示“内部使用”的成员变量，按约定不应被外部直接访问或修改。

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # self:Value = Value(data=11.0, grad=0), 反向传播的起始节点c

        # topological order all of the children in the graph
        topo = []  # topo列表记录节点顺序
        visited = set()  # visited集合记录已访问节点，避免重复访问
        def build_topo(v):
            if v not in visited:
                visited.add(v)  
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        # reversed(topo) = [Value(data=11.0, grad=1), Value(data=3.0, grad=0), Value(data=3, grad=0), Value(data=1.0, grad=0), Value(data=8.0, grad=0), Value(data=2.0, grad=0), Value(data=4, grad=0)]
        for v in reversed(topo):  # reversed函数用于反向遍历一个序列
            v._backward()  
        # Value类中定义了多个_backward()方法，一个节点如何决定要调用哪个_backward()?
        # 看生成当前节点时的运算是什么？如果是两数相加得到当前节点，则调用__add__函数下的嵌套函数。
        # 调用当前节点的_backward()函数时，实际更新的是当前节点的前驱节点(生成该节点的节点)的梯度grad。

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
