from micrograd.micrograd.engine import Value


def main():

    a = Value(-4.0)  # Value(data=-4.0, grad=0)
    # a.data:float = -4.0
    # a.grad:int = 0
    # a._backward:function = <function Value.__init__.<locals>.<lambda> at 0x7f9ddeb85a20>
    # a._prev:set = {}
    # a._op:str = ''

    b = Value(2.0)  # Value(data=2.0, grad=0)
    # b.data:float = 2.0
    # b.grad:int = 0
    # b._backward:function = <function Value.__init__.<locals>.<lambda> at 0x7f9ddeb860e0>
    # b._prev:set = {}
    # b._op:str = ''

    c = a + b  # Value(data=-2.0, grad=0), a + b 实际会调用 a.__add__(b)。
    # c.data:float = -2.0
    # c.grad:int = 0
    # c._backward:function = <function Value.__add__.<locals>._backward at 0x7f9ddeb86200>
    # c._prev:set = {Value(data=-4.0, grad=0), Value(data=2.0, grad=0)}  # 存储产生c的a和b
    # c._op:str = '+'
    
    d = a * b + b**3  # Value(data=0.0, grad=0)
    # d.data:float = 0.0
    # d.grad:int = 0
    # d._backward:function = <function Value.__add__.<locals>._backward at 0x7f9ddeb863b0>
    # d._prev:set = {Value(data=8.0, grad=0), Value(data=-8.0, grad=0)}  # 存储产生d的a*b和b**3
    # d._op:str = '+'

    c += c + 1  # Value(data=-3.0, grad=0), 等价于 c = c + (c+1)
    # c.data:float = -3.0
    # c.grad:int = 0
    # c._backward:function = <function Value.__add__.<locals>._backward at 0x7f0c9532e560>
    # c._prev:set = {Value(data=-1.0, grad=0), Value(data=-2.0, grad=0)}
    # c._op:str = '+'

    c += 1 + c + (-a)  # Value(data=-1.0, grad=0), 等价于 c = c + (1 + c + (-a))
    # c.data:float = -1.0
    # c.grad:int = 0
    # c._backward:function = <function Value.__add__.<locals>._backward at 0x7f1d45e668c0>
    # c._prev:set = {Value(data=-3.0, grad=0), Value(data=2.0, grad=0)}
    # c._op:str = '+'

    d += d * 2 + (b + a).relu()  # Value(data=0.0, grad=0)
    d += 3 * d + (b - a).relu()  # Value(data=6.0, grad=0)
    e = c - d  # Value(data=-7.0, grad=0)
    f = e**2  # Value(data=49.0, grad=0)
    g = f / 2.0  # Value(data=24.5, grad=0)
    g += 10.0 / f  # Value(data=24.70408163265306, grad=0)

    print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
    # a = Value(data=-4.0, grad=0)
    # b = Value(data=2.0, grad=0)
    # c = Value(data=-1.0, grad=0)
    # d = Value(data=6.0, grad=0)
    # e = Value(data=-7.0, grad=0)
    # f = Value(data=49.0, grad=0)
    # g = Value(data=24.70408163265306, grad=0)
    g.backward()
    # a = Value(data=-4.0, grad=138.83381924198252)
    # b = Value(data=2.0, grad=645.5772594752186)
    # c = Value(data=-1.0, grad=-6.941690962099126)
    # d = Value(data=6.0, grad=6.941690962099126)
    # e = Value(data=-7.0, grad=-6.941690962099126)
    # f = Value(data=49.0, grad=0.4958350687213661)
    # g = Value(data=24.70408163265306, grad=1)
    print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
    print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db



    print("done")


if __name__ == "__main__":
    main()