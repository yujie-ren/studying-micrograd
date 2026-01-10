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

    c = 2 * a + 3 * b  # Value(data=-2.0, grad=0)
    # c.data:float = -2.0
    # c.grad:int = 0
    # c._backward:function = <function Value.__add__.<locals>._backward at 0x7f9ddeb86200>
    # c._prev:set = {Value(data=-4.0, grad=0), Value(data=2.0, grad=0)}  # 存储产生c的a和b
    # c._op:str = '+'

    d = c ** 2  # Value(data=4.0, grad=0)
    # d.data:float = 4.0
    # d.grad:int = 0
    # d._backward:function = <function Value.__pow__.<locals>._backward at 0x7fa3dd52a4d0>
    # d._prev:set = {Value(data=-2.0, grad=0)}
    # d._op:str = '**2'

    # a = Value(data=-4.0, grad=0)
    # b = Value(data=2.0, grad=0)
    # c = Value(data=-2.0, grad=0)
    # d = Value(data=4.0, grad=0)
    d.backward()
    # a = Value(data=-4.0, grad=-8.0)
    # b = Value(data=2.0, grad=-12.0)
    # c = Value(data=-2.0, grad=-4.0)
    # d = Value(data=4.0, grad=1)

    print(d)




    print("done")


if __name__ == "__main__":
    main()