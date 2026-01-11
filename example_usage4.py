from micrograd.micrograd.engine import Value


def main():

    a = Value(1.0)  # Value(data=1.0, grad=0)
    # a.data = 1.0
    # a.grad = 0
    # a._backward = <function Value.__init__.<locals>.<lambda> at 0x7f0dd8735a20>
    # a._prev = {}
    # a._op = ''

    b = Value(2.0)  # Value(data=2.0, grad=0)
    # b.data = 2.0
    # b.grad = 0
    # b._backward = <function Value.__init__.<locals>.<lambda> at 0x7f0dd8736290>
    # b._prev = {}
    # b._op = ''

    c = 3 * a + 4 * b  # Value(data=11.0, grad=0)
    # c.data = 11.0
    # c.grad = 0
    # c._backward = <function Value.__add__.<locals>._backward at 0x7f0dd87365f0>
    # c._prev = {Value(data=8.0, grad=0), Value(data=3.0, grad=0)}
    # c.op = '+'

    # a = Value(data=1.0, grad=0)
    # b = Value(data=2.0, grad=0)
    # c = Value(data=11.0, grad=0)
    c.backward()
    # a = Value(data=1.0, grad=3)
    # b = Value(data=2.0, grad=4)
    # c = Value(data=11.0, grad=1)


    print("done")


if __name__ == "__main__":
    main()