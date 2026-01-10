from micrograd.micrograd.engine import Value


def main():

    a = Value(-4.0)  # Value(data=-4.0, grad=0)
    b = Value(2.0)  # Value(data=2.0, grad=0)

    c = a + b  # Value(data=-2.0, grad=0)
    # a + b 实际会调用 a.__add__(b)。
    print(c)



if __name__ == "__main__":
    main()