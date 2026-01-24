from micrograd.micrograd.nn import MLP


def main():

    # initialize a model 
    # model = MLP(2, [16, 16, 1]) # 2-(hidden) layer neural network
    model = MLP(2, [3, 3, 1]) # 2-(hidden) layer neural network
    print(model)
    print("number of parameters", len(model.parameters()))

    parameters = model.parameters()
    print(parameters)





if __name__ == "__main__":
    main()