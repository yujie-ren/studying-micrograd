# studying-micrograd
source rep: 
https://github.com/karpathy/micrograd


## 一次反向传播实际做了什么？

直觉理解：
反向传播就是在计算：如果我稍微改一下某个参数，loss会怎么变？

精确描述(GPT):
反向传播(backpropagation)做的事情是：高效地计算损失函数对模型中所有可学习参数的偏导数(梯度)。

李宏毅关于Neural Network的讲解：
https://www.youtube.com/watch?v=Dr-WRlEFefw&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=12