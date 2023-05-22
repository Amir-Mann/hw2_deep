r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


1.a. The shape of the jacobian is '(N = 64, if_features = 1024, N = 64, out_features = 512)' 

1.b. yes, this matrix is sparse. because $\forall j\neq i:\frac{\partial y_i}{\partial x_j} = 0$, 
     and this is because we treat each sample independently from the rest of the batch.
     
1.c. There is no need to materialize the jacobian,  we can calculate $\frac{\partial L}{\partial X}$
     by performing matrix multiplication: $\frac{\partial L}{\partial Y} \times W$ 
     
2.a. The shape of the jacobian is '(N = 64, out_features = 512, out_features = 512, in_features = 1024)'

2.b. yes, this matrix is sparse. because $\forall k\neq j:\frac{\partial Y_{(i, j)}}{\partial W_{(k, y)}} = 0$, 
     and this is because each neuron is only effected by hhis owm weights.
     
2.c. There is no need to materialize the jacobian,  we can calculate $\frac{\partial L}{\partial Y}$
     by performing matrix multiplication: $\frac{\partial L}{\partial Y}^T \times X$ 




"""

part1_q2 = r"""
**Your answer:**


Yes we can! we can train a neural network without back propagation. we can expand the expression of the loss as a function of each o our
parameters, deriviate it analiticly by hand and so find the gradient with respept to each parameter. 

However, this approach is very hard to implement. And would require redoing the calculations every time we change the networking.
Making nerual networks withour backprop not very feasible.
 

"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    lr = 0.01
    reg = 0.1
    wstd = 0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 1
    lr_vanilla = 0.0000002
    lr_momentum = 0.0001
    lr_rmsprop = 0.00001
    reg = 0.000002
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lt = 0.001
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**

Yes, while training with cross entropy loss it is possible for the loss to increase for a few epochs
while the test accuracy also icreases.

And a example of when might this happen is when there is a spesific difficult or misslabled sample in the
test dataset, which the model predicts to be of the worng class with rising confidency, while also
improving on most other sample predictions.
This will cause the loss to expload (because cross entropy heavly punished confident wrong prediction)
while the overall accuracy would still improve in the testset.

"""

part2_q3 = r"""
**Your answer:**

1. Back propagation is a method for calculating derivatives of loss with respect to different parameters
   in a deep network and isn't an optimization method. GD is an optimization method that uses the derivative
   with regard to each parameter reach a minmum of a given loss function, and isn't a
   method for calculating derivatives.

2. GD calculets the gradient over the entire dataset and updates the parameters only once.
   SGD calculets the gradient over a single sample (or a mini batch in mini-batch SGD) and updates the parametrs only once.
   GD makes very calculated and slow moves down the loss surface as deffined over the dataset.
   SGD makes quicker to calculate moves down an aproximation of the loss surfaces of a different example (or mini batch) each time.

3. Becuase SGD is so much faster at preforming each step, it will converge in much shorter time
   allthough in more steps. SGD also preforms a ort of regularization, by fitting the model only
   to a small subset of the dataset instead of the whole thing. Often it is not possible to fit the entire
   dataset into memory, in those cases running GD is not feasible and SGD solves that problem.
   
4. 
"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    hidden_dims = 200
    activation = 'relu'
    out_activation = 'sigmoid'
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.00005
    weight_decay = 0.00002
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    lr = 0.003
    weight_decay = 0.00001
    momentum = 0.98
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""