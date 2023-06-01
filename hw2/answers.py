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
    hidden_dims = 35
    activation = 'tanh'
    out_activation = 'none'
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
    lr = 0.0025
    weight_decay = 0.001
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1. The model is qualitativly close to the optimal on our defined learning objective. We can see that the learing
   is at a plateau on the last few epochs meaning we got into a local minimum, and from our expirmentaions with hp
   we found that the best preformance on this data set wasn't much higher.
   All this to say that the model is infact fairly close to the overall best model giving our optimization goals
   which means the optimization error isn't very high.

2. During the learning process, the training set shows consistent improvement, whereas the test set 
   exhibits unstable progress (has "jumps" in the beginning), indicating potential generalization issues in our model. 
   however, when comparing the final results on both sets, the model's accuracy on the training
   set is only 2-2.5% higher than that on the test set, indicating relatively good generalization. In conclusion,
   despite the uneven improvement of the test set, our model demonstrates similar performance on both the 
   test and training sets, showing that the generalization error is not significant.
   
3. From our knowledge of the distribution, it can be well approximated by an mlp model of our chosen hidden dims.
   This is because the distribution is nearly seperable by a simple curve. more over, MLPs are highly expressive
   as we saw in the tutorials. Therefor, we can say that the approximation error in not significant.  

"""

part3_q2 = r"""
**Your answer:**
We would have expected the confusion matrix to be well balanced (FPR close to FNR) because
the shift from the train ds to the validation ds is symetrical around the origin and the same
amount of False Positives would be added as of False Negatives.
"""

part3_q3 = r"""
**Your answer:**
In both cases knowledge of the problem and "cost" of mistakes changes the threshold we choose.
1. In the first scenario, FP are expansive to send into farther examinations, while FN are quit
    allright because the person isn't in serious danger. For this case we would choose higher 
    threshold to classify more FN and less FP.
2. In the second scenario, FN are really costly as they would cost someones life. So for this case
    we would rather get more FP over FN. So we would set a lower threshold to classfy more FP then FN.
In both cases, setting a reasonable threshold (not 0 or 1) is still important, otherwise the model 
is usless, and there is a need to find a good compromise taking into acount the adjusted "costs".
"""


part3_q4 = r"""
**Your answer:**
General note/ananlysis: We run the expiraments a few times with diffrenet HP, and found the TANH nonlin preforms very well,
we think because of that maybe some of the subtle diffrences between different architectors might have been lost.

1. Increasing the width of the model seems to make it more expressive, which sometimes can cause over fitting: for deeper
   models the wider netoworks altough with more detailed decision boundries have lower test accuracies. for more shallow
   models the expresive power increase helps with test set preformance. Increasing the width makes for more specialized
   decision boundries but not allways better accuracies.

2. We can see that deeper models also have higher exprasive power, and again we can see that on nearo networks the
   test set accuracies improve with depth, however on wider MPLs making them deepr causes overfitting, making for worst
   test/validation accuracies.
   In summary, increasing the depth makes for more specialized decision boundries but not allways better accuracies.

3. Both the 1 deep 32 width and 4 deep 8 width seem to yeild close decision boundries, and similar result on the test set,
   however on the validation set the deeper model preforms 5% worst. We think that because of the tanH non line which alows
   for easy exprasion of the cruve of the dataset with only one hidden layer, making the wide network more efficient slightly.

4. Generally there is not a single trend in the relation of the validation accuracy (before threshold selection) to the test
   accuract (after threshold selection), sometimes the validation is better than the test and sometimes it's the other way
   around, alltough in most cases there is an improvment (better threshold for genralization actully helps somewhat).
   An intersting trend we did find was that threshold selction had a stablizing effect on the test accuracy:
   While the validation accuracy veries drasticly (~20% diffrence) between our model the test accuracy varies less
   even with very different archtectures (only ~8%). We think this improvment/stablization is because by choosing the threshold on the
   validation we can correct some baises in the training set by moving slightly the boundries for better genralization.
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
    lr = 0.01
    weight_decay = 0.0002
    momentum = 0.9
    loss_fn = torch.nn.CrossEntropyLoss()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
1. As we saw in the tutorials the formula for number of paramters are: $K \cdot (C_{in}F^2 + 1)$
   $parameter count for normal residual block = 2 * 256 (256 \cdot 3^2 + 1) = 1180160$
   $parameter count for bottelneck = 64 (256 \cdot 1 + 1) + 64 (64 \cdot 3^2 + 1) + 256 (64 \cdot 1 + 1) = 70016$

2. We can get a good estimation to the number of flops as $W \cdot H \cdot N \cdot Parameters_in_conv_layears$.
   This is because for each sample, height, and width, there would be around 1 multiplication with each parameter
   in the layers to calculate the output. This means as per our last answer that the bottelneck block would require
   around $70,000 \cdot N \cdot H \cdot W$ flops while the normal resblock around $1,180,000 \cdot N \cdot H \cdot W$ flops. N representing num of samples,
   W is the image Width and H is the image Height (which are constant throughout the resblocks).

3. (1) The bottle neck is worse at combining input spatially since the respetive view of each pixle is only 3x3, compared to
   5x5 recptive field in the regular resblock. (2) The bottle neck can have input featuremaps  effecting across all output feature 
   maps at each conv layer, because of the nature of the convolution operation itself, matching the exprassive ness of normal resblock.
   Overall the bottleneck block in the diagram is less exprasive also because it has only 1 actual convolution with spatial information
   and because each of the 256 output channels is a linear combination of only 64 convalution results.
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