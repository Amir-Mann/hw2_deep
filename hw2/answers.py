r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**


1.a. The shape of the jacobian is '(N = 64, in_features = 1024, N = 64, out_features = 512)' 

1.b. Yes, this matrix is sparse. because $\forall j\neq i:\frac{\partial y_i}{\partial x_j} = 0$, 
     and this is because we treat each sample independently from the rest of the batch.
     
1.c. There is no need to materialize the jacobian, we can calculate $\frac{\partial L}{\partial X}$
     by performing matrix multiplication: $\frac{\partial L}{\partial Y} \times W$ 
     
2.a. The shape of the jacobian is '(N = 64, out_features = 512, out_features = 512, in_features = 1024)'

2.b. Yes, this matrix is sparse. because $\forall k\neq j:\frac{\partial Y_{(i, j)}}{\partial W_{(k, y)}} = 0$, 
     and this is because each neuron is only effected by his owm weights.
     
2.c. There is no need to materialize the jacobian,  we can calculate $\frac{\partial L}{\partial Y}$
     by performing matrix multiplication: $\frac{\partial L}{\partial Y}^T \times X$ 
"""

part1_q2 = r"""
**Your answer:**

Yes we can! we can train a neural network without back propagation. we can expand the expression of the loss as a function of each o our
parameters, deriviate it analiticly by hand and so find the gradient with respept to each parameter. 

However, this approach is very hard to implement. And would require redoing the calculations every time we change the networking.
Making nerual networks without backprop not very feasible.
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

And an example of when might this happen is when there is a spesific difficult or misslabled sample in the
test dataset, which the model predicts to be of the worng class with rising confidency, while also
improving on most other sample predictions.
This will cause the loss to expload (because cross entropy heavly punished confident wrong prediction)
while the overall accuracy would still improve in the testset.

"""

part2_q3 = r"""
**Your answer:**

1. Back propagation is a method for calculating derivatives of loss with respect to different parameters
   in a deep network and isn't an optimization method. GD is an optimization method that uses the derivative
   with regard to each parameter in order to reach a minmum of a given loss function, and isn't a
   method for calculating derivatives.

2. GD calculets the gradient over the entire dataset and updates the parameters only once.
   SGD calculets the gradient over each sample (or a mini batch in mini-batch SGD) and updates the parametrs every time.
   GD makes very calculated and slow steps down the loss surface as deffined over the dataset.
   SGD makes quicker to calculate steps down an aproximation of the loss surfaces, 
   using the loss of over a single, different sample (or mini batch) each time.

3. Becuase SGD is so much faster at preforming each step, it will converge in much shorter time
   allthough in more steps. SGD also preforms a sort of regularization, by fitting the model only
   to a small subset of the dataset instead of the whole thing. Often it is not possible to fit the entire
   dataset into memory, in those cases running GD is not feasible and SGD solves that problem.
   
4. AODIFSUASOD*UV*ADSUVOADSUVOICXUVO*AUVO(AUEW)
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
   This is because the distribution is best seperated by a simple curve. more over, MLPs are highly expressive
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
1. The depths has a varied affect on accuracy in this settings, while deeper model are more expressive
   and complex, they are also harder to train. We can see that in our case the depth=4 model preformed
   the best, while the second place was different in depening on the width, but generaly we saw that the
   depth=8 model didn't cause early stopping - because the optimization is a very hard task.
   We think that for that reason 4 is a good comprimaize for depth.

2. Yes, the networks of depth=16 were not trainable. This is due to the exploding/vanishing gradient
   phenomenas, after 16 gradients the loss function has very unpredictable effect on the first weight
   making the task of learning them imposible. We might resolve this problem at least partialy by:
   a. Adding skip conections between every few conv layers, (resblocks) this would cause the loss derivative
      to directly effect the first layers of the network.
   b. Adding Batch Normalization layers in the network can lower the covarient shift, and thus lessening
      the problem with unscaled multiplicative gradients at the first layers of the network.
"""

part5_q2 = r"""
**Your answer:**
We can see a few interesting trends in the second experiment. First of all we can see that the wider the
network is the better interms of preformance both on test and train datasets and at any depth. This is
very different to the mixed results in the first experiment regarding depth. Secondly we can see that
wider netoworks tend to train in less epochs and reach early stopping faster. We are not certain that 
this means actuall faster training phase, because each apoch requires more calculations due to around
quedratic amount of parameters and flops.
Lastly, we can see that the 128-wide networks which where only trained in this experiment out preform the
networks of the first experiment, reaching 75% accuracy on test set. Around 3-5% more then the best ones
there.
"""

part5_q3 = r"""
**Your answer:**
In this experiment we can see that the deeper the network (up to the given bound of 8 layer) is the better
it preforms on the test set, while all the network can achive overfitting and are early stopped to prevent
this.
We can see a werid phenomana is that the network with 6(L=3) layers is trained the fastest, and there is not a
single trend correltion. We think this is because while the 4(L=2) layers network is not expressive enough,
it keeps on learning intill the very last epoch, the 8(L=4) layers network is very deep and hard to optimize
over and therefor takes more time to start overfitting compared to the 6(L=3) layers one.
This architecture of less features in the begining and more features in the last layers out preforms the
homogeneous one in the previous experiment, reaching 78% accuracy in the best one on the test set.
"""

part5_q4 = r"""
**Your answer:**
Almost on all datasets we saw that the shorter the network the better it preforms. (to put it in context - 
the shalowest networks we trained here are about the same length as the deepest trainable networks in previous experiments)
This has an exception on the test set in the second experiment where the 12 layers out preformed the 6 layers.
So it seems there is a sweet spot at around 8 - 12 layers deep for this classification over cifar-10.

comparison to experiment 1.1:
The first experment is very much comparable to the first run group of this experiment - fixed K, L varied.
We can see that using batch norm and skip cnnections we are able to train 16 and even 32 deep networks. 
We can also see that 8-deep 32-wide network outpreforms by 6% on the test set, when comparing the resnet to 
the cnn architectures.

comparison to experiment 1.3:
In the third experment we saw that the deeper the network is - the better it preformes. Here, we can see
the mirror image of this trend, after 12 layers deep the network starts fall in prformance. 
Also we can see that the models here reach about 2% higher accuracy. Finally we can say that using 3 
different channels amounts has not significantly improved preformance. This is in contrast to the huge
preformance improvement from homogeneous channel amount to 2 phased channel amount.
"""

part5_q5 = r""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**
1. The model preformed quite bad: it has big misclasifies in both images.
   In the first one the bounding boxes are pretty well but it classifies three dolphines as two persons
   and a surfboard. In the second one the bounding boxes are not well aligned for the correctly classified
   dog. A cat in the picture is not detected as a seperate object, but the clasification
   of two dogs who are next to the cat are labeled "cat" falsly.

2. The model seems to fail because of very dark silhouettes, abscuring the detail of the dolphines, and
   because of heavy relyence on contex for inferring. Also the model fails becuase there are a banch of
   objects to the detect overlapping eachother.
   To improve that, we might try brightining very dark spots in images, bluring the background of the image,
   we can try and run the model on smaller segments of the clottered image.
   We may try to train the model with less context - filters bluring backgrounds maybe.
   Try and train the model with a heavy loss for detecting the wrong amount of bounding boxes in the image.
   Using attenion to find the areas most importent for the classifcation and making smaller bounding boxes
   around them / making sure each area is only effacting 1 boxes label. Giving stronger weight at the center
   of each bounding box after decision boundries where choosen and reclassifing. Maybe sending each cropped
   bounding box into a new different classifier.
   
"""


part6_q2 = r"""
**Your answer:**

"""


part6_q3 = r"""
**Your answer:**
First Image - Toothbrushes: The model missclassfies the pens and other writing tools as all toothbrushes.
This could be explained by clutterdeness - alot of vering objects in the same space over lapping each other.
In addition the model might not know and wasn't trained on some of the objects like the rooler.

Second image - Apples: The model missclassifies a robber duck as two appels.
This could be explained by model bias - when it sees red and yellow object surrounded by leaves it guess its
an apple, because probably all the examples in his train dataset of red objects between leaves where appels.

Third image - A mouse, a dog and a bear: The model misclassifies all the objects in the image, a coin and
two toy (lamas), probably because it is not familiar with these objects, how ever maybe the cropped coin
is misslabeled due to occlusion.
The biggest mistake the model makes here is giving the half lama two misslabels in about the same bounding
box, calling it a dog and a bear intertwined. This is most likly due to occlusion, making the recognition
of the object very hard for it.
"""

part6_bonus = r"""
**Your answer:**
Not done (yet?)
"""