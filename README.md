### The FFA Algorithm

In [The Forward-Forward Algorithm: Some Preliminary
Investigations](https://www.cs.toronto.edu/~hinton/FFA13.pdf) - Geoffrey 
Hinton proposes an alternative to backpropagation, called the 
Forward-Forward Algorithm.

In FFA - the input is passed twice through the network in a 
forward-fashion, while no backward pass happens. Instead of a loss 
function, a "goodness" function is used to evaluate the predictions of the 
network. One of the forward passes is a "positive" pass, while the other 
is the "negative" pass.

- The positive pass aims to maximize the goodness on real data (actual 
input features and labels)
- The negative pass aims to minimize the goodness on fake data (corrupt 
input features and wrong labels)

While many goodness measures can be used, the paper proposes a sum of the 
squares of activations of intermediate ReLUs and a negative sum of the 
squares of activations of intermediate ReLUs.

The way data is corrupted for the negative pass is also variable - there 
are many ways to corrupt data. Creating masks (and inverted masks) and 
mixing up training data by multiplying two input feature sets with the 
masks and combining them into a hybrid image are used.

The following is a minimal TensorFlow-based implementation of the 
Forward-Forward Algorithm.

**Note:** The implementation is a work-in-progress and will periodically 
be updated. It's released now to (hopefully) start a discussion in this 
lane.

### TODO:

- Creating hybrid negative data (i.e. 
tf.reduce_sum(masks*input_image_pairs))
- Accounting for negative data
- Figuring out how to extract intermediate activations in TF


### Why Try To Move from Backpropagation?

Backpropagation is great! Though, it doesn't seem like it's what's 
happening in the brain, in which a much faster operation is taking place. 
In a classical deep learning workflow, we train networks in three basic 
steps:

- Forward pass
- Backward pass
- Updating weights

During the forward pass and backward pass, *no learning is hapenning*. 
Additionally, backpropagation requires the knowledge of the structure of 
the network during the forward pass, and is tricky for physical 
implementation[1](https://www.nature.com/articles/s41467-022-35216-2).

Moving away from backpropagation is likely to allow us to train networks 
without knowledge of the structure (or physical system), and potentially 
with higher efficiency.

### FFA Doesn't Replace Backpropagation

FFA, while a great step towards a broader solution, doesn't currently 
replace backprop and isn't meant to replace it. Some limitations and 
unknowns include:

- Doesn't work with networks that make use of weight-sharing (such as 
CNNs)
- Isn't clear how well it scales
- Requires potentially expensive creation of negative data
- Isn't faster than backprop (because two forward-passes are done)
- What makes for a good "goodness" function? Which activation functions 
yield the best results?


Further announced work by Geoffrey Hinton will tackle these questions. In 
the meantime, making the research accessible, and *inciting discussions* 
can help possibly derive further methods inspired by FFA.
