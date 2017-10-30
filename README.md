# Deep Learning

## Neuron (Node)

Takes an input layer (values) and produces an output. Signals come from other input neurons.
Inputs are independent variables (descriptors of a particular object). Need to be standardized or normalized first.

The neural takes the weighted sum of the synapses. Theta is the threshold function.

```
ϕ(Σ sup m, sub i = 1, w sub i, x sub i)
```

## Synapse

The signals sent from input neurons. Are assigned certain weights. Used to determine which signals are important to the neural network. Get adjusted across the neural network.

## Output

Can be category or a single value.

## Sigmoid Function

```
ϕ(x) = (1 / 1 + e ^ -x)
```

## Rectifier Function

```
ϕ(x) = max(x, 0)
```

## Hyperbolic Tangent Function

```
ϕ(x) = (1 - e ^ -2x) / (1 - e ^ -2x)
```

## How do Neural Networks Work

y = actual value
ŷ = output value (predicted value)

Perceptron = simple neural network

## Back Propagation

C = Σ.5(ŷ - y) ^ 2

Cost function determines what the error is between the predicted value and the actual value.

One epoch is when you go through an entire data set. Adjusts all the weights at the same time.

## Gradient Descent

Curse of dimensionality - even on the worlds fastest computer it takes a huge amount of time to find the best optimization.

Look at angle of cost function at a point on the grid (the ball), then roll the ball to another point on the grid and repeat until the ball is in the middle. The method ends up being a zig-zag.

### Stochastic Gradient Descent

Happens when you don't have an optimized neural network. Helps you find the global minimum when there's more fluxuations. It's faster because it doesn't have to load all data into memory.

Batch gradient descent will give you the same result each time, stochastic is chosen at random.

Mini-batch gradient descent - combination of both methods. Does batch stochastic processes.

#### Steps

1. Randomly assign weights close to 0, but greater than 0.
2. Input the first observation of your dataset in the input layer, each layer is one input node.
3. Forward-propagation: from left-to-right, the neurons are activated in a way that impact of each neuron's activation is limited by the weights. Propagate the activations until getting the predicted result y.
4. Compare predicted result against the actual result and measure the generated error.
5. Back-propagation: from right-to-left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights.
6. Repeat steps 1 - 5 and update the weights after each observation (reinforcement learning) or repeat steps 1 - 5 and only update the weights after a batch of observations (batch learning).
7. When the whole training set has passed thru the entire neural network (ANN), that makes an epoch.

### The Bias-Variance Trade-off

Variance trade-off happens when there's a lot of variance in the results and because of that each time you train your neural network, you can get varying results of accuracy. k-Fold Cross Validation seeks to fix this issue. It takes the training set and splits it into 10 iterations with 10 folds.
You can take an average or standard deviation of the iterations and compare them.

Use dropout when you have high variance and overfitting so that the neurons can run more independently of each other. Drop out will randomly disable neurons during each iteration.

## Convolutional Neural Networks

You have an input image, process it through a CNN, and Output Label (Image Class).

Example

:) -> CNN -> Happy

B/W Image 2px x 2px - 2d array. Each bit is between 0 and 255 for the pixel color.
Colored Image 2px x 2px - 3d array. 3 layers (RGB) for each pixel color.  

### Convolutional Operation

(f * g)(t)^def=∫+∞-∞f(T)g(t - T)dT

Input Image (X) Feature Detector = Feature Map

To calculate the feature map compare the matching pixels in the image based on a 3 x 3 pixel grid. If any pixel matches up then add 1 to the total sum.

The feature map allows the computer to isolate the image based on the images defining features (nose, eyes, etc). Gets rid of unimportant data. Each feature map is a layer of feature data with a different feature detector.

Primary goal of the CNN is to isolate certain features.

### ReLU Layer

Rectifier acts as a filter to break-up linearity. If a color is a linear gradient (e.g. white to black) the black is removed by the Rectifier layer.

### Max Pooling

The ability to recognize an object with different features (textures, patterns, lighting, etc).
Special variance means it doesn't care if the features are a bit different (distorted) from each other.

To get the pooled feature map from the feature map, you have to take the maximum value from the array.

Sub-sampling is a generalization.

### Flattening

Flatten a pooled feature map (row by row into a column). This can easily be translated to a vector line.

### Full Connection

Add a whole ANN to the CNN. Hidden layers don't have to be fully connected in an ANN, but with a CNN they have to be fully connected. We have two outputs because we need an output per category that we have (i.e. dog and cat). Loss function tells us how well our network is working. Weights are adjusted and feature detectors in an ANN to minimize loss/error.

Weight is between 0 and 1. 0 being it didn't find the feature and 1 being that it did find the feature in the CNN. The more neurons that are attributed (lighting up) to the dog neuron, the more the score increases.

During back-propagation, if a feature is useless to training a network, it is disregarded.

In the final connected neural layer, the neurons get to vote which category it thinks is accurate.

### Softmax and Cross-Entropy

## Recurrent Neural Networks

Parts of Brain:

- Cerebrum
- Cerebellum

Lobes:

- Frontal Lobe - RNN (short-term memory)
- Parietal Lobe - Spatial coordination system
- Temporal Lobe - ANN (long-term memory)
- Occipital Lobe - CNN (vision)

Short-term memory is the ability to know what kind of information was stored in the previous
neural network and pass that information into the future. You need short-term memory in order to know the context of what's happening now.

### One-to-Many

The ability to make multiple connections from one input.

Example: black and white dog jumps over a bar.

### Many-to-One

The ability to take many inputs and get one output.

Example: sentiment analysis.

### Many-to-Many

The ability to take many inputs and get one output.

Example: Google translator.

### Vanishing Gradient

As the data passes through the neurons and is trained the gradient descent can be large or small.

If Wrec (weights) is small, then you have a vanishing gradient problem.
If Wrec (weights) is large, then you have an exploding gradient problem.

#### Solutions

1. Exploding Gradient
  - Truncated Back-propagation
  - Penalties
  - Gradient Clipping
2. Vanishing Gradient
  - Weight Initialization
  - Echo State Networks
  - Long Short-Term Memory Networks (LSTMs)

### LSTMs

If Wrec is 1 the vanishing gradient problem is solved. Has a memory cell that has a free flow
of information. No issues with back-propagation. Any line in the LSTM is a vector.

- Vector transfer is a simple transfer of vector data.
- Concatenation mean that there are two pipes running in parallel to each other.
- Copy is splitting a vector two different locations.
- Point-wise Operation

## Self-Organizing Maps (SOMs)

Supervised deep-learning methods are ANN, CNN, and RNNs. Unsupervised learning includes SOMs, Deep Boltzmann Machines, and AutoEncoders.

SOMs are useful for feature detection. Useful for reducing dimensionality. Reduces columns into a 2D map.

### K-Means Clustering

Allows you to categorize groups.

Steps:

1. Choose the number of clusters
2. Select at random k points (centroids).
3. Assign each data point to each the closest centroids.
4. Compute and place the new centroids in the clusters.
5. Reassign each data point to the new closest centroid. If reassignment took place repeat step 4.

Best matching unit is the unit with the calculated weight that is closest value.
Weights are updated based on the BMU.

### How to SOMs Learn?

- SOMs maintain the topology of the input set.
- SOMs reveal correlations that are not easily defined.
- SOMs classify data without supervision.
- No target vector or back propagation required
- No lateral connections between output nodes


## Libraries

- Theano - computations library (CPU and GPU).
- Tensorflow - computations library.
- Keras - wraps Theano and Tensorflow libraries.
- SciPy/NumPy - for preprocessing data
- Pandas - For reading CSV

## Books & Links

- Efficient BackProp - Normalization
- Cross Validated (2015) - Cost functions
- A Neural Network in 13 lines of Python
- Neural Networks and Deep Learning
- Introduction to Computational Neural Networks - Jianxin Wu
- Understanding Convolutional Neural Networks with a Mathematical Model - C.C. Jay Kuo
- Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.
- Evaluation of Operations in Convolutional Architectures for Object Recognition
- The 9 Deep Learning Papers You Need to Know About CNNs
- On the difficulty of training recurrent neural networks
- http://colah.github.io/
- http://karpathy.github.io/
- Kohohen's Self Organizing Feature Maps (http://www.ai-junkie.com/ann/som/som1.html)
