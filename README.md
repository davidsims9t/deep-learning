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

## Libraries

Theano - computations library (CPU and GPU).
Tensorflow - computations library.
Keras - wraps Theano and Tensorflow libraries.

## Books & Links

Efficient BackProp - Normalization
Cross Validated (2015) - Cost functions
A Neural Network in 13 lines of Python
Neural Networks and Deep Learning
