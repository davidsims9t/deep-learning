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

One epoch is when you go through an entire data set.

## Gradient Descent

Curse of dimensionality - even on the worlds fastest computer it takes a huge amount of time to find the best optimization.

Look at angle of cost function at a point on the grid (the ball), then roll the ball to another point on the grid and repeat until the ball is in the middle. The method ends up being a zig-zag.

## Stochastic Gradient Descent

Happens when you don't have an optimized neural network. Helps you find the global minimum when there's more fluxuations. It's faster because it doesn't have to load all data into memory.

Batch gradient descent will give you the same result each time, stochastic is chosen at random.

Mini-batch gradient descent - combination of both methods. Does batch stochastic processes.

## Books & Links

Efficient BackProp - Normalization
Cross Validated (2015) - Cost functions
A Neural Network in 13 lines of Python
Neural Networks and Deep Learning
