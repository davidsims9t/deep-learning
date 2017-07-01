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

## Books

Efficient BackProp - Normalization
