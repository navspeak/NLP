# Recurrent Neural Network (RNN) Explained with Simple Vectors and Matrices

## Core Idea of RNN

A normal neural network only looks at the **current input**.

An **RNN** also remembers the **previous hidden state**.

At each time step:

* Current input = `x_t`
* Previous memory = `h_(t-1)`
* New memory = `h_t`

---

## Formula

At time step `t`:

```python
h_t = tanh(x_t @ Wx.T + h_prev @ Wh.T + b)
```
- Wx = weights for the current input x_t
- Wh = weights for the previous hidden state / memory h_prev

---

## Example Shapes

Suppose:

* Input size = 3 => has 3 features
* Hidden size = 2

### Input vector

```python
x_t = [1, 0, 2]
x_t.shape = (1,3) = (batch_size, input_size)
```
- for an RNN with hidden size = H=> Wx.shape = (H, input_size)
### Previous hidden state

```python
h_prev = [0.5, -0.3]
h_prev.shape = (1,2)
```

### Weights
- Wx = weights for the current input x_t
- Wh = weights for the previous hidden state / memory h_prev

```python
Wx.shape = (2,3)
Wh.shape = (2,2)
b.shape  = (1,2)
```

---

## Example Values

```python
x_t = [[1, 0, 2]]

h_prev = [[0.5, -0.3]]

Wx = [
 [0.2, 0.1, 0.4],
 [0.5, 0.3, 0.2]
]

Wh = [
 [0.1, 0.2],
 [0.3, 0.4]
]

b = [[0.1, 0.1]]
```

---

## Step 1: Input Contribution

```python
x_t @ Wx.T
[1, 0, 2].[[0.2, 0.5],
           [0.1, 0.3]
           [0.4, 0.2]]
= [1*0.2+0+2*0.4, 1*0.5+0+2*0.2]    = =[[1.0,0.9]]       

```

---

## Step 2: Memory Contribution

```python
h_prev @ Wh.T
[[0.5, -0.3]] @ [[0.1,0.3]]
                 [0.2, 0.4]

= [[-0.01, 0.03]]
```

---

## Step 3: Add Bias

```python
z = [[1.0,0.9]] + [[-0.01,0.03]] + [[0.1,0.1]]
  = [[1.09,1.03]]
```

---

## Step 4: Activation

```python
h_t = tanh(z)
=[[0.797, 0.774]]
```

This becomes the **new memory**.

---

## Next Time Step

```python
h_prev = h_t
```

The model carries memory forward.

---

## Mental Picture

```text
x_t ----\\
         +--> h_t --> next step
h_prev --/
```

Current input + old memory = new memory.

---

## Full Python Example

```python
import numpy as np

x = np.array([[1,0,2]])
h_prev = np.array([[0.5,-0.3]])

Wx = np.array([
 [0.2,0.1,0.4],
 [0.5,0.3,0.2]
])

Wh = np.array([
 [0.1,0.2],
 [0.3,0.4]
])

b = np.array([[0.1,0.1]])

z = x @ Wx.T + h_prev @ Wh.T + b
h = np.tanh(z)

print(h)
```

---

## Why RNNs Matter

Useful for sequences:

* Text
* Speech
* Time series
* Signals

---

## Why They Were Later Replaced

Limitations:

* Forget long history
* Vanishing gradients
* Slow sequential training

Successors:

* LSTM
* GRU
* Transformers

---

## One-Line Summary

```python
new_memory = f(current_input + old_memory)
```

An RNN is a neural network with memory.
