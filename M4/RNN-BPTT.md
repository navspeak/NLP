```
z_t = x_t @ Wx.T + h_prev @ Wh.T + b
h_t = tanh(z_t)
```
We will compute:
- forward pass
- assume a gradient comes from above
- compute dz
- then compute dWx, dWh, db, and dh_prev

1. Given value:
```python
x_t = [[1, 0, 2]]          # shape (1,3)
h_prev = [[0.5, -0.3]]     # shape (1,2)

Wx = [
 [0.2, 0.1, 0.4],
 [0.5, 0.3, 0.2]
]                          # shape (2,3)

Wh = [
 [0.1, 0.2],
 [0.3, 0.4]
]                          # shape (2,2)

b = [[0.1, 0.1]]           # shape (1,2)
```
2. Forward Pass:
    1. Input Contribution: `x_t@W.T`
    ```
    [[1,0,2]] @ [[0.2,0.5],
                [0.1,0.3],
                [0.4,0.2]]   = [[1.0, 0.9]]
    ```
    2. Recurrent contribution:`h_prev @ Wh.T`
    ```
    [[0.5, -0.3]] @ [[0.1,0.3],
                     [0.2,0.4]] = [[-0.01, 0.03]]
    ```
    3. Add Bias
    ```
    z_t = [[1.0, 0.9]] + [[-0.01, 0.03]] + [[0.1, 0.1]]
        = [[1.09, 1.03]]
    ```
    4. Apply tanh: `h_t = tanh(z_t) =  [[0.797, 0.774]]`

3. Assume gradient from next layer / loss
- Now suppose the loss sends this gradient into h_t: `dh_t = [[0.2, -0.1]]`
- This means:
    - loss wants first hidden value to increase a bit
    - loss wants second hidden value to decrease a bit

4. Backprop through tanh
- Since: `h_t = tanh(z_t)` we use: $\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z)$
- so, `dz_t = dh_t * (1-h_t**2)`
- First compute:
```
h_t**2 ≈ [[0.797^2, 0.774^2]]
       ≈ [[0.635, 0.599]]

Then:

1 - h_t**2 ≈ [[0.365, 0.401]]

Now elementwise multiply with dh_t:

dz_t = [[0.2, -0.1]] * [[0.365, 0.401]]
    ≈ [[0.073, -0.0401]]

So: dz_t ≈ [[0.073, -0.0401]]
```
This is the gradient at the pre-activation z_t.

5. Gradient of bias



