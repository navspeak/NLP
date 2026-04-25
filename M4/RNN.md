# RNN — Hidden State Vector Computation
- RNN is a normal neural network looks only at the current input.
- An RNN also remembers the previous hidden state.
- So at each time step:
    - current input = x_t
    - previous memory = h_(t-1)
    - new memory = h_t

- classic RNN equation: `h_t = tanh(x_t @ Wx.T + h_prev @ Wh.T + b)`

## Unrolled RNN Diagram

```
      J(1,θ)         J(2,θ)         J(t,θ)
        ↑               ↑               ↑
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│h(1)           │ │h(2)           │ │h(t)           │
│               │ │               │ │               │
│   ●   ●   ●  │─Wh→  ●   ●   ●  │─Wh→  ●   ●   ●  │
│               │ │               │ │               │
└───────────────┘ └───────────────┘ └───────────────┘
        ↑               ↑               ↑
       x(1)            x(2)            x(t)
```

### What each part means

```
┌───────────────┐
│h(t)           │  ← hidden state label (top left)
│               │
│   ●   ●   ●  │  ← red circles = neurons in the hidden state vector
│               │     each ● holds one value of h_t = [h1, h2, h3]
└───────────────┘

─Wh→               ← same recurrent weight matrix passed between every box
                      this is the weight symmetry — one W shared across all t

↑ x(t)             ← input at this time step feeds UP into the box

↑ J(t,θ)           ← loss at this time step flows UP out of the box
                      θ = all model parameters (W_xh, W_hh, b)
                      J(t,θ) = how wrong the output was at step t
```

### Why J(t,θ) at every step?

In many RNN tasks (e.g. language model predicting next word), there is a loss
at **every** time step — not just the last one:

```
Total loss J(θ) = J(1,θ) + J(2,θ) + ... + J(t,θ)
```

Backprop sums gradients from all time steps and flows them **back through Wh**
— this is **Backpropagation Through Time (BPTT)**.

```
Gradient flow (backwards):

J(1,θ) ←─── J(2,θ) ←─── J(t,θ)
   ↓    Wh      ↓    Wh      ↓
  h(1) ◄────  h(2) ◄────  h(t)
```

The further back in time, the more times the gradient passes through Wh —
this is exactly where vanishing/exploding gradient comes from.

## The Formula

```
h_t = tanh(W_hh · h_(t-1) + W_xh · x_t + b)
```

Three things combine to produce the new hidden state:

```
x_t       — current input (e.g. current word, current game frame)
h_(t-1)   — previous hidden state (memory from last step)
W_xh      — weights for input
W_hh      — weights for hidden state (the recurrent connection)
b         — bias
tanh      — squishes output to (-1, 1)
```

---

## Step by Step — Vector Maths

```
Setup: input_size=2, hidden_size=3

x_t     = [0.5, 0.8]          ← current input,        shape (2,)
h_(t-1) = [0.1, 0.2, 0.3]     ← previous hidden state, shape (3,)
```

### W_xh · x_t  — input contribution

W_xh shape is (3, 2) — 3 hidden neurons, each with 2 weights for the input:

```
W_xh = [[ 0.4,  0.6],     ← neuron 1's weights for x
         [-0.3,  0.8],     ← neuron 2's weights for x
         [ 0.1, -0.5]]     ← neuron 3's weights for x

W_xh · x_t = [[ 0.4,  0.6],   [0.5]   =  [0.4×0.5 + 0.6×0.8]   =  [0.68]
               [-0.3,  0.8], × [0.8]      [-0.3×0.5 + 0.8×0.8]      [0.49]
               [ 0.1, -0.5]]              [0.1×0.5 + -0.5×0.8]       [-0.35]
```

### W_hh · h_(t-1)  — memory contribution

W_hh shape is (3, 3) — 3 hidden neurons, each with 3 weights for the previous hidden state:

```
W_hh = [[ 0.2, -0.1,  0.4],
         [ 0.3,  0.5, -0.2],
         [-0.1,  0.2,  0.3]]

W_hh · h_(t-1) = W_hh · [0.1, 0.2, 0.3]

               = [0.2×0.1 + -0.1×0.2 + 0.4×0.3]   =  [0.12]
                 [0.3×0.1 +  0.5×0.2 + -0.2×0.3]      [0.07]
                 [-0.1×0.1 + 0.2×0.2 +  0.3×0.3]      [0.12]
```

### Add + bias

```
b = [0.1, 0.1, 0.1]

z = W_xh·x_t  +  W_hh·h_(t-1)  +  b
  = [0.68,      [0.12,            [0.1,     [0.90]
     0.49,   +   0.07,        +    0.1,  =   0.66]
    -0.35]       0.12]             0.1]     [-0.13]
```

### Apply tanh element-wise

```
h_t = tanh([0.90, 0.66, -0.13])
        =  [0.716, 0.578, -0.129]    ← new hidden state, shape (3,)
```

This `h_t` now becomes `h_(t-1)` for the next time step.

---

## Visually

```
         x_t [0.5, 0.8]
              │
              │ × W_xh
              ↓
    ┌─────────+─────────┐
    │                   │
h_(t-1) ──×W_hh──→    [+]  ──→ tanh ──→ h_t
    │                   │
    └───────────────────┘
              ↑
             +b

h_t then becomes h_(t-1) for the NEXT step
```

---

## Why tanh?

```
Without tanh:  values grow unboundedly → exploding numbers after many steps
With tanh:     values stay in (-1, 1)  → stable across long sequences

tanh(-3) = -0.995   (saturates at -1)
tanh(0)  =  0.0
tanh(3)  =  0.995   (saturates at +1)
```

---

## In PyTorch

```python
rnn = nn.RNN(input_size=2, hidden_size=3)

# sequence of 5 time steps, batch of 1
x = torch.rand(5, 1, 2)          # (seq_len, batch, input_size)
h0 = torch.zeros(1, 1, 3)        # initial hidden state — all zeros

output, h_n = rnn(x, h0)

# output shape: (5, 1, 3) — h_t at every time step
# h_n shape:   (1, 1, 3) — final hidden state after step 5
```

PyTorch runs `h_t = tanh(W_hh · h_(t-1) + W_xh · x_t + b)` automatically at each step.

---

## Mathematical Example — 3 Time Steps

**Setup:**
```
input_size  = 1   (scalar input)
hidden_size = 2   (2 neurons)
sequence    = x = [1.0, 2.0, 3.0]
target      = y = [1.0, 2.0, 3.0]   (predict the input)
h_0         = [0.0, 0.0]             (start with zeros)

Parameters (fixed, not yet trained):
W_xh = [[0.5],        shape (2,1) — 2 neurons, 1 input weight each
         [0.3]]

W_hh = [[0.4, 0.1],   shape (2,2) — 2 neurons, 2 hidden weights each
         [0.2, 0.3]]

W_hy = [[0.6, 0.4]]   shape (1,2) — output layer, maps h → scalar

b    = [0.0, 0.0]     bias
```

### Forward Pass

**Step t=1, x=1.0**
```
z_1 = W_hh · h_0  +  W_xh · x_1  +  b

    = [[0.4, 0.1], · [0.0]  +  [[0.5], · [1.0]  +  [0.0]
       [0.2, 0.3]]   [0.0]]     [0.3]]              [0.0]

    = [0.0, 0.0]  +  [0.5, 0.3]  +  [0.0, 0.0]  =  [0.5, 0.3]

h_1 = tanh([0.5, 0.3]) = [0.462, 0.291]

ŷ_1 = W_hy · h_1 = 0.6×0.462 + 0.4×0.291 = 0.393

J_1 = (ŷ_1 - y_1)² = (0.393 - 1.0)² = 0.368
```

**Step t=2, x=2.0**
```
z_2 = W_hh · h_1  +  W_xh · x_2  +  b

    = [0.4×0.462 + 0.1×0.291,  0.2×0.462 + 0.3×0.291]  +  [1.0, 0.6]
    = [0.214, 0.179]  +  [1.0, 0.6]  =  [1.214, 0.779]

h_2 = tanh([1.214, 0.779]) = [0.837, 0.652]

ŷ_2 = 0.6×0.837 + 0.4×0.652 = 0.763

J_2 = (0.763 - 2.0)² = 1.530
```

**Step t=3, x=3.0**
```
z_3 = W_hh · h_2  +  W_xh · x_3

    = [0.4×0.837 + 0.1×0.652,  0.2×0.837 + 0.3×0.652]  +  [1.5, 0.9]
    = [0.400, 0.363]  +  [1.5, 0.9]  =  [1.900, 1.263]

h_3 = tanh([1.900, 1.263]) = [0.956, 0.852]

ŷ_3 = 0.6×0.956 + 0.4×0.852 = 0.915

J_3 = (0.915 - 3.0)² = 4.348
```

**Total Loss**
```
J(θ) = J_1 + J_2 + J_3 = 0.368 + 1.530 + 4.348 = 6.246
```

---

### BPTT — Gradient Flow Back Through W_hh

To update W_hh we need dJ/dW_hh. Each time step contributes:

```
dJ/dW_hh = dJ_1/dW_hh  +  dJ_2/dW_hh  +  dJ_3/dW_hh
```

The gradient at each step must travel back through W_hh to reach earlier steps:

```
dJ_3/dW_hh  →  flows through h_3 only          (1 hop)
dJ_2/dW_hh  →  flows through h_2 → h_1         (2 hops, × W_hh once)
dJ_1/dW_hh  →  flows through h_1 → h_0         (3 hops, × W_hh twice)
```

Concretely, gradient from J_1 back to W_hh at t=1:
```
dJ_1/dh_1  =  2(ŷ_1 - y_1) · W_hy · (1 - h_1²)      ← tanh derivative
           =  2(0.393 - 1.0) · [0.6, 0.4] · [1-0.462², 1-0.291²]
           =  -1.214          · [0.6, 0.4] · [0.787,    0.915]
           =  [-0.573, -0.444]
```

Gradient from J_3 flowing back to t=1 (multiplies W_hh twice):
```
[[0.4,0.1],[0.2,0.3]]² = [[0.18, 0.07],
                           [0.14, 0.11]]

Each multiplication shrinks the signal → early steps get near-zero gradient
→ W_hh learns nothing from x_1's influence on J_3
```

### What This Looks Like on the Diagram

```
      J(1,θ)=0.368    J(2,θ)=1.530    J(3,θ)=4.348
           ↑                ↑                ↑
  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
  │h(1)            │ │h(2)            │ │h(3)            │
  │  ● 0.462       │ │  ● 0.837       │ │  ● 0.956       │
  │  ● 0.291       │ │  ● 0.652       │ │  ● 0.852       │
  └────────────────┘ └────────────────┘ └────────────────┘
           ↑         Wh↑               Wh↑
          x=1.0          x=2.0               x=3.0

← gradient shrinks with each Wh hop going left (vanishing gradient)
```

---

## Symmetry in RNN

### 1. Weight Symmetry (Shared Weights Across Time)

The **same W_xh, W_hh, and b are used at every time step** — they never change across the sequence.

```
Step 1:   h1 = tanh(W_hh · h0  +  W_xh · x1  +  b)
Step 2:   h2 = tanh(W_hh · h1  +  W_xh · x2  +  b)
Step 3:   h3 = tanh(W_hh · h2  +  W_xh · x3  +  b)
                    ─────────────────────────────────
                    same W_hh, W_xh, b throughout
```

Compare to MLP:
```
MLP:   each layer has its OWN weights  W1, W2, W3 ...
RNN:   every time step reuses the SAME weights
```

Why this matters:
- Far fewer parameters — one set of weights handles sequences of any length
- Pattern learned at step 2 applies at step 50

### 2. Structural Symmetry (Unrolling)

An RNN unrolled over 3 steps looks like a 3-layer MLP — but with tied weights:

```
Unrolled RNN:                     MLP (for comparison):

x1 ──→ [W_xh, W_hh] ──→ h1       x ──→ [W1] ──→ h1
                  ↓                              ↓
x2 ──→ [W_xh, W_hh] ──→ h2            [W2] ──→ h2
                  ↓                              ↓
x3 ──→ [W_xh, W_hh] ──→ h3            [W3] ──→ h3
         ↑
    same weights at every level (weight tying = the symmetry)
```

### 3. The Problem This Symmetry Creates

Because the same weights are applied repeatedly, gradients during backprop get
**multiplied by W_hh at every step**:

```
Gradient at step 1 = gradient × W_hh × W_hh × W_hh × ... (t times)

If W_hh values < 1:   0.9^50 = 0.005   → vanishes  (nothing learned from early steps)
If W_hh values > 1:   1.1^50 = 117     → explodes  (training collapses)
```

### 4. How LSTM Breaks the Problem

LSTM introduces **gates** — learned values between 0 and 1 that control how much
of the hidden state flows through:

```
Vanilla RNN:   h_t = tanh(W_hh · h_(t-1) + W_xh · x_t + b)
                     always full overwrite — symmetric multiplication

LSTM:          c_t = f_t ⊙ c_(t-1)  +  i_t ⊙ candidate
                     ↑
                     forget gate (0=forget, 1=keep)
                     additive update — gradient flows without multiplying W repeatedly
```

The **additive** rather than **multiplicative** update breaks the vanishing gradient.

### Summary

| Symmetry | What it means | Consequence |
|---|---|---|
| Weight sharing across time | same W at every step | efficient, fewer params |
| Unrolled = deep network | gradients flow back through all steps | vanishing/exploding gradient |
| LSTM gates | break full symmetry with learned gating | stable long-range memory |

---

## Why Vanilla RNN Has a Problem

```
Step 1 → h1 → Step 2 → h2 → ... → Step 50 → h50

By step 50, h1's signal has been multiplied through 49 tanh operations.
tanh squishes to (-1,1) each time → early information nearly vanishes.
```

This is the **vanishing gradient problem** — why LSTM/GRU were invented.  
They add **gates** to control what to remember and what to forget,  
instead of always overwriting the hidden state.

---

## LSTM / GRU — The Fix

| Model | Gates | Memory |
|---|---|---|
| Vanilla RNN | none | overwrites hidden state every step |
| GRU | reset, update | lighter — good for shorter sequences |
| LSTM | forget, input, output | full control — best for long sequences |

```python
lstm = nn.LSTM(input_size=2, hidden_size=3)   # returns (output, (h_n, c_n))
gru  = nn.GRU(input_size=2, hidden_size=3)    # returns (output, h_n)
```

LSTM has two states: `h_n` (hidden) and `c_n` (cell — the long-term memory).
