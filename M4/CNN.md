# CNN — Relationship to the Basic Training Loop

## The Core Insight

The training loop never changes. Only the **model** changes.

```python
output = model(X)            # X is a number vector today, an image tomorrow
loss   = criterion(output, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Side-by-Side: Linear vs CNN

```
Linear (what you built):        CNN (Atari DQN / image input):

X = [pos, vel, angle, score]    X = pixels of game screen
     4 numbers                       (1, 84, 84) tensor

     ↓ nn.Linear(4, 8)               ↓ nn.Conv2d layers
     x @ W + b                        kernel slides over image
                                       detects edges, shapes, patterns

     ↓ nn.Linear(8, 2)               ↓ nn.Flatten()
                                       collapses to 1D vector
                                      ↓ nn.Linear(..., 2)

     [score_action0, score_action1]  [score_action0, score_action1]
```

**Output is identical** — a vector of action scores. The agent picks `argmax`.

---

## Why Not Just Use Linear for Images?

```
Image = 84×84 pixels = 7,056 numbers

nn.Linear(7056, 8):
  - each neuron connects to ALL 7056 pixels
  - ignores spatial structure (nearby pixels are related!)
  - massive parameter count → slow, overfits

nn.Conv2d:
  - small 3×3 filter slides across image
  - each neuron looks at ONE local patch at a time
  - shared weights across positions → efficient
  - naturally detects edges → shapes → objects
```

---

## What Conv2d Actually Does

```
Input image (5×5):          3×3 filter (kernel):

1 2 3 4 5                   w1 w2 w3
6 7 8 9 0                   w4 w5 w6
1 2 3 4 5                   w7 w8 w9
6 7 8 9 0
1 2 3 4 5

Filter slides across every 3×3 patch and computes a dot product.
Output = one number per position = a "feature map".
Multiple filters → multiple feature maps → detect different patterns.
```

---

## Shape Flow Through a CNN

```
Input:          (batch, 1,  8,  8)   ← grayscale 8×8 image
After Conv2d:   (batch, 4,  6,  6)   ← 4 filters, spatial shrinks by 2 (no padding)
After ReLU:     (batch, 4,  6,  6)   ← shape unchanged
After Flatten:  (batch, 144)         ← 4 × 6 × 6 = 144
After Linear:   (batch, 2)           ← 2 action scores
```

---

## The Same 5 Loop Steps, Different Model

```python
# Linear model (game state)
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

# CNN model (image state) — loop is identical
model = nn.Sequential(
    nn.Conv2d(1, 4, kernel_size=3),   # 1 channel in, 4 filters
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(4 * 6 * 6, 2)          # flattened → 2 action scores
)
```

---

## Key Terms

| Term | Meaning |
|---|---|
| `in_channels` | colour channels of input (1=grayscale, 3=RGB) |
| `out_channels` | number of filters to learn |
| `kernel_size` | size of the sliding filter (3 = 3×3) |
| `padding=1` | pad border so output H,W stays same as input |
| `nn.Flatten()` | collapses (batch, C, H, W) → (batch, C×H×W) |
| feature map | output of one filter sliding over the image |
