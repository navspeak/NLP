# M4 — PyTorch & CNN Notes

## ANN Hierarchy

**ANN = Artificial Neural Network** — the umbrella term for all neural networks.

```
ANN  (Artificial Neural Network)
│
│── any network of neurons with weights, trained by backprop
│
├── FNN / MLP  (Feedforward / Multi-Layer Perceptron)
│     └── nn.Linear → ReLU → nn.Linear
│         input flows one direction, no loops
│
├── CNN  (Convolutional Neural Network)
│     └── FNN but with Conv2d layers for spatial/image data
│
├── RNN  (Recurrent Neural Network)
│     └── has loops — output feeds back in, used for sequences/text
│
└── ... (Transformers, GANs, etc.)
```

### One-line Distinctions

| Name | Input | Key layer |
|---|---|---|
| ANN | anything | generic term |
| MLP/FNN | flat vector | `nn.Linear` |
| CNN | image / spatial | `nn.Conv2d` |
| RNN | sequence | `nn.LSTM` / `nn.GRU` |

### Key Insight
The **training loop never changes** regardless of ANN type:
```python
output = model(X)
loss   = criterion(output, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```
Only the model definition (what's inside `nn.Sequential`) changes.

---

## RNN (Recurrent Neural Network)

**RNN = neural network with a loop** — output from the previous step feeds back as input to the next.

### Core Difference from CNN/MLP

```
MLP/CNN — no memory:

x ──→ [neurons] ──→ output
       each input processed independently

RNN — has memory:

x₁ ──→ [neurons] ──→ output₁
            ↓  (hidden state passed forward)
x₂ ──→ [neurons] ──→ output₂
            ↓
x₃ ──→ [neurons] ──→ output₃
```

The **hidden state** is the memory — it carries information from previous steps.

### What It's Used For

Data where **order matters** — sequences:

| Task | Input sequence | Output |
|---|---|---|
| Text generation | words so far | next word |
| Sentiment analysis | words in review | positive/negative |
| Speech recognition | audio frames | text |
| Stock prediction | price history | next price |

### In PyTorch

```python
rnn  = nn.RNN(input_size=4, hidden_size=8)
lstm = nn.LSTM(input_size=4, hidden_size=8)   # handles long sequences better
gru  = nn.GRU(input_size=4, hidden_size=8)    # lighter than LSTM
```

### Which Architecture for Which Data

```
Data type          Best choice
──────────────────────────────
flat numbers   →   MLP  (nn.Linear)
image/spatial  →   CNN  (nn.Conv2d)
sequence/text  →   RNN  (nn.LSTM / nn.GRU)
```

### Relation to RL
RNNs appear in RL when the agent needs **memory** — e.g. a game where one frame isn't enough
to know what's happening (ball direction, enemy intent). The hidden state acts as the agent's
working memory across time steps.

---

## Dense Layer (Fully Connected Layer)

**Dense layer = `nn.Linear` in PyTorch** — same thing, different frameworks use different names.

| Framework | Name |
|---|---|
| PyTorch | `nn.Linear` |
| Keras / TensorFlow | `Dense` |
| Textbooks / papers | "Fully Connected (FC) layer" |

All three mean: every input neuron connects to every output neuron — `x @ W.T + b`.

### Where it sits in a CNN

```
Conv2d      ← learns spatial features (edges, shapes)
ReLU
MaxPool
     ↓
Flatten     ← image → flat vector
     ↓
nn.Linear   ← Dense: combines all features
ReLU
     ↓
nn.Linear   ← Dense: final classification / action scores
```

Conv layers **extract features**. Dense/Linear layers **make the decision**.

> Keras `Dense(128, activation='relu')` = PyTorch `nn.Linear(in, 128)` + `nn.ReLU()`

---

## Files

| File | Contents |
|---|---|
| `PyTorch_Tutorial.ipynb` | Step-by-step PyTorch — tensors → autograd → training loop → CNN |
| `CNN.md` | CNN vs Linear — relationship, shape flow, parameter comparison |
| `RNN.md` | Hidden state computation, vanishing gradient, LSTM/GRU |
| `CNN_Pytorch.ipynb` | Full CNN on CIFAR-10 |
