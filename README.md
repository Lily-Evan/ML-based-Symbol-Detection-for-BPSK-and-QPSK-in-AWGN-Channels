
# 🧠 Machine Learning for Symbol Decision in AWGN Channel

**Course:** Digital Communication Systems  
**Student:** Panagiota Grosdouli  
**Language:** Python 3.x (NumPy, Matplotlib)

---

## 🎯 Objective
This project investigates the use of a **neural network (MLP)** for **symbol decision** in **BPSK** and **QPSK** modulation schemes under **Additive White Gaussian Noise (AWGN)**.  
The goal is to train a small MLP to learn the optimal decision rule and compare its Bit Error Rate (BER) to that of the classical maximum-likelihood detector.

---

## 📘 Theoretical Background
For an AWGN channel:

```text
y = s + n
n ~ N(0, σ²)
```

Relation between signal-to-noise ratio and noise variance:

```text
Eb/N0 = γb
σ² = 1 / (2 * γb)
```

Theoretical BER for BPSK:

```text
Pb = Q(sqrt(2 * Eb/N0))
```

---

## ⚙️ Implementation Steps

### 1️⃣ Data Generation
- Generate random bits.  
- Map them to BPSK or QPSK symbols.  
- Add AWGN noise for selected `Eb/N0` values.

### 2️⃣ MLP Training
- Architecture: `[Input] → ReLU(16–24 neurons) → Sigmoid(Output)`  
- Loss function: Binary Cross Entropy  
- Training SNR = 6 dB  
- Evaluate generalization for 0–12 dB.

### 3️⃣ Evaluation
- Compute **BER** for both modulations.  
- Compare **MLP vs Optimal Detector**.

---

## 🧪 Execution
To run the project:

```bash
python ml_symbol_decision.py
```

Automatically generates:
- `ml_vs_opt_ber.csv`
- `ml_vs_opt_ber.png`
- `qpsk_mlp_decisions_6dB.png`

---

## 📈 Results

### BER Curves (MLP vs Optimal)
![BER Curves](ml_vs_opt_ber.png)

### QPSK MLP Decisions @ 6 dB
![QPSK Decisions](qpsk_mlp_decisions_6dB.png)

---

## 🧠 Discussion
- The MLP effectively learns the **optimal decision boundary**.  
- In AWGN, its performance closely matches the **maximum-likelihood detector**.  
- At low SNR, small deviations occur due to limited training data.

---

## 🔮 Extensions
- Introduce **Rayleigh fading**:

```text
y = h * s + n
h ~ CN(0,1)
```

- Train with pilot estimates (`ĥ`) as extra input.  
- Explore CNN/LSTM architectures for sequence detection.

---

## 📚 Conclusion
Neural networks offer a **modern, data-driven approach** to symbol detection,  
capable of adapting to complex, non-ideal channel conditions.
