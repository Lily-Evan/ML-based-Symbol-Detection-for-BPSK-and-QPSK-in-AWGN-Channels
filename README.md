# ML-based Symbol Detection for BPSK & QPSK in AWGN and Rayleigh Fading

This repository contains NumPy-based implementations of **neural symbol detectors** for BPSK and QPSK in:

- **AWGN channels** (benchmark case with known optimal detectors)
- **Rayleigh fading channels with imperfect CSI**  
  (a more realistic and challenging mismatched scenario)

Two main Python scripts are provided:

- `ψηφιακατηλεπ.py` — baseline MLP with one hidden layer and Gradient Descent  
- `adam.py` — deeper MLP with multiple hidden layers and Adam optimizer

Both models generate full BER curves, training logs, and diagnostic plots.

---

## 1. Theoretical Background

### 1.1 AWGN Channel

We consider the standard baseband AWGN model:

\[
y = s + n,\quad n \sim \mathcal{N}(0, \sigma^2)
\]

The bit-energy-to-noise ratio is:

\[
\gamma_b = \frac{E_b}{N_0},\quad 
\sigma^2 = \frac{1}{2\gamma_b}
\]

The theoretical BER for BPSK (and Gray-coded QPSK) is:

\[
P_b = Q\left(\sqrt{2\gamma_b}\right).
\]

---

### 1.2 Modulation Schemes

#### BPSK
- Bits: \(b \in \{0,1\}\)
- Symbol mapping:
  \[
  b=0 \rightarrow +1,\quad b=1 \rightarrow -1
  \]
- Optimal ML decision (AWGN):
  \[
  \hat{b} = \mathbb{1}\{y < 0\}
  \]

#### QPSK (2D real representation)
- Two bits per symbol \((b_0, b_1)\)
- Mapping:
  \[
  I = 1 - 2b_0, \quad Q = 1 - 2b_1
  \]
- AWGN ML decision:
  \[
  \hat{b}_0 = \mathbb{1}\{y_I < 0\}, \quad
  \hat{b}_1 = \mathbb{1}\{y_Q < 0\}
  \]

---

### 1.3 Rayleigh Fading with Imperfect CSI

Channel:

\[
y = h s + n,\quad h \sim \mathcal{CN}(0,1)
\]

Imperfect channel estimate:

\[
\hat{h} = h + e,\quad e \sim \mathcal{CN}(0,\sigma_e^2)
\]

**Classical mismatched detector:**
- assumes \(\hat{h}\) is correct
- equalizes: \(y_{\text{eq}} = y/\hat{h}\)  
- performs AWGN QPSK detection  

**Neural detector:**
- input vector:
  \[
  [\Re(y), \Im(y), \Re(\hat{h}), \Im(\hat{h})]
  \]
- learns robust decisions accounting for CSI uncertainty.

---

## 2. Repository Files

