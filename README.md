# ML-Based Symbol Detection for BPSK & QPSK  
### Neural Receivers under AWGN & Rayleigh Fading Channels

This repository provides two NumPy-only implementations of neural symbol detectors for BPSK and QPSK modulation. The aim is to study whether small feedforward neural networks can learn (and sometimes surpass) classical communication-theory detectors under different channel conditions.

---

## 🎯 Project Goals

- Learn the **optimal detector** in an AWGN channel using a compact MLP.  
- Investigate whether a neural detector can outperform the **mismatched classical detector** under Rayleigh fading with **imperfect CSI**.  
- Compare empirical BER curves against **theoretical baselines**.

---

## 📂 Repository Structure

| File              | Description                                                       |
|-------------------|-------------------------------------------------------------------|
| `ψηφιακατηλεπ.py` | Simple MLP (1 hidden layer) trained with vanilla Gradient Descent |
| `adam.py`         | Deeper MLP trained with the Adam optimizer (more realistic setup) |

Each script runs independently and automatically generates plots, CSV files, and training logs.

---

## 📡 Main Functionality

### 1. Dataset Generation

The scripts generate synthetic datasets consisting of:

- **Modulations:** BPSK & QPSK  
- **Channels:**
  - AWGN
  - Rayleigh fading (optional)  
- **Channel State Information (CSI):**
  - Perfect CSI (AWGN)
  - **Imperfect CSI** for Rayleigh fading scenarios

---

### 2. Neural Detector Architecture

Both scripts implement MLP-based receivers with:

- **Input:** Received complex samples (mapped to real-valued features)  
- **Hidden layers:** ReLU activations  
- **Output:** Bit-wise predictions via sigmoid activations  
- **Loss function:** Binary Cross-Entropy (BCE)  

**Optimization:**

- `ψηφιακατηλεπ.py`: Gradient Descent  
- `adam.py`: Adam optimizer  

---

### 3. Performance Evaluation

The scripts compute and compare **Bit Error Rate (BER)** as a function of **Eb/N₀**.

They include comparisons against:

- **Optimal detector** in AWGN  
- **Mismatched classical detector** in Rayleigh fading with imperfect CSI  
- **Theoretical BER** curves based on the Q-function (for AWGN)

---

### 4. Visualizations & Outputs

Each script automatically saves:

#### AWGN Results

- `ber_awgn_results*.csv`  
- `ber_awgn_curves*.png`  
- `qpsk_mlp_decisions_awgn_6dB*.png`  

#### Rayleigh Fading Results

- `ber_fading_results*.csv`  
- `ber_fading_curves*.png`  
- `qpsk_mlp_decisions_fading_6dB*.png`  

#### Training Diagnostics

- `training_losses_all*.png` (training & validation loss curves)

Example visualizations include:

- BER vs. Eb/N₀ curves (AWGN & Rayleigh fading)  
- Training/validation loss vs. epoch curves  
- QPSK decision scatter plots at **6 dB** (AWGN and fading)

---

## 🚀 How to Run

Make sure you have Python and NumPy installed.

From the terminal:

```bash
# Version 1: Simple MLP + Gradient Descent
python ψηφιακατηλεπ.py

# Version 2: Deeper MLP + Adam optimizer
python adam.py
```

Each script will automatically:

- Generate datasets  
- Train the neural detectors  
- Evaluate BER over a predefined Eb/N₀ range  
- Save all results (plots, CSVs, training logs) in the current directory  

---

## 📊 Example Training Logs

**BPSK (AWGN)**

```
=== Training BPSK MLP (AWGN) ===
Epoch 1/10: loss=0.2976 val_loss=0.2989
...
Epoch 10/10: loss=0.1022 val_loss=0.1026
```

**QPSK (AWGN)**

```
=== Training QPSK MLP (AWGN) ===
Epoch 1/12: loss=0.1556 val_loss=0.1538
...
Epoch 12/12: loss=0.0549 val_loss=0.0543
```

**QPSK (Rayleigh Fading, Imperfect CSI)**

```
=== Training QPSK MLP (Rayleigh fading, imperfect CSI) ===
Epoch 1/15: loss=0.6250 val_loss=0.6283
...
Epoch 15/15: loss=0.5763 val_loss=0.5814
```

**Adam Version (Faster Convergence)**

```
=== Training BPSK MLP (AWGN) with Adam ===
loss ≈ 0 from epoch 2 onward
```

---

## 🧠 Key Insights

- **AWGN channel**  
  The MLP successfully learns the **optimal maximum-likelihood detector**.  
  → The simulated BER curves closely match the theoretical ones.

- **Rayleigh fading with imperfect CSI**  
  The neural detector learns a **robust nonlinear decision rule**.  
  → It often **outperforms the mismatched classical detector**.

- **Adam vs. Gradient Descent**  
  - Adam converges significantly faster.  
  - It provides more stable training, especially in fading scenarios.

---

## 🔮 Possible Extensions

This project can serve as a starting point for more advanced research directions, such as:

- CNN- or RNN-based **sequence detectors**  
- Joint **equalization + detection** using deep learning  
- Training over **OFDM** channel models  
- **Adversarial robustness** experiments against structured noise  
- Exporting models for **on-device / TinyML inference**  

---

## 📜 License

This project is released under the **MIT License**.
