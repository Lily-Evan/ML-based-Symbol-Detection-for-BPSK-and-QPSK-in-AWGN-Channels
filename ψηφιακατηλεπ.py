#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-based symbol decision for BPSK & QPSK over AWGN and Rayleigh fading (imperfect CSI)
-------------------------------------------------------------------------------

This script is a *self-contained mini-lab* for digital communications with ML:

1) Baseline AWGN channel:
   - BPSK and QPSK modulation
   - Labeled datasets (train / val / test) for a chosen training SNR
   - Tiny MLP (NumPy-only) trained to perform symbol/bit decisions
   - BER comparison:
        - MLP-based detector
        - Classical optimal detector (threshold-based)
        - Theoretical BER from Q-function (AWGN, bit-level)

2) Extended scenario: Rayleigh fading channel with imperfect CSI:
   - QPSK modulation over flat Rayleigh fading
   - Per-symbol random channel h ~ CN(0, 1)
   - Imperfect channel estimate h_hat = h + e (estimation error)
   - MLP input: [Re(y), Im(y), Re(h_hat), Im(h_hat)]
   - Classical "mismatched" detector: assumes h_hat is the true channel,
     performs one-tap equalization and standard QPSK detection.
   - BER comparison:
        - MLP-based detector (data-driven, robust to estimation error)
        - Classical mismatched detector

The code:
   - Is fully written in NumPy + Matplotlib
   - Trains 3 small MLPs:
        (a) BPSK over AWGN
        (b) QPSK over AWGN
        (c) QPSK over Rayleigh fading with imperfect CSI
   - Saves CSV files and plots for BER curves and training losses

This file can serve as:
   - The core experimental setup for a course project
   - The basis for a paper-style report: AWGN baseline + fading extension
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# =============================================================================
# 1. Global parameters / hyperparameters
# =============================================================================

# Training SNR for AWGN and fading scenarios (in dB)
TRAIN_SNR_DB_AWGN = 6.0          # SNR used to train AWGN detectors
TRAIN_SNR_DB_FADING = 6.0        # SNR used to train Rayleigh fading detector

# Range of SNRs (Eb/N0 in dB) for evaluation
EVAL_SNRS_DB = np.arange(0, 13, 1)   # 0, 1, 2, ..., 12 dB

# Dataset sizes for AWGN (you can increase for final experiments)
N_BPSK_TRAIN, N_BPSK_VAL = 100_000, 20_000
N_QPSK_TRAIN, N_QPSK_VAL = 100_000, 20_000
N_TEST_BPSK, N_TEST_QPSK = 100_000, 50_000

# Dataset sizes for fading scenario (QPSK over Rayleigh with imperfect CSI)
N_QPSK_FAD_TRAIN, N_QPSK_FAD_VAL = 120_000, 30_000
N_TEST_QPSK_FAD = 80_000

# Training hyperparameters for the three MLP models
EPOCHS_BPSK, EPOCHS_QPSK_AWGN = 10, 12
EPOCHS_QPSK_FADING = 15

BATCH_SIZE = 512

LR_BPSK = 5e-3
LR_QPSK_AWGN = 7e-3
LR_QPSK_FADING = 7e-3

# Hidden layer sizes
HID_BPSK = 16              # 1D input -> 1D output
HID_QPSK_AWGN = 24         # 2D input -> 2D output
HID_QPSK_FADING = 32       # 4D input (y, h_hat) -> 2D output

# Rayleigh fading: variance of channel estimation error
# h_hat = h + e,    E[|e|^2] = EST_VAR_H
EST_VAR_H = 0.1           # adjust to control quality of CSI

# Global random seed for reproducibility
RANDOM_SEED = 7


# =============================================================================
# 2. Utility functions (SNR, noise, theory)
# =============================================================================

def db2lin(db):
    """
    Convert dB to linear scale, i.e., 10^(db/10).
    """
    return 10.0 ** (db / 10.0)


def awgn_sigma(EbN0_dB):
    """
    Compute the standard deviation of AWGN for a given Eb/N0 (in dB).

    Assumptions:
      - Bit energy Eb = 1
      - N0 = 1 / (Eb/N0)
      - Real-valued noise with variance sigma^2 = N0 / 2 per real dimension
    """
    EbN0_lin = db2lin(EbN0_dB)
    N0 = 1.0 / EbN0_lin       # since Eb = 1
    return np.sqrt(N0 / 2.0)  # per real dimension


def Qfunc(x):
    """
    Q-function, implemented in a numerically stable manner using the error function:
       Q(x) = 0.5 * (1 - erf(x / sqrt(2))).
    """
    import numpy as _np
    import math as _math
    x = _np.asarray(x, dtype=float)
    vec = _np.vectorize(lambda t: 0.5 * (1.0 - _math.erf(t / _np.sqrt(2.0))))
    return vec(x)


def theory_ber_bit_awgn(EbN0_dB):
    """
    Theoretical bit error probability over AWGN for BPSK (and Gray-coded QPSK):
       P_b = Q( sqrt(2 * Eb/N0) ).
    """
    EbN0_lin = db2lin(EbN0_dB)
    return Qfunc(np.sqrt(2 * EbN0_lin))


# =============================================================================
# 3. Dataset generation: BPSK/QPSK in AWGN
# =============================================================================

def gen_bpsk_awgn(n_bits, EbN0_dB):
    """
    Generate BPSK samples over AWGN.

    Bits: b in {0,1}
    Mapping: 0 -> +1,  1 -> -1
    Channel: y = s + n,  n ~ N(0, sigma^2)
    """
    bits = np.random.randint(0, 2, size=n_bits, dtype=np.uint8)
    symbols = 1 - 2 * bits  # 0 -> +1, 1 -> -1
    sigma = awgn_sigma(EbN0_dB)
    noise = sigma * np.random.randn(n_bits)
    y = symbols + noise

    # Return inputs as column vector (n,1) for the MLP
    X = y.reshape(-1, 1)
    return X, bits


def gen_qpsk_awgn(n_symbols, EbN0_dB):
    """
    Generate QPSK samples over AWGN in a real 2D representation.

    Bits per symbol: [b0, b1], each in {0,1}
    Mapping: 0 -> +1, 1 -> -1 for each dimension (I and Q)
    Channel:
       yI = I + nI
       yQ = Q + nQ
    where nI, nQ ~ N(0, sigma^2).
    """
    # 2 bits per symbol
    bits = np.random.randint(0, 2, size=2 * n_symbols, dtype=np.uint8)
    b0 = bits[0::2]   # bits on I
    b1 = bits[1::2]   # bits on Q

    I = 1 - 2 * b0
    Q = 1 - 2 * b1

    sigma = awgn_sigma(EbN0_dB)
    nI = sigma * np.random.randn(n_symbols)
    nQ = sigma * np.random.randn(n_symbols)

    yI = I + nI
    yQ = Q + nQ

    # Input to MLP: 2D real vector [yI, yQ]
    X = np.stack([yI, yQ], axis=1)

    # Labels: two bits per symbol
    Y = np.stack([b0, b1], axis=1)
    return X, Y


# =============================================================================
# 4. Dataset generation: QPSK over Rayleigh fading with imperfect CSI
# =============================================================================

def gen_qpsk_rayleigh_imperfect(n_symbols, EbN0_dB, est_var_h=0.1):
    """
    Generate QPSK samples over a flat Rayleigh fading channel with imperfect CSI.

    Model:
       s = sI + j*sQ,  sI,sQ in {+1,-1}   (QPSK, unit amplitude per dimension)
       h ~ CN(0,1)                       (complex Rayleigh fading)
       n ~ CN(0, N0)                     (complex AWGN)
       y = h * s + n
       h_hat = h + e                     (imperfect estimate)
          with e ~ CN(0, est_var_h)

    MLP input:
       X = [ Re(y), Im(y), Re(h_hat), Im(h_hat) ]

    Labels:
       Y = [b0, b1], original bits.

    Notes:
       - We use awgn_sigma(EbN0_dB) for the real and imaginary parts of noise.
       - For the channel, we generate real and imaginary parts with variance 0.5,
         so that E[|h|^2] = 1.
    """
    # Generate random bits
    bits = np.random.randint(0, 2, size=2 * n_symbols, dtype=np.uint8)
    b0 = bits[0::2]
    b1 = bits[1::2]

    # QPSK mapping: 0 -> +1, 1 -> -1 (I and Q)
    I = 1 - 2 * b0
    Q = 1 - 2 * b1
    s = I + 1j * Q  # complex QPSK symbol

    # Rayleigh fading channel: h ~ CN(0,1)
    hr = np.random.randn(n_symbols) / np.sqrt(2.0)
    hi = np.random.randn(n_symbols) / np.sqrt(2.0)
    h = hr + 1j * hi

    # AWGN noise: n ~ CN(0, N0)
    sigma = awgn_sigma(EbN0_dB)
    nr = sigma * np.random.randn(n_symbols)
    ni = sigma * np.random.randn(n_symbols)
    n = nr + 1j * ni

    # Received signal
    y = h * s + n

    # Imperfect channel estimate: h_hat = h + e,  e ~ CN(0, est_var_h)
    er = np.sqrt(est_var_h / 2.0) * np.random.randn(n_symbols)
    ei = np.sqrt(est_var_h / 2.0) * np.random.randn(n_symbols)
    e = er + 1j * ei
    h_hat = h + e

    # Inputs to MLP: concatenate real/imag parts of y and h_hat
    X = np.stack([y.real, y.imag, h_hat.real, h_hat.imag], axis=1)

    # Labels: bits b0,b1
    Y = np.stack([b0, b1], axis=1)

    return X, Y, y, h_hat


# =============================================================================
# 5. Simple fully-connected MLP in NumPy
# =============================================================================

class MLP:
    """
    Minimalistic 2-layer MLP:
        Input -> ReLU(hidden) -> Sigmoid(output)

    - Supports multi-label outputs (e.g., 2 bits for QPSK).
    - Trains with Binary Cross Entropy (BCE) via gradient descent.
    """

    def __init__(self, input_dim, output_dim, hidden=16, lr=1e-2, seed=1):
        rng = np.random.default_rng(seed)

        # Weight matrices and biases
        self.W1 = rng.normal(0, 0.1, size=(input_dim, hidden))
        self.b1 = np.zeros((1, hidden))

        self.W2 = rng.normal(0, 0.1, size=(hidden, output_dim))
        self.b2 = np.zeros((1, output_dim))

        self.lr = lr

    def forward(self, X):
        """
        Forward pass:
           Z1 = X W1 + b1
           A1 = ReLU(Z1)
           Z2 = A1 W2 + b2
           Yhat = sigmoid(Z2)
        """
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)  # ReLU
        Z2 = A1 @ self.W2 + self.b2
        Yhat = 1.0 / (1.0 + np.exp(-Z2))  # Sigmoid
        cache = (X, Z1, A1, Z2, Yhat)
        return Yhat, cache

    @staticmethod
    def bce_loss(Yhat, Y):
        """
        Binary Cross Entropy loss over all samples and output units:
           L = - mean( Y*log(Yhat) + (1-Y)*log(1-Yhat) ).
        """
        eps = 1e-12  # for numerical stability
        Yhat_clipped = np.clip(Yhat, eps, 1 - eps)
        return -np.mean(Y * np.log(Yhat_clipped) +
                        (1 - Y) * np.log(1 - Yhat_clipped))

    def backward(self, cache, Y):
        """
        Backward pass: compute gradients of BCE loss w.r.t. parameters
        and perform one gradient descent step.
        """
        X, Z1, A1, Z2, Yhat = cache
        m = X.shape[0]

        # Derivative of BCE wrt Z2 (using sigmoid derivative implicitly)
        dZ2 = (Yhat - Y) / m
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)  # ReLU derivative

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, Y, Xval=None, Yval=None,
            epochs=10, batch_size=256, verbose=True):
        """
        Mini-batch gradient descent training loop.

        Returns:
            losses     : list of training losses per epoch
            val_losses : list of validation losses per epoch (if val data given)
        """
        n = X.shape[0]
        losses, val_losses = [], []

        for ep in range(epochs):
            # Shuffle the training data at each epoch
            idx = np.random.permutation(n)
            Xs, Ys = X[idx], Y[idx]

            # Process mini-batches
            for i in range(0, n, batch_size):
                xb = Xs[i:i + batch_size]
                yb = Ys[i:i + batch_size]
                Yhat, cache = self.forward(xb)
                self.backward(cache, yb)

            # Compute full training loss at end of epoch
            Yhat_all, _ = self.forward(X)
            loss = self.bce_loss(Yhat_all, Y)
            losses.append(float(loss))

            # Compute validation loss (if provided)
            if Xval is not None and Yval is not None:
                Yhat_val, _ = self.forward(Xval)
                vloss = self.bce_loss(Yhat_val, Yval)
                val_losses.append(float(vloss))
                if verbose:
                    print(f"Epoch {ep+1}/{epochs}: "
                          f"loss={loss:.4f} val_loss={vloss:.4f}")
            else:
                if verbose:
                    print(f"Epoch {ep+1}/{epochs}: loss={loss:.4f}")

        return losses, val_losses

    def predict_bits(self, X):
        """
        Predict bits (0/1) given input X, by thresholding the sigmoid outputs at 0.5.
        """
        Yhat, _ = self.forward(X)
        return (Yhat >= 0.5).astype(np.uint8)


# =============================================================================
# 6. Evaluation utilities (BER computation)
# =============================================================================

def evaluate_ber_bpsk_awgn(model, EbN0_dB_list, n_test=100_000):
    """
    Evaluate BER for BPSK over AWGN, for:
       - MLP-based detector
       - Optimal classical detector (threshold at 0)
    """
    ber_mlp, ber_opt = [], []

    for eb in EbN0_dB_list:
        X, bits = gen_bpsk_awgn(n_test, eb)

        # MLP detector
        bhat_mlp = model.predict_bits(X).reshape(-1)
        ber_mlp.append(np.mean(bhat_mlp != bits))

        # Classical optimal detector: sign threshold at 0
        bhat_opt = (X.reshape(-1) < 0).astype(np.uint8)
        ber_opt.append(np.mean(bhat_opt != bits))

    return np.array(ber_mlp), np.array(ber_opt)


def evaluate_ber_qpsk_awgn(model, EbN0_dB_list, n_test=50_000):
    """
    Evaluate BER for QPSK over AWGN, for:
       - MLP-based detector
       - Optimal classical detector (sign per axis)
    """
    ber_mlp, ber_opt = [], []

    for eb in EbN0_dB_list:
        X, Y = gen_qpsk_awgn(n_test, eb)

        # MLP detector
        Yhat_mlp = model.predict_bits(X)
        ber_mlp.append(np.mean(Yhat_mlp != Y))

        # Classical optimal detector: sign threshold on I and Q
        b0_opt = (X[:, 0] < 0).astype(np.uint8)
        b1_opt = (X[:, 1] < 0).astype(np.uint8)
        Yopt = np.stack([b0_opt, b1_opt], axis=1)
        ber_opt.append(np.mean(Yopt != Y))

    return np.array(ber_mlp), np.array(ber_opt)


def evaluate_ber_qpsk_rayleigh(model, EbN0_dB_list,
                               n_test=50_000, est_var_h=0.1):
    """
    Evaluate BER for QPSK over Rayleigh fading with imperfect CSI, for:
       - MLP-based detector (input: [Re(y), Im(y), Re(h_hat), Im(h_hat)])
       - Classical "mismatched" detector:
           - assumes h_hat is the true channel,
           - computes y_eq = y / h_hat,
           - applies sign-based QPSK detection on y_eq.

    Returns:
        ber_mlp   : BER of the MLP-based detector
        ber_mis   : BER of the mismatched classical detector
    """
    ber_mlp, ber_mis = [], []

    for eb in EbN0_dB_list:
        # Generate test samples for this SNR
        X, Y_true, y, h_hat = gen_qpsk_rayleigh_imperfect(
            n_test, eb, est_var_h=est_var_h
        )

        # MLP-based detector
        Yhat_mlp = model.predict_bits(X)
        ber_mlp.append(np.mean(Yhat_mlp != Y_true))

        # Classical mismatched detector:
        #   equalize using h_hat and then detect as in AWGN
        y_eq = y / h_hat
        b0_mis = (y_eq.real < 0).astype(np.uint8)
        b1_mis = (y_eq.imag < 0).astype(np.uint8)
        Y_mis = np.stack([b0_mis, b1_mis], axis=1)
        ber_mis.append(np.mean(Y_mis != Y_true))

    return np.array(ber_mlp), np.array(ber_mis)


# =============================================================================
# 7. Main experiment pipeline
# =============================================================================

def main():
    # Fix global seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # 7.1 Train MLP for BPSK over AWGN
    # -------------------------------------------------------------------------
    print("\n=== Training BPSK MLP (AWGN) ===")
    Xtr_b, Ytr_b = gen_bpsk_awgn(N_BPSK_TRAIN, TRAIN_SNR_DB_AWGN)
    Xval_b, Yval_b = gen_bpsk_awgn(N_BPSK_VAL, TRAIN_SNR_DB_AWGN)

    # Convert labels to float column vectors for BCE
    Ytr_b = Ytr_b.reshape(-1, 1).astype(np.float64)
    Yval_b = Yval_b.reshape(-1, 1).astype(np.float64)

    mlp_bpsk = MLP(
        input_dim=1,
        output_dim=1,
        hidden=HID_BPSK,
        lr=LR_BPSK,
        seed=1
    )

    losses_b, vlosses_b = mlp_bpsk.fit(
        Xtr_b, Ytr_b,
        Xval_b, Yval_b,
        epochs=EPOCHS_BPSK,
        batch_size=BATCH_SIZE,
        verbose=True
    )

    # -------------------------------------------------------------------------
    # 7.2 Train MLP for QPSK over AWGN
    # -------------------------------------------------------------------------
    print("\n=== Training QPSK MLP (AWGN) ===")
    Xtr_q, Ytr_q = gen_qpsk_awgn(N_QPSK_TRAIN, TRAIN_SNR_DB_AWGN)
    Xval_q, Yval_q = gen_qpsk_awgn(N_QPSK_VAL, TRAIN_SNR_DB_AWGN)

    Ytr_q = Ytr_q.astype(np.float64)
    Yval_q = Yval_q.astype(np.float64)

    mlp_qpsk_awgn = MLP(
        input_dim=2,
        output_dim=2,
        hidden=HID_QPSK_AWGN,
        lr=LR_QPSK_AWGN,
        seed=2
    )

    losses_q_awgn, vlosses_q_awgn = mlp_qpsk_awgn.fit(
        Xtr_q, Ytr_q,
        Xval_q, Yval_q,
        epochs=EPOCHS_QPSK_AWGN,
        batch_size=BATCH_SIZE,
        verbose=True
    )

    # -------------------------------------------------------------------------
    # 7.3 Train MLP for QPSK over Rayleigh fading with imperfect CSI
    # -------------------------------------------------------------------------
    print("\n=== Training QPSK MLP (Rayleigh fading, imperfect CSI) ===")
    Xtr_f, Ytr_f, _, _ = gen_qpsk_rayleigh_imperfect(
        N_QPSK_FAD_TRAIN,
        TRAIN_SNR_DB_FADING,
        est_var_h=EST_VAR_H
    )
    Xval_f, Yval_f, _, _ = gen_qpsk_rayleigh_imperfect(
        N_QPSK_FAD_VAL,
        TRAIN_SNR_DB_FADING,
        est_var_h=EST_VAR_H
    )

    Ytr_f = Ytr_f.astype(np.float64)
    Yval_f = Yval_f.astype(np.float64)

    mlp_qpsk_fading = MLP(
        input_dim=4,      # [Re(y), Im(y), Re(h_hat), Im(h_hat)]
        output_dim=2,     # two bits per symbol
        hidden=HID_QPSK_FADING,
        lr=LR_QPSK_FADING,
        seed=3
    )

    losses_q_fad, vlosses_q_fad = mlp_qpsk_fading.fit(
        Xtr_f, Ytr_f,
        Xval_f, Yval_f,
        epochs=EPOCHS_QPSK_FADING,
        batch_size=BATCH_SIZE,
        verbose=True
    )

    # -------------------------------------------------------------------------
    # 7.4 Evaluate AWGN BER (BPSK & QPSK) vs theory & optimal
    # -------------------------------------------------------------------------
    print("\n=== Evaluating BER over AWGN ===")
    ber_mlp_b, ber_opt_b = evaluate_ber_bpsk_awgn(
        mlp_bpsk, EVAL_SNRS_DB, n_test=N_TEST_BPSK
    )
    ber_mlp_q, ber_opt_q = evaluate_ber_qpsk_awgn(
        mlp_qpsk_awgn, EVAL_SNRS_DB, n_test=N_TEST_QPSK
    )
    ber_theory = theory_ber_bit_awgn(EVAL_SNRS_DB)

    # -------------------------------------------------------------------------
    # 7.5 Evaluate Rayleigh fading BER (QPSK) vs mismatched detector
    # -------------------------------------------------------------------------
    print("\n=== Evaluating BER over Rayleigh fading (imperfect CSI) ===")
    ber_mlp_fad, ber_mis_fad = evaluate_ber_qpsk_rayleigh(
        mlp_qpsk_fading, EVAL_SNRS_DB,
        n_test=N_TEST_QPSK_FAD,
        est_var_h=EST_VAR_H
    )

    # -------------------------------------------------------------------------
    # 7.6 Save CSV results
    # -------------------------------------------------------------------------
    # AWGN results
    with open("ber_awgn_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "EbN0_dB",
            "BER_BPSK_MLP",
            "BER_BPSK_Optimal",
            "BER_QPSK_MLP",
            "BER_QPSK_Optimal",
            "BER_Theory_Bit"
        ])
        for i, eb in enumerate(EVAL_SNRS_DB):
            w.writerow([
                float(eb),
                float(ber_mlp_b[i]),
                float(ber_opt_b[i]),
                float(ber_mlp_q[i]),
                float(ber_opt_q[i]),
                float(ber_theory[i])
            ])

    # Fading results
    with open("ber_fading_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "EbN0_dB",
            "BER_QPSK_MLP_Fading",
            "BER_QPSK_Mismatched_Fading",
            "Est_Var_h"
        ])
        for i, eb in enumerate(EVAL_SNRS_DB):
            w.writerow([
                float(eb),
                float(ber_mlp_fad[i]),
                float(ber_mis_fad[i]),
                float(EST_VAR_H)
            ])

    # -------------------------------------------------------------------------
    # 7.7 Plots: AWGN BER curves
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.semilogy(EVAL_SNRS_DB, ber_theory, "-.", linewidth=1.8,
                 label="Theory (AWGN, bit)")
    plt.semilogy(EVAL_SNRS_DB, ber_opt_b, "-", linewidth=1.8,
                 label="BPSK Optimal (AWGN)")
    plt.semilogy(EVAL_SNRS_DB, ber_mlp_b, "o", markersize=4,
                 label="BPSK MLP (AWGN)")
    plt.semilogy(EVAL_SNRS_DB, ber_opt_q, "--", linewidth=1.8,
                 label="QPSK Optimal (AWGN)")
    plt.semilogy(EVAL_SNRS_DB, ber_mlp_q, "s", markersize=4,
                 label="QPSK MLP (AWGN)")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("ML-based Detector vs Optimal & Theory (BPSK & QPSK, AWGN)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(loc="upper right")
    plt.ylim(1e-5, 1)
    plt.tight_layout()
    plt.savefig("ber_awgn_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -------------------------------------------------------------------------
    # 7.8 Plots: Rayleigh fading BER curves
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.semilogy(EVAL_SNRS_DB, ber_mis_fad, "--", linewidth=1.8,
                 label="QPSK mismatched detector (fading)")
    plt.semilogy(EVAL_SNRS_DB, ber_mlp_fad, "o-", markersize=4,
                 label="QPSK MLP (fading, imperfect CSI)")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("QPSK over Rayleigh Fading (Imperfect CSI): "
              "MLP vs Mismatched Detector")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(loc="upper right")
    plt.ylim(1e-4, 1)
    plt.tight_layout()
    plt.savefig("ber_fading_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -------------------------------------------------------------------------
    # 7.9 Plots: Training / validation losses
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    if losses_b:
        plt.plot(losses_b, label="BPSK train loss (AWGN)")
    if vlosses_b:
        plt.plot(vlosses_b, label="BPSK val loss (AWGN)")
    if losses_q_awgn:
        plt.plot(losses_q_awgn, label="QPSK train loss (AWGN)")
    if vlosses_q_awgn:
        plt.plot(vlosses_q_awgn, label="QPSK val loss (AWGN)")
    if losses_q_fad:
        plt.plot(losses_q_fad, label="QPSK train loss (fading)")
    if vlosses_q_fad:
        plt.plot(vlosses_q_fad, label="QPSK val loss (fading)")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training / Validation Losses (All MLP Models)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_losses_all.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -------------------------------------------------------------------------
    # 7.10 Diagnostic scatter for QPSK over AWGN @ 6 dB
    # -------------------------------------------------------------------------
    Xdiag, Ydiag = gen_qpsk_awgn(6000, TRAIN_SNR_DB_AWGN)
    Yhat_diag = mlp_qpsk_awgn.predict_bits(Xdiag)
    correct = np.all(Yhat_diag == Ydiag, axis=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(Xdiag[correct, 0], Xdiag[correct, 1],
                s=8, alpha=0.5, label="Correct")
    plt.scatter(Xdiag[~correct, 0], Xdiag[~correct, 1],
                s=8, alpha=0.5, label="Wrong")
    plt.title("QPSK MLP Decisions @ Eb/N0 = 6 dB (AWGN)")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qpsk_mlp_decisions_awgn_6dB.png",
                dpi=250, bbox_inches="tight")
    plt.show()

    # -------------------------------------------------------------------------
    # 7.11 Diagnostic scatter for QPSK over Rayleigh fading @ 6 dB
    # -------------------------------------------------------------------------
    Xdiag_f, Ydiag_f, ydiag_f, hhat_diag_f = gen_qpsk_rayleigh_imperfect(
        6000, TRAIN_SNR_DB_FADING, est_var_h=EST_VAR_H
    )
    Yhat_diag_f = mlp_qpsk_fading.predict_bits(Xdiag_f)
    correct_f = np.all(Yhat_diag_f == Ydiag_f, axis=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(ydiag_f.real[correct_f], ydiag_f.imag[correct_f],
                s=8, alpha=0.5, label="Correct")
    plt.scatter(ydiag_f.real[~correct_f], ydiag_f.imag[~correct_f],
                s=8, alpha=0.5, label="Wrong")
    plt.title("QPSK MLP Decisions @ Eb/N0 = 6 dB "
              "(Rayleigh fading, imperfect CSI)")
    plt.xlabel("Re{y}")
    plt.ylabel("Im{y}")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qpsk_mlp_decisions_fading_6dB.png",
                dpi=250, bbox_inches="tight")
    plt.show()

    print("\nSaved files:")
    print("  - ber_awgn_results.csv")
    print("  - ber_fading_results.csv")
    print("  - ber_awgn_curves.png")
    print("  - ber_fading_curves.png")
    print("  - training_losses_all.png")
    print("  - qpsk_mlp_decisions_awgn_6dB.png")
    print("  - qpsk_mlp_decisions_fading_6dB.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
