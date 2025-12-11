#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-based symbol decision for BPSK & QPSK over AWGN and Rayleigh fading (imperfect CSI)
with deeper MLPs and Adam optimizer
-------------------------------------------------------------------------------

This script implements a more "realistic" neural detector:

- Multiple hidden layers (configurable per model)
- Adam optimizer (instead of plain SGD) for stable training
- Many epochs & large datasets (configurable) for research-level simulations

Scenarios:
1) AWGN channel
   - BPSK and QPSK modulation
   - MLP learns the optimal (threshold-based) detector
   - BER vs:
        * MLP-based detector
        * Classical optimal detector
        * Theoretical BER (Q-function)

2) Rayleigh fading with imperfect CSI
   - QPSK modulation over flat Rayleigh fading
   - Imperfect channel estimation: h_hat = h + e
   - MLP input: [Re(y), Im(y), Re(h_hat), Im(h_hat)]
   - Classical mismatched detector uses h_hat as if it were true
   - BER vs:
        * MLP-based detector (data-driven, robust)
        * Classical mismatched detector

The code is fully NumPy-based and self-contained.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# =============================================================================
# 1. Global parameters / hyperparameters
# =============================================================================

# Quick demo vs heavy run flag (for fast tests vs long training)
HEAVY_RUN = False  # set to True for more data & epochs

# Training SNRs (in dB)
TRAIN_SNR_DB_AWGN = 6.0          # SNR used to train AWGN detectors
TRAIN_SNR_DB_FADING = 6.0        # SNR used to train Rayleigh fading detector

# Range of SNRs (Eb/N0 in dB) for evaluation
EVAL_SNRS_DB = np.arange(0, 13, 1)   # 0, 1, ..., 12 dB

if HEAVY_RUN:
    # Larger datasets and more epochs for "serious" experiments
    N_BPSK_TRAIN, N_BPSK_VAL = 300_000, 60_000
    N_QPSK_TRAIN, N_QPSK_VAL = 300_000, 60_000
    N_TEST_BPSK, N_TEST_QPSK = 300_000, 100_000

    N_QPSK_FAD_TRAIN, N_QPSK_FAD_VAL = 350_000, 80_000
    N_TEST_QPSK_FAD = 150_000

    EPOCHS_BPSK, EPOCHS_QPSK_AWGN = 30, 35
    EPOCHS_QPSK_FADING = 40
else:
    # Smaller, still reasonable values for debugging / fast runs
    N_BPSK_TRAIN, N_BPSK_VAL = 80_000, 20_000
    N_QPSK_TRAIN, N_QPSK_VAL = 80_000, 20_000
    N_TEST_BPSK, N_TEST_QPSK = 80_000, 40_000

    N_QPSK_FAD_TRAIN, N_QPSK_FAD_VAL = 100_000, 30_000
    N_TEST_QPSK_FAD = 60_000

    EPOCHS_BPSK, EPOCHS_QPSK_AWGN = 12, 14
    EPOCHS_QPSK_FADING = 18

# Mini-batch size
BATCH_SIZE = 512

# Learning rates (for Adam)
LR_BPSK = 3e-3
LR_QPSK_AWGN = 3e-3
LR_QPSK_FADING = 3e-3

# Hidden layer configurations (multiple hidden layers)
HID_BPSK = [32, 32]             # 1D input -> 2-layer MLP
HID_QPSK_AWGN = [32, 32]        # 2D input -> 2-layer MLP
HID_QPSK_FADING = [64, 64]      # 4D input -> 2-layer MLP (slightly larger)

# Rayleigh fading: variance of channel estimation error
EST_VAR_H = 0.1  # h_hat = h + e, E[|e|^2] = EST_VAR_H

# Global random seed
RANDOM_SEED = 7


# =============================================================================
# 2. Utility functions (SNR, noise, theory)
# =============================================================================

def db2lin(db):
    """Convert dB to linear scale: 10^(db/10)."""
    return 10.0 ** (db / 10.0)


def awgn_sigma(EbN0_dB):
    """
    Compute the standard deviation of AWGN for a given Eb/N0 (in dB).

    Assumptions:
      - Bit energy Eb = 1
      - N0 = 1 / (Eb/N0)
      - Real-valued noise with variance sigma^2 = N0 / 2 per real dimension.
    """
    EbN0_lin = db2lin(EbN0_dB)
    N0 = 1.0 / EbN0_lin
    return np.sqrt(N0 / 2.0)


def Qfunc(x):
    """
    Numerically stable Q-function via erf:
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
    X = y.reshape(-1, 1)  # shape (n, 1) for MLP
    return X, bits


def gen_qpsk_awgn(n_symbols, EbN0_dB):
    """
    Generate QPSK samples over AWGN in 2D real representation.

    Bits per symbol: [b0, b1]
    Mapping per dimension: 0 -> +1, 1 -> -1
    Channel:
       yI = I + nI
       yQ = Q + nQ
    """
    bits = np.random.randint(0, 2, size=2 * n_symbols, dtype=np.uint8)
    b0 = bits[0::2]   # I-bit
    b1 = bits[1::2]   # Q-bit

    I = 1 - 2 * b0
    Q = 1 - 2 * b1

    sigma = awgn_sigma(EbN0_dB)
    nI = sigma * np.random.randn(n_symbols)
    nQ = sigma * np.random.randn(n_symbols)

    yI = I + nI
    yQ = Q + nQ

    X = np.stack([yI, yQ], axis=1)
    Y = np.stack([b0, b1], axis=1)
    return X, Y


# =============================================================================
# 4. Dataset generation: QPSK over Rayleigh fading with imperfect CSI
# =============================================================================

def gen_qpsk_rayleigh_imperfect(n_symbols, EbN0_dB, est_var_h=0.1):
    """
    Generate QPSK over Rayleigh fading with imperfect CSI.

    Model:
       s = sI + j*sQ,  sI,sQ in {+1, -1} (QPSK)
       h ~ CN(0,1)
       n ~ CN(0, N0)
       y = h * s + n
       h_hat = h + e,  e ~ CN(0, est_var_h)

    MLP input:
       X = [Re(y), Im(y), Re(h_hat), Im(h_hat)]

    Labels:
       Y = [b0, b1].
    """
    bits = np.random.randint(0, 2, size=2 * n_symbols, dtype=np.uint8)
    b0 = bits[0::2]
    b1 = bits[1::2]

    I = 1 - 2 * b0
    Q = 1 - 2 * b1
    s = I + 1j * Q

    # Rayleigh channel h ~ CN(0, 1)
    hr = np.random.randn(n_symbols) / np.sqrt(2.0)
    hi = np.random.randn(n_symbols) / np.sqrt(2.0)
    h = hr + 1j * hi

    # AWGN noise
    sigma = awgn_sigma(EbN0_dB)
    nr = sigma * np.random.randn(n_symbols)
    ni = sigma * np.random.randn(n_symbols)
    n = nr + 1j * ni

    y = h * s + n

    # Imperfect channel estimate: h_hat = h + e
    er = np.sqrt(est_var_h / 2.0) * np.random.randn(n_symbols)
    ei = np.sqrt(est_var_h / 2.0) * np.random.randn(n_symbols)
    e = er + 1j * ei
    h_hat = h + e

    X = np.stack([y.real, y.imag, h_hat.real, h_hat.imag], axis=1)
    Y = np.stack([b0, b1], axis=1)

    return X, Y, y, h_hat


# =============================================================================
# 5. Multi-layer MLP with Adam optimizer (NumPy only)
# =============================================================================

class MLP:
    """
    Multi-layer perceptron with:
      - arbitrary number of hidden layers,
      - ReLU activation on hidden layers,
      - Sigmoid activation on output layer (multi-label),
      - Binary Cross Entropy loss,
      - Adam optimizer.

    Architecture:
       input_dim -> [hidden_dims...] -> output_dim
    """

    def __init__(self, input_dim, output_dim,
                 hidden_dims=(32, 32),
                 lr=1e-3,
                 seed=1,
                 adam_beta1=0.9,
                 adam_beta2=0.999,
                 adam_eps=1e-8):
        rng = np.random.default_rng(seed)

        # Build layer dimensions list: [in, h1, h2, ..., out]
        self.layer_dims = [input_dim] + list(hidden_dims) + [output_dim]
        self.L = len(self.layer_dims) - 1  # number of layers with params

        # Initialize parameters: W[l], b[l]
        self.params = {}
        for l in range(1, self.L + 1):
            in_dim = self.layer_dims[l - 1]
            out_dim = self.layer_dims[l]
            # He initialization for ReLU layers; similar scale for last layer
            std = np.sqrt(2.0 / in_dim)
            self.params[f"W{l}"] = rng.normal(0, std, size=(in_dim, out_dim))
            self.params[f"b{l}"] = np.zeros((1, out_dim))

        # Adam optimizer state
        self.lr = lr
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.eps = adam_eps
        self.m = {f"W{l}": np.zeros_like(self.params[f"W{l}"])
                  for l in range(1, self.L + 1)}
        self.v = {f"W{l}": np.zeros_like(self.params[f"W{l}"])
                  for l in range(1, self.L + 1)}
        self.m.update({f"b{l}": np.zeros_like(self.params[f"b{l}"])
                       for l in range(1, self.L + 1)})
        self.v.update({f"b{l}": np.zeros_like(self.params[f"b{l}"])
                       for l in range(1, self.L + 1)})
        self.t = 0  # Adam time step

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(x.dtype)

    @staticmethod
    def bce_loss(Yhat, Y):
        """
        Binary Cross Entropy loss:
           L = - mean( Y*log(Yhat) + (1-Y)*log(1-Yhat) )
        """
        eps = 1e-12
        Yhat_clipped = np.clip(Yhat, eps, 1 - eps)
        return -np.mean(Y * np.log(Yhat_clipped) +
                        (1 - Y) * np.log(1 - Yhat_clipped))

    def forward(self, X):
        """
        Forward pass through all L layers.

        For layers 1..L-1: affine + ReLU
        For layer L: affine + Sigmoid
        """
        cache = {"A0": X}
        A = X
        for l in range(1, self.L):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = A @ W + b
            A = self.relu(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        # Last layer with Sigmoid
        W = self.params[f"W{self.L}"]
        b = self.params[f"b{self.L}"]
        Z = A @ W + b
        Yhat = self.sigmoid(Z)
        cache[f"Z{self.L}"] = Z
        cache[f"A{self.L}"] = Yhat

        return Yhat, cache

    def backward(self, cache, Y):
        """
        Backward pass: compute gradients for all layers.
        """
        grads = {}
        m = Y.shape[0]
        A_prev = cache[f"A{self.L - 1}"] if self.L > 1 else cache["A0"]
        Yhat = cache[f"A{self.L}"]
        ZL = cache[f"Z{self.L}"]

        # Derivative for last layer (Sigmoid + BCE)
        dZL = (Yhat - Y) / m
        grads[f"dW{self.L}"] = A_prev.T @ dZL
        grads[f"db{self.L}"] = np.sum(dZL, axis=0, keepdims=True)

        dA_prev = dZL @ self.params[f"W{self.L}"].T

        # Hidden layers (ReLU)
        for l in range(self.L - 1, 0, -1):
            Z = cache[f"Z{l}"]
            A_prev = cache[f"A{l-1}"] if l > 1 else cache["A0"]

            dZ = dA_prev * self.relu_deriv(Z)
            grads[f"dW{l}"] = A_prev.T @ dZ
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)

            if l > 1:
                dA_prev = dZ @ self.params[f"W{l}"].T

        return grads

    def adam_update(self, grads):
        """
        Perform one Adam update step using the gradients 'grads'.
        """
        self.t += 1
        for l in range(1, self.L + 1):
            for param_name in ["W", "b"]:
                key = f"{param_name}{l}"
                g = grads[f"d{param_name}{l}"]
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g * g)

                # Bias-corrected estimates
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                # Parameter update
                self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def fit(self, X, Y, Xval=None, Yval=None,
            epochs=10, batch_size=256, verbose=True, label="model"):
        """
        Training loop with mini-batch Adam.

        Returns:
            train_losses, val_losses (lists)
        """
        n = X.shape[0]
        train_losses, val_losses = [], []

        for ep in range(epochs):
            # Shuffle training data
            idx = np.random.permutation(n)
            Xs, Ys = X[idx], Y[idx]

            # Mini-batch training
            for i in range(0, n, batch_size):
                xb = Xs[i:i + batch_size]
                yb = Ys[i:i + batch_size]

                Yhat, cache = self.forward(xb)
                grads = self.backward(cache, yb)
                self.adam_update(grads)

            # Epoch losses
            Yhat_all, _ = self.forward(X)
            loss = self.bce_loss(Yhat_all, Y)
            train_losses.append(float(loss))

            if Xval is not None and Yval is not None:
                Yhat_val, _ = self.forward(Xval)
                vloss = self.bce_loss(Yhat_val, Yval)
                val_losses.append(float(vloss))
                if verbose:
                    print(f"[{label}] Epoch {ep+1}/{epochs}: "
                          f"loss={loss:.4f}, val_loss={vloss:.4f}")
            else:
                if verbose:
                    print(f"[{label}] Epoch {ep+1}/{epochs}: loss={loss:.4f}")

        return train_losses, val_losses

    def predict_bits(self, X):
        """
        Predict bits (0/1) by thresholding sigmoid outputs at 0.5.
        """
        Yhat, _ = self.forward(X)
        return (Yhat >= 0.5).astype(np.uint8)


# =============================================================================
# 6. Evaluation utilities (BER computation)
# =============================================================================

def evaluate_ber_bpsk_awgn(model, EbN0_dB_list, n_test=100_000):
    """
    Evaluate BER for BPSK over AWGN:
       - MLP-based detector
       - Classical optimal detector (threshold at 0)
    """
    ber_mlp, ber_opt = [], []
    for eb in EbN0_dB_list:
        X, bits = gen_bpsk_awgn(n_test, eb)

        bhat_mlp = model.predict_bits(X).reshape(-1)
        ber_mlp.append(np.mean(bhat_mlp != bits))

        # classical detector: sign threshold
        bhat_opt = (X.reshape(-1) < 0).astype(np.uint8)
        ber_opt.append(np.mean(bhat_opt != bits))

    return np.array(ber_mlp), np.array(ber_opt)


def evaluate_ber_qpsk_awgn(model, EbN0_dB_list, n_test=50_000):
    """
    Evaluate BER for QPSK over AWGN:
       - MLP-based detector
       - Classical optimal detector (sign per axis)
    """
    ber_mlp, ber_opt = [], []
    for eb in EbN0_dB_list:
        X, Y = gen_qpsk_awgn(n_test, eb)

        Yhat_mlp = model.predict_bits(X)
        ber_mlp.append(np.mean(Yhat_mlp != Y))

        b0_opt = (X[:, 0] < 0).astype(np.uint8)
        b1_opt = (X[:, 1] < 0).astype(np.uint8)
        Yopt = np.stack([b0_opt, b1_opt], axis=1)
        ber_opt.append(np.mean(Yopt != Y))

    return np.array(ber_mlp), np.array(ber_opt)


def evaluate_ber_qpsk_rayleigh(model, EbN0_dB_list,
                               n_test=50_000, est_var_h=0.1):
    """
    Evaluate BER for QPSK over Rayleigh fading with imperfect CSI:
       - MLP-based detector
       - Classical mismatched detector:
            y_eq = y / h_hat, then AWGN-like detection.
    """
    ber_mlp, ber_mis = [], []
    for eb in EbN0_dB_list:
        X, Y_true, y, h_hat = gen_qpsk_rayleigh_imperfect(
            n_test, eb, est_var_h=est_var_h
        )

        # MLP-based
        Yhat_mlp = model.predict_bits(X)
        ber_mlp.append(np.mean(Yhat_mlp != Y_true))

        # mismatched classical detector
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
    np.random.seed(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # 7.1 BPSK over AWGN
    # -------------------------------------------------------------------------
    print("\n=== Training BPSK MLP (AWGN) with Adam ===")
    Xtr_b, Ytr_b = gen_bpsk_awgn(N_BPSK_TRAIN, TRAIN_SNR_DB_AWGN)
    Xval_b, Yval_b = gen_bpsk_awgn(N_BPSK_VAL, TRAIN_SNR_DB_AWGN)

    Ytr_b = Ytr_b.reshape(-1, 1).astype(np.float64)
    Yval_b = Yval_b.reshape(-1, 1).astype(np.float64)

    mlp_bpsk = MLP(
        input_dim=1,
        output_dim=1,
        hidden_dims=HID_BPSK,
        lr=LR_BPSK,
        seed=1
    )

    losses_b, vlosses_b = mlp_bpsk.fit(
        Xtr_b, Ytr_b,
        Xval_b, Yval_b,
        epochs=EPOCHS_BPSK,
        batch_size=BATCH_SIZE,
        verbose=True,
        label="BPSK-AWGN"
    )

    # -------------------------------------------------------------------------
    # 7.2 QPSK over AWGN
    # -------------------------------------------------------------------------
    print("\n=== Training QPSK MLP (AWGN) with Adam ===")
    Xtr_q, Ytr_q = gen_qpsk_awgn(N_QPSK_TRAIN, TRAIN_SNR_DB_AWGN)
    Xval_q, Yval_q = gen_qpsk_awgn(N_QPSK_VAL, TRAIN_SNR_DB_AWGN)

    Ytr_q = Ytr_q.astype(np.float64)
    Yval_q = Yval_q.astype(np.float64)

    mlp_qpsk_awgn = MLP(
        input_dim=2,
        output_dim=2,
        hidden_dims=HID_QPSK_AWGN,
        lr=LR_QPSK_AWGN,
        seed=2
    )

    losses_q_awgn, vlosses_q_awgn = mlp_qpsk_awgn.fit(
        Xtr_q, Ytr_q,
        Xval_q, Yval_q,
        epochs=EPOCHS_QPSK_AWGN,
        batch_size=BATCH_SIZE,
        verbose=True,
        label="QPSK-AWGN"
    )

    # -------------------------------------------------------------------------
    # 7.3 QPSK over Rayleigh fading with imperfect CSI
    # -------------------------------------------------------------------------
    print("\n=== Training QPSK MLP (Rayleigh fading, imperfect CSI) with Adam ===")
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
        input_dim=4,
        output_dim=2,
        hidden_dims=HID_QPSK_FADING,
        lr=LR_QPSK_FADING,
        seed=3
    )

    losses_q_fad, vlosses_q_fad = mlp_qpsk_fading.fit(
        Xtr_f, Ytr_f,
        Xval_f, Yval_f,
        epochs=EPOCHS_QPSK_FADING,
        batch_size=BATCH_SIZE,
        verbose=True,
        label="QPSK-FADING"
    )

    # -------------------------------------------------------------------------
    # 7.4 Evaluate BER (AWGN baselines)
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
    # 7.5 Evaluate BER (Rayleigh fading scenario)
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
    with open("ber_awgn_results_adam.csv", "w", newline="") as f:
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

    with open("ber_fading_results_adam.csv", "w", newline="") as f:
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
    # 7.7 Plot AWGN BER curves
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
    plt.title("ML-based Detector vs Optimal & Theory (BPSK & QPSK, AWGN, Adam)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(loc="upper right")
    plt.ylim(1e-5, 1)
    plt.tight_layout()
    plt.savefig("ber_awgn_curves_adam.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -------------------------------------------------------------------------
    # 7.8 Plot Rayleigh fading BER curves
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.semilogy(EVAL_SNRS_DB, ber_mis_fad, "--", linewidth=1.8,
                 label="QPSK mismatched detector (fading)")
    plt.semilogy(EVAL_SNRS_DB, ber_mlp_fad, "o-", markersize=4,
                 label="QPSK MLP (fading, imperfect CSI, Adam)")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("QPSK over Rayleigh Fading (Imperfect CSI): "
              "MLP vs Mismatched Detector (Adam)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(loc="upper right")
    plt.ylim(1e-4, 1)
    plt.tight_layout()
    plt.savefig("ber_fading_curves_adam.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -------------------------------------------------------------------------
    # 7.9 Plot training / validation losses
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
    plt.title("Training / Validation Losses (All MLP Models, Adam)")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_losses_all_adam.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -------------------------------------------------------------------------
    # 7.10 Diagnostic scatterplots (optional, for visualization)
    # -------------------------------------------------------------------------
    Xdiag, Ydiag = gen_qpsk_awgn(6000, TRAIN_SNR_DB_AWGN)
    Yhat_diag = mlp_qpsk_awgn.predict_bits(Xdiag)
    correct = np.all(Yhat_diag == Ydiag, axis=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(Xdiag[correct, 0], Xdiag[correct, 1],
                s=8, alpha=0.5, label="Correct")
    plt.scatter(Xdiag[~correct, 0], Xdiag[~correct, 1],
                s=8, alpha=0.5, label="Wrong")
    plt.title("QPSK MLP Decisions @ 6 dB (AWGN, Adam)")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qpsk_mlp_decisions_awgn_6dB_adam.png",
                dpi=250, bbox_inches="tight")
    plt.show()

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
    plt.title("QPSK MLP Decisions @ 6 dB "
              "(Rayleigh fading, imperfect CSI, Adam)")
    plt.xlabel("Re{y}")
    plt.ylabel("Im{y}")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qpsk_mlp_decisions_fading_6dB_adam.png",
                dpi=250, bbox_inches="tight")
    plt.show()

    print("\nSaved files:")
    print("  - ber_awgn_results_adam.csv")
    print("  - ber_fading_results_adam.csv")
    print("  - ber_awgn_curves_adam.png")
    print("  - ber_fading_curves_adam.png")
    print("  - training_losses_all_adam.png")
    print("  - qpsk_mlp_decisions_awgn_6dB_adam.png")
    print("  - qpsk_mlp_decisions_fading_6dB_adam.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
