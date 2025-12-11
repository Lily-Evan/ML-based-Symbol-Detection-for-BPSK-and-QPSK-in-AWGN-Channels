#!/usr/bin/env python3
# ψηφιακατηλεπεργασια.py
#
# ML-Based Symbol Detection for BPSK & QPSK
# AWGN & Rayleigh Fading
#
# NumPy-only implementation of:
# - Dataset generation (BPSK/QPSK)
# - AWGN & Rayleigh channels (with imperfect CSI)
# - MLP neural receiver (any depth)
# - BER vs Eb/N0 evaluation
# - Classical optimal / mismatched detectors

import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from typing import Tuple, Dict, List
from math import erfc

# ============================================================
# Utility functions
# ============================================================

def db2lin(x_db: float) -> float:
    """Convert dB to linear scale."""
    return 10.0 ** (x_db / 10.0)


def qfunc(x: np.ndarray) -> np.ndarray:
    """
    Q-function using math.erfc via numpy.vectorize:
    Q(x) = 0.5 * erfc(x / sqrt(2)).
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * np.vectorize(erfc)(x / np.sqrt(2.0))


# ============================================================
# Modulation / Demodulation
# ============================================================

def bpsk_mod(bits: np.ndarray) -> np.ndarray:
    """BPSK modulation: 0 -> +1, 1 -> -1."""
    return 1 - 2 * bits  # 0 -> 1, 1 -> -1


def bpsk_demod(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision BPSK demodulation."""
    return (symbols < 0).astype(int)


def qpsk_mod(bits: np.ndarray) -> np.ndarray:
    """
    QPSK modulation (Gray mapping).
    Input: bits of shape (N,) with even length.
    Output: complex symbols of shape (N/2,)
    Mapping:
        00 -> +1 + 1j
        01 -> -1 + 1j
        11 -> -1 - 1j
        10 -> +1 - 1j
    Normalized by 1/sqrt(2).
    """
    assert bits.size % 2 == 0, "QPSK needs even number of bits."
    b0 = bits[0::2]
    b1 = bits[1::2]

    I = 1 - 2 * b0
    Q = 1 - 2 * b1
    symbols = (I + 1j * Q) / np.sqrt(2.0)
    return symbols


def qpsk_demod(symbols: np.ndarray) -> np.ndarray:
    """
    Hard-decision QPSK demodulation (inverse of qpsk_mod).
    Output bits shape: (2 * len(symbols),)
    """
    sym = symbols * np.sqrt(2.0)
    I = np.real(sym)
    Q = np.imag(sym)

    b0 = (I < 0).astype(int)
    b1 = (Q < 0).astype(int)

    bits = np.empty(2 * len(symbols), dtype=int)
    bits[0::2] = b0
    bits[1::2] = b1
    return bits


# ============================================================
# Channel models
# ============================================================

def awgn_channel(x: np.ndarray, ebn0_db: float, bits_per_symbol: int = 1) -> np.ndarray:
    """
    AWGN channel for BPSK/QPSK.
    x: symbols (real or complex)
    Eb/N0 given in dB.
    """
    ebn0_lin = db2lin(ebn0_db)
    # For unit-energy constellations, Es = 1, Eb = Es / bits_per_symbol
    # N0 = Eb / (Eb/N0), noise variance per dimension = N0/2
    eb = 1.0 / bits_per_symbol
    n0 = eb / ebn0_lin
    noise_var = n0 / 2.0

    if np.iscomplexobj(x):
        noise = np.sqrt(noise_var) * (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape))
    else:
        noise = np.sqrt(noise_var) * np.random.randn(*x.shape)
    return x + noise


def rayleigh_fading_channel(
    x: np.ndarray,
    ebn0_db: float,
    bits_per_symbol: int = 1,
    imperfect_csi: bool = True,
    csi_error_var: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rayleigh flat fading channel.
    y = h * x + n, h ~ CN(0,1)
    Returns:
        y: received symbols
        h: true channel coefficients
        h_hat: (possibly imperfect) channel estimates
    """
    h = (np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)) / np.sqrt(2.0)
    x_faded = h * x
    y = awgn_channel(x_faded, ebn0_db, bits_per_symbol=bits_per_symbol)

    if imperfect_csi:
        e = np.sqrt(csi_error_var / 2.0) * (
            np.random.randn(*h.shape) + 1j * np.random.randn(*h.shape)
        )
        h_hat = h + e
    else:
        h_hat = h

    return y, h, h_hat


# ============================================================
# Classical detectors
# ============================================================

def classical_detector_awgn_bpsk(y: np.ndarray) -> np.ndarray:
    """Optimal BPSK detector in AWGN."""
    return bpsk_demod(y)


def classical_detector_awgn_qpsk(y: np.ndarray) -> np.ndarray:
    """Optimal QPSK detector in AWGN."""
    return qpsk_demod(y)


def mismatched_detector_rayleigh_qpsk(y: np.ndarray, h_hat: np.ndarray) -> np.ndarray:
    """
    Classical mismatched detector in Rayleigh fading:
    equalize with estimated channel and then apply AWGN QPSK detector.
    """
    eps = 1e-12
    y_eq = y / (h_hat + eps)
    return qpsk_demod(y_eq)


# ============================================================
# MLP implementation (NumPy, generic depth)
# ============================================================

class MLP:
    def __init__(self, layer_sizes: List[int], lr: float = 1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Generic fully-connected MLP with ReLU hidden layers and sigmoid output.
        Adam-like optimizer implemented in NumPy.

        layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
        """
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.num_layers = len(layer_sizes) - 1
        self.params = {}
        self.opt_state = {}
        self._init_params()

    def _init_params(self):
        for l in range(1, len(self.layer_sizes)):
            in_dim = self.layer_sizes[l - 1]
            out_dim = self.layer_sizes[l]
            W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
            b = np.zeros((1, out_dim))
            self.params[f"W{l}"] = W
            self.params[f"b{l}"] = b

            self.opt_state[f"mW{l}"] = np.zeros_like(W)
            self.opt_state[f"vW{l}"] = np.zeros_like(W)
            self.opt_state[f"mb{l}"] = np.zeros_like(b)
            self.opt_state[f"vb{l}"] = np.zeros_like(b)

        self.t = 0  # Adam time step

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_grad(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass.
        Returns cache with layer outputs and pre-activations.
        """
        cache = {"A0": X}
        A = X
        for l in range(1, self.num_layers + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            Z = A @ W + b
            cache[f"Z{l}"] = Z
            if l == self.num_layers:
                A = self.sigmoid(Z)
            else:
                A = self.relu(Z)
            cache[f"A{l}"] = A
        return cache

    @staticmethod
    def bce_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return loss

    def backward(self, cache: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, np.ndarray]:
        grads = {}
        A_out = cache[f"A{self.num_layers}"]
        m = y_true.shape[0]

        dA = (A_out - y_true) / m

        for l in reversed(range(1, self.num_layers + 1)):
            Z = cache[f"Z{l}"]
            A_prev = cache[f"A{l-1}"]
            W = self.params[f"W{l}"]

            if l == self.num_layers:
                dZ = dA
            else:
                dZ = dA * self.relu_grad(Z)

            dW = A_prev.T @ dZ
            db = np.sum(dZ, axis=0, keepdims=True)
            dA = dZ @ W.T

            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db

        return grads

    def update_params(self, grads: Dict[str, np.ndarray]):
        self.t += 1
        for l in range(1, self.num_layers + 1):
            for param_name in ["W", "b"]:
                key = f"{param_name}{l}"
                m_key = f"m{param_name}{l}"
                v_key = f"v{param_name}{l}"

                g = grads[f"d{param_name}{l}"]
                m = self.opt_state[m_key]
                v = self.opt_state[v_key]

                m = self.beta1 * m + (1 - self.beta1) * g
                v = self.beta2 * v + (1 - self.beta2) * (g ** 2)

                m_hat = m / (1 - self.beta1 ** self.t)
                v_hat = v / (1 - self.beta2 ** self.t)

                self.params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

                self.opt_state[m_key] = m
                self.opt_state[v_key] = v

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        cache = self.forward(X)
        return cache[f"A{self.num_layers}"]

    def predict_bits(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 10,
        batch_size: int = 256,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        n_samples = X_train.shape[0]
        history = {"loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n_samples)
            X_train_shuf = X_train[idx]
            y_train_shuf = y_train[idx]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_train_shuf[start:end]
                y_batch = y_train_shuf[start:end]

                cache = self.forward(X_batch)
                loss = self.bce_loss(cache[f"A{self.num_layers}"], y_batch)
                grads = self.backward(cache, y_batch)
                self.update_params(grads)

            train_pred = self.predict_proba(X_train)
            train_loss = self.bce_loss(train_pred, y_train)
            val_pred = self.predict_proba(X_val)
            val_loss = self.bce_loss(val_pred, y_val)

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if verbose:
                print(
                    f"Epoch {epoch}/{epochs}: loss={train_loss:.4f} val_loss={val_loss:.4f}"
                )

        return history


# ============================================================
# Dataset builders for neural receiver
# ============================================================

def build_dataset_bpsk_awgn(
    n_bits: int,
    ebn0_db: float
) -> Tuple[np.ndarray, np.ndarray]:
    bits = np.random.randint(0, 2, size=n_bits)
    x = bpsk_mod(bits)
    y = awgn_channel(x, ebn0_db, bits_per_symbol=1)
    X = y.reshape(-1, 1).astype(np.float32)
    y_bits = bits.reshape(-1, 1).astype(np.float32)
    return X, y_bits


def build_dataset_qpsk_awgn(
    n_bits: int,
    ebn0_db: float
) -> Tuple[np.ndarray, np.ndarray]:
    # n_bits must be even
    if n_bits % 2 != 0:
        n_bits += 1

    bits = np.random.randint(0, 2, size=n_bits)
    x = qpsk_mod(bits)  # N_symbols = n_bits / 2
    y = awgn_channel(x, ebn0_db, bits_per_symbol=2)

    # Features per symbol: [Re(y), Im(y)]
    X = np.stack([np.real(y), np.imag(y)], axis=1).astype(np.float32)

    # Labels: two bits per symbol -> (N_symbols, 2)
    n_symbols = x.size
    bits_sym = bits.reshape(n_symbols, 2)
    y_bits = bits_sym.astype(np.float32)

    return X, y_bits


def build_dataset_qpsk_rayleigh_imperfect_csi(
    n_bits: int,
    ebn0_db: float,
    csi_error_var: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    # n_bits must be even
    if n_bits % 2 != 0:
        n_bits += 1

    bits = np.random.randint(0, 2, size=n_bits)
    x = qpsk_mod(bits)
    y, h, h_hat = rayleigh_fading_channel(
        x, ebn0_db, bits_per_symbol=2, imperfect_csi=True, csi_error_var=csi_error_var
    )

    # Features per symbol: [Re(y), Im(y), Re(h_hat), Im(h_hat)]
    X = np.stack(
        [np.real(y), np.imag(y), np.real(h_hat), np.imag(h_hat)], axis=1
    ).astype(np.float32)

    n_symbols = x.size
    bits_sym = bits.reshape(n_symbols, 2)
    y_bits = bits_sym.astype(np.float32)

    return X, y_bits


# ============================================================
# BER computation utilities
# ============================================================

def compute_ber(true_bits: np.ndarray, pred_bits: np.ndarray) -> float:
    true_bits = np.asarray(true_bits).ravel()
    pred_bits = np.asarray(pred_bits).ravel()
    assert true_bits.shape == pred_bits.shape
    return np.mean(true_bits != pred_bits)


def theoretical_ber_bpsk_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    ebn0_lin = db2lin(ebn0_db)
    return qfunc(np.sqrt(2 * ebn0_lin))


def theoretical_ber_qpsk_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    # Same per-bit BER as BPSK in AWGN
    return theoretical_ber_bpsk_awgn(ebn0_db)


# ============================================================
# High-level experiments
# ============================================================

def train_mlp_for_scenario(
    scenario: str = "bpsk_awgn",
    ebn0_db_train: float = 6.0,
    n_bits_train: int = 100_000,
    n_bits_val: int = 20_000,
    hidden_sizes: List[int] = [64, 64],
    lr: float = 1e-3,
    epochs: int = 10,
) -> Tuple[MLP, Dict[str, List[float]]]:
    print(f"\n=== Training MLP for scenario: {scenario} at Eb/N0={ebn0_db_train} dB ===")

    if scenario == "bpsk_awgn":
        X_train, y_train = build_dataset_bpsk_awgn(n_bits_train, ebn0_db_train)
        X_val, y_val = build_dataset_bpsk_awgn(n_bits_val, ebn0_db_train)

    elif scenario == "qpsk_awgn":
        X_train, y_train = build_dataset_qpsk_awgn(n_bits_train, ebn0_db_train)
        X_val, y_val = build_dataset_qpsk_awgn(n_bits_val, ebn0_db_train)

    elif scenario == "qpsk_rayleigh":
        X_train, y_train = build_dataset_qpsk_rayleigh_imperfect_csi(
            n_bits_train, ebn0_db_train
        )
        X_val, y_val = build_dataset_qpsk_rayleigh_imperfect_csi(
            n_bits_val, ebn0_db_train
        )

    else:
        raise ValueError("Unknown scenario")

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]  # 1 for BPSK, 2 for QPSK

    layer_sizes = [input_dim] + hidden_sizes + [output_dim]
    mlp = MLP(layer_sizes, lr=lr)

    history = mlp.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=512,
        verbose=True
    )

    return mlp, history


def evaluate_ber_curve_mlp(
    mlp: MLP,
    scenario: str,
    ebn0_db_list: List[float],
    n_bits: int = 200_000
) -> List[float]:
    ber_list = []
    for ebn0_db in ebn0_db_list:
        print(f"[MLP] Evaluating {scenario} at Eb/N0={ebn0_db:.1f} dB...")

        if scenario == "bpsk_awgn":
            bits = np.random.randint(0, 2, size=n_bits)
            x = bpsk_mod(bits)
            y = awgn_channel(x, ebn0_db, bits_per_symbol=1)
            X = y.reshape(-1, 1).astype(np.float32)
            y_hat_bits = mlp.predict_bits(X).ravel()
            ber = compute_ber(bits, y_hat_bits)

        elif scenario == "qpsk_awgn":
            if n_bits % 2 != 0:
                n_bits += 1
            bits = np.random.randint(0, 2, size=n_bits)
            x = qpsk_mod(bits)
            y = awgn_channel(x, ebn0_db, bits_per_symbol=2)
            X = np.stack([np.real(y), np.imag(y)], axis=1).astype(np.float32)

            n_symbols = x.size
            bits_sym = bits.reshape(n_symbols, 2)
            y_hat_sym = mlp.predict_bits(X).astype(int)

            bits_flat = bits_sym.ravel()
            y_hat_flat = y_hat_sym.ravel()
            ber = compute_ber(bits_flat, y_hat_flat)

        elif scenario == "qpsk_rayleigh":
            if n_bits % 2 != 0:
                n_bits += 1
            bits = np.random.randint(0, 2, size=n_bits)
            x = qpsk_mod(bits)
            y, h, h_hat = rayleigh_fading_channel(
                x, ebn0_db, bits_per_symbol=2, imperfect_csi=True
            )
            X = np.stack(
                [np.real(y), np.imag(y), np.real(h_hat), np.imag(h_hat)], axis=1
            ).astype(np.float32)

            n_symbols = x.size
            bits_sym = bits.reshape(n_symbols, 2)
            y_hat_sym = mlp.predict_bits(X).astype(int)

            bits_flat = bits_sym.ravel()
            y_hat_flat = y_hat_sym.ravel()
            ber = compute_ber(bits_flat, y_hat_flat)

        else:
            raise ValueError("Unknown scenario")

        ber_list.append(ber)
    return ber_list


def evaluate_ber_curve_classical_qpsk_rayleigh(
    ebn0_db_list: List[float],
    n_bits: int = 200_000
) -> List[float]:
    ber_list = []
    for ebn0_db in ebn0_db_list:
        print(f"[Classical] QPSK Rayleigh mismatched at Eb/N0={ebn0_db:.1f} dB...")
        if n_bits % 2 != 0:
            n_bits += 1
        bits = np.random.randint(0, 2, size=n_bits)
        x = qpsk_mod(bits)
        y, h, h_hat = rayleigh_fading_channel(
            x, ebn0_db, bits_per_symbol=2, imperfect_csi=True
        )
        bits_hat = mismatched_detector_rayleigh_qpsk(y, h_hat)
        ber = compute_ber(bits, bits_hat)
        ber_list.append(ber)
    return ber_list


# ============================================================
# Plot & CSV helpers
# ============================================================

def save_ber_csv(filename: str, ebn0_db_list: List[float], ber_dict: Dict[str, List[float]]):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["EbN0_dB"] + list(ber_dict.keys())
        writer.writerow(header)
        for i, ebn0 in enumerate(ebn0_db_list):
            row = [ebn0]
            for key in ber_dict.keys():
                row.append(ber_dict[key][i])
            writer.writerow(row)
    print(f"Saved BER CSV to: {filename}")


def plot_ber_curves(
    filename: str,
    ebn0_db_list: List[float],
    ber_dict: Dict[str, List[float]],
    title: str
):
    plt.figure()
    for label, ber in ber_dict.items():
        plt.semilogy(ebn0_db_list, ber, marker="o", label=label)
    plt.grid(True, which="both")
    plt.xlabel("Eb/N0 [dB]")
    plt.ylabel("BER")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved BER plot to: {filename}")


def plot_training_curves(
    filename: str,
    history: Dict[str, List[float]],
    title: str
):
    plt.figure()
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved training curve to: {filename}")


# ============================================================
# Example main experiment
# ============================================================

def main():
    out_dir = Path(".")
    out_dir.mkdir(exist_ok=True)

    ebn0_db_list = [0, 2, 4, 6, 8, 10]
    ebn0_arr = np.array(ebn0_db_list)

    # 1) BPSK AWGN
    mlp_bpsk, hist_bpsk = train_mlp_for_scenario(
        scenario="bpsk_awgn",
        ebn0_db_train=6.0,
        n_bits_train=80_000,
        n_bits_val=20_000,
        hidden_sizes=[32],
        lr=5e-3,
        epochs=10
    )

    ber_mlp_bpsk = evaluate_ber_curve_mlp(
        mlp_bpsk, "bpsk_awgn", ebn0_db_list, n_bits=100_000
    )
    ber_theory_bpsk = theoretical_ber_bpsk_awgn(ebn0_arr)

    ber_dict_bpsk = {
        "MLP": ber_mlp_bpsk,
        "Theory_BPSK": list(ber_theory_bpsk)
    }
    save_ber_csv(str(out_dir / "ber_bpsk_awgn_results.csv"), ebn0_db_list, ber_dict_bpsk)
    plot_ber_curves(
        str(out_dir / "ber_bpsk_awgn_curves.png"),
        ebn0_db_list,
        ber_dict_bpsk,
        title="BPSK in AWGN: BER"
    )
    plot_training_curves(
        str(out_dir / "training_bpsk_awgn.png"),
        hist_bpsk,
        title="BPSK MLP Training (AWGN)"
    )

    # 2) QPSK AWGN
    mlp_qpsk_awgn, hist_qpsk_awgn = train_mlp_for_scenario(
        scenario="qpsk_awgn",
        ebn0_db_train=6.0,
        n_bits_train=100_000,
        n_bits_val=20_000,
        hidden_sizes=[64, 64],
        lr=5e-3,
        epochs=12
    )

    ber_mlp_qpsk_awgn = evaluate_ber_curve_mlp(
        mlp_qpsk_awgn, "qpsk_awgn", ebn0_db_list, n_bits=120_000
    )
    ber_theory_qpsk = theoretical_ber_qpsk_awgn(ebn0_arr)

    ber_dict_qpsk_awgn = {
        "MLP": ber_mlp_qpsk_awgn,
        "Theory_QPSK": list(ber_theory_qpsk)
    }

    save_ber_csv(
        str(out_dir / "ber_qpsk_awgn_results.csv"),
        ebn0_db_list,
        ber_dict_qpsk_awgn
    )
    plot_ber_curves(
        str(out_dir / "ber_qpsk_awgn_curves.png"),
        ebn0_db_list,
        ber_dict_qpsk_awgn,
        title="QPSK in AWGN: BER"
    )
    plot_training_curves(
        str(out_dir / "training_qpsk_awgn.png"),
        hist_qpsk_awgn,
        title="QPSK MLP Training (AWGN)"
    )

    # 3) QPSK Rayleigh fading with imperfect CSI
    mlp_qpsk_rayleigh, hist_qpsk_fading = train_mlp_for_scenario(
        scenario="qpsk_rayleigh",
        ebn0_db_train=6.0,
        n_bits_train=120_000,
        n_bits_val=30_000,
        hidden_sizes=[64, 64],
        lr=3e-3,
        epochs=15
    )

    ber_mlp_qpsk_rayleigh = evaluate_ber_curve_mlp(
        mlp_qpsk_rayleigh, "qpsk_rayleigh", ebn0_db_list, n_bits=150_000
    )
    ber_classical_mismatched = evaluate_ber_curve_classical_qpsk_rayleigh(
        ebn0_db_list, n_bits=150_000
    )

    ber_dict_qpsk_fading = {
        "MLP": ber_mlp_qpsk_rayleigh,
        "Mismatched_detector": ber_classical_mismatched
    }

    save_ber_csv(
        str(out_dir / "ber_qpsk_fading_results.csv"),
        ebn0_db_list,
        ber_dict_qpsk_fading
    )
    plot_ber_curves(
        str(out_dir / "ber_qpsk_fading_curves.png"),
        ebn0_db_list,
        ber_dict_qpsk_fading,
        title="QPSK Rayleigh Fading (Imperfect CSI): BER"
    )
    plot_training_curves(
        str(out_dir / "training_qpsk_rayleigh.png"),
        hist_qpsk_fading,
        title="QPSK MLP Training (Rayleigh, imperfect CSI)"
    )

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
