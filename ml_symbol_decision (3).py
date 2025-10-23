
"""
ML-based symbol decision for BPSK & QPSK over AWGN
- Generates labeled datasets (train/val/test) at multiple Eb/N0
- Trains a tiny 1-hidden-layer MLP (NumPy-only) to predict bits from noisy observations
- Compares BER vs the optimal (classical) detector and vs THEORY (Q-function)
- Saves plots & CSVs; shows plots on screen
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# ---------------- Parameters (easy toggles) ----------------
TRAIN_SNR_DB = 6.0
EVAL_SNRS_DB = np.arange(0, 13, 1)

# Use larger sizes for final experiments; smaller for quick demo
N_BPSK_TRAIN, N_BPSK_VAL = 8_000, 2_000   # e.g., 200_000, 50_000
N_QPSK_TRAIN, N_QPSK_VAL = 6_000, 2_000   # e.g., 150_000, 40_000
N_TEST_BPSK, N_TEST_QPSK = 5_000, 3_000   # e.g., 200_000, 80_000

EPOCHS_BPSK, EPOCHS_QPSK = 4, 5            # e.g., 12, 15
BATCH_SIZE = 512
LR_BPSK, LR_QPSK = 5e-3, 7e-3
HID_BPSK, HID_QPSK = 16, 24
RANDOM_SEED = 7

# -------------- Utilities --------------
def db2lin(db):
    return 10.0**(db/10.0)

def awgn_sigma(EbN0_dB):
    EbN0_lin = db2lin(EbN0_dB)
    N0 = 1.0 / EbN0_lin  # Eb=1
    return np.sqrt(N0/2.0)  # per real dimension

def Qfunc(x):
    # Stable Q-function via erf; vectorized
    import numpy as _np, math as _math
    x = _np.asarray(x, dtype=float)
    vec = _np.vectorize(lambda t: 0.5*(1.0 - _math.erf(t/_np.sqrt(2.0))))
    return vec(x)

def theory_ber_bit(EbN0_dB):
    EbN0 = db2lin(EbN0_dB)
    return Qfunc(np.sqrt(2*EbN0))

def gen_bpsk(n, EbN0_dB):
    bits = np.random.randint(0, 2, size=n, dtype=np.uint8)
    s = 1 - 2*bits  # 0->+1, 1->-1
    sigma = awgn_sigma(EbN0_dB)
    y = s + sigma*np.random.randn(n)
    return y.reshape(-1,1), bits

def gen_qpsk(n_symbols, EbN0_dB):
    bits = np.random.randint(0, 2, size=2*n_symbols, dtype=np.uint8)
    b0 = bits[0::2]
    b1 = bits[1::2]
    I = 1 - 2*b0
    Q = 1 - 2*b1
    sigma = awgn_sigma(EbN0_dB)
    yI = I + sigma*np.random.randn(n_symbols)
    yQ = Q + sigma*np.random.randn(n_symbols)
    X = np.stack([yI, yQ], axis=1)
    # labels are two bits per symbol
    Y = np.stack([b0, b1], axis=1)
    return X, Y

# -------------- MLP (NumPy) --------------
class MLP:
    def __init__(self, input_dim, output_dim, hidden=16, lr=1e-2, seed=1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, size=(input_dim, hidden))
        self.b1 = np.zeros((1, hidden))
        self.W2 = rng.normal(0, 0.1, size=(hidden, output_dim))
        self.b2 = np.zeros((1, output_dim))
        self.lr = lr

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)  # ReLU
        Z2 = A1 @ self.W2 + self.b2
        # Sigmoid outputs (multi-label for bits)
        Yhat = 1.0/(1.0 + np.exp(-Z2))
        cache = (X, Z1, A1, Z2, Yhat)
        return Yhat, cache

    @staticmethod
    def bce_loss(Yhat, Y):
        eps = 1e-12  # stability
        Yhat = np.clip(Yhat, eps, 1 - eps)
        return -np.mean(Y*np.log(Yhat) + (1-Y)*np.log(1-Yhat))

    def backward(self, cache, Y):
        X, Z1, A1, Z2, Yhat = cache
        m = X.shape[0]
        # derivative of BCE w.r.t. Z2 (using sigmoid derivative implicitly)
        dZ2 = (Yhat - Y)/m
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)  # ReLU'
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Gradient step
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, Y, Xval=None, Yval=None, epochs=10, batch_size=256, verbose=True):
        n = X.shape[0]
        losses, val_losses = [], []
        for ep in range(epochs):
            # shuffle
            idx = np.random.permutation(n)
            Xs, Ys = X[idx], Y[idx]
            # mini-batch
            for i in range(0, n, batch_size):
                xb = Xs[i:i+batch_size]
                yb = Ys[i:i+batch_size]
                Yhat, cache = self.forward(xb)
                self.backward(cache, yb)
            # epoch loss
            Yhat_all, _ = self.forward(X)
            loss = self.bce_loss(Yhat_all, Y)
            losses.append(float(loss))
            if Xval is not None:
                Yhat_val, _ = self.forward(Xval)
                vloss = self.bce_loss(Yhat_val, Yval)
                val_losses.append(float(vloss))
            if verbose:
                if Xval is not None:
                    print(f"Epoch {ep+1}/{epochs}: loss={loss:.4f} val_loss={vloss:.4f}")
                else:
                    print(f"Epoch {ep+1}/{epochs}: loss={loss:.4f}")
        return losses, val_losses

    def predict_bits(self, X):
        Yhat, _ = self.forward(X)
        return (Yhat >= 0.5).astype(np.uint8)

# -------------- Training & Evaluation --------------
def evaluate_ber_bpsk(model, EbN0_dB_list, n_test=100_000):
    ber_mlp, ber_opt = [], []
    for eb in EbN0_dB_list:
        X, bits = gen_bpsk(n_test, eb)
        # MLP
        bhat_mlp = model.predict_bits(X).reshape(-1)
        ber_mlp.append(np.mean(bhat_mlp != bits))
        # Optimal detector (threshold at 0)
        bhat_opt = (X.reshape(-1) < 0).astype(np.uint8)
        ber_opt.append(np.mean(bhat_opt != bits))
    return np.array(ber_mlp), np.array(ber_opt)

def evaluate_ber_qpsk(model, EbN0_dB_list, n_test=50_000):
    ber_mlp, ber_opt = [], []
    for eb in EbN0_dB_list:
        X, Y = gen_qpsk(n_test, eb)
        # MLP
        Yhat_mlp = model.predict_bits(X)
        ber_mlp.append(np.mean(Yhat_mlp != Y))
        # Optimal detector (sign per axis)
        b0_opt = (X[:,0] < 0).astype(np.uint8)
        b1_opt = (X[:,1] < 0).astype(np.uint8)
        Yopt = np.stack([b0_opt, b1_opt], axis=1)
        ber_opt.append(np.mean(Yopt != Y))
    return np.array(ber_mlp), np.array(ber_opt)

def main():
    np.random.seed(RANDOM_SEED)

    # --------- BPSK: prepare data ---------
    Xtr, Ytr = gen_bpsk(N_BPSK_TRAIN, TRAIN_SNR_DB)
    Xval, Yval = gen_bpsk(N_BPSK_VAL, TRAIN_SNR_DB)
    Ytr = Ytr.reshape(-1,1).astype(np.float64)
    Yval = Yval.reshape(-1,1).astype(np.float64)

    mlp_bpsk = MLP(input_dim=1, output_dim=1, hidden=HID_BPSK, lr=LR_BPSK, seed=1)
    losses_b, vlosses_b = mlp_bpsk.fit(Xtr, Ytr, Xval, Yval, epochs=EPOCHS_BPSK, batch_size=BATCH_SIZE, verbose=True)

    # --------- QPSK: prepare data ---------
    Xtr_q, Ytr_q = gen_qpsk(N_QPSK_TRAIN, TRAIN_SNR_DB)
    Xval_q, Yval_q = gen_qpsk(N_QPSK_VAL, TRAIN_SNR_DB)
    Ytr_q = Ytr_q.astype(np.float64)
    Yval_q = Yval_q.astype(np.float64)

    mlp_qpsk = MLP(input_dim=2, output_dim=2, hidden=HID_QPSK, lr=LR_QPSK, seed=2)
    losses_q, vlosses_q = mlp_qpsk.fit(Xtr_q, Ytr_q, Xval_q, Yval_q, epochs=EPOCHS_QPSK, batch_size=BATCH_SIZE, verbose=True)

    # --------- Evaluate BER vs Eb/N0 ---------
    ber_mlp_b, ber_opt_b = evaluate_ber_bpsk(mlp_bpsk, EVAL_SNRS_DB, n_test=N_TEST_BPSK)
    ber_mlp_q, ber_opt_q = evaluate_ber_qpsk(mlp_qpsk, EVAL_SNRS_DB, n_test=N_TEST_QPSK)
    ber_theory = theory_ber_bit(EVAL_SNRS_DB)

    # Save CSV
    with open("ml_vs_opt_ber.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EbN0_dB", "BER_BPSK_MLP", "BER_BPSK_Optimal", "BER_QPSK_MLP", "BER_QPSK_Optimal", "BER_Theory_Bit"])
        for i, eb in enumerate(EVAL_SNRS_DB):
            w.writerow([float(eb), float(ber_mlp_b[i]), float(ber_opt_b[i]), float(ber_mlp_q[i]), float(ber_opt_q[i]), float(ber_theory[i])])

    # --------- Plots ---------
    # BER curves
    plt.figure(figsize=(8,6))
    plt.semilogy(EVAL_SNRS_DB, ber_theory, '-.', linewidth=1.8, label="Theory (bit)")
    plt.semilogy(EVAL_SNRS_DB, ber_opt_b, '-', linewidth=1.8, label="BPSK Optimal")
    plt.semilogy(EVAL_SNRS_DB, ber_mlp_b, 'o', markersize=4, label="BPSK MLP")
    plt.semilogy(EVAL_SNRS_DB, ber_opt_q, '--', linewidth=1.8, label="QPSK Optimal")
    plt.semilogy(EVAL_SNRS_DB, ber_mlp_q, 's', markersize=4, label="QPSK MLP")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("ML-based Detector vs Optimal & Theory (BPSK & QPSK, AWGN)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(loc="upper right")
    plt.ylim(1e-5, 1)
    plt.tight_layout()
    plt.savefig("ml_vs_opt_ber.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Training curves
    plt.figure(figsize=(8,5))
    if losses_b: plt.plot(losses_b, label="BPSK train loss")
    if vlosses_b: plt.plot(vlosses_b, label="BPSK val loss")
    if losses_q: plt.plot(losses_q, label="QPSK train loss")
    if vlosses_q: plt.plot(vlosses_q, label="QPSK val loss")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss")
    plt.title("Training / Validation Losses")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_losses.png", dpi=240, bbox_inches="tight")
    plt.show()

    # Diagnostic scatter for QPSK @ 6 dB
    Xdiag, Ydiag = gen_qpsk(6000, 6.0)
    Yhat = mlp_qpsk.predict_bits(Xdiag)
    correct = np.all(Yhat == Ydiag, axis=1)
    plt.figure(figsize=(6,6))
    plt.scatter(Xdiag[correct,0], Xdiag[correct,1], s=8, alpha=0.5, label="Correct")
    plt.scatter(Xdiag[~correct,0], Xdiag[~correct,1], s=8, alpha=0.5, label="Wrong")
    plt.title("QPSK MLP Decisions @ Eb/N0 = 6 dB")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qpsk_mlp_decisions_6dB.png", dpi=220, bbox_inches="tight")
    plt.show()

    print("Saved: ml_vs_opt_ber.csv, ml_vs_opt_ber.png, training_losses.png, qpsk_mlp_decisions_6dB.png")
    print("Done.")

if __name__ == "__main__":
    main()
