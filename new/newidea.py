"""
ML-based symbol decision for BPSK & QPSK over AWGN + Rayleigh fading
με uncertainty-aware ML δέκτη (PhD-style extension).

Τι κάνει ο κώδικας:

1) AWGN μέρος (κλασικό):
   - BPSK & QPSK σε AWGN
   - Εκπαίδευση μικρού MLP (NumPy-only)
   - Σύγκριση BER με κλασικό optimal detector & θεωρία (Q-function)

2) Fading + Uncertainty μέρος (ερευνητικό):
   - QPSK σε Rayleigh flat fading: y = h * s + n, h ~ CN(0,1)
   - Imperfect CSI: h_hat = h + θόρυβος "pilot"
   - Classical detector με equalization y / h_hat
   - Oracle detector με perfect h (κάτω φράγμα)
   - Uncertainty-aware MLP:
       είσοδος: [Re(y), Im(y), Re(h_hat), Im(h_hat)]
       έξοδος:
         * bits (2 outputs: b0, b1)
         * confidence (1 output: "πιθανότητα ότι και τα 2 bits είναι σωστά")

   - BER curves: MLP vs classical (imperfect CSI) vs oracle
   - Confidence vs πραγματικό error (calibration)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# ---------------- Γενικές Παράμετροι ----------------
TRAIN_SNR_DB = 6.0
EVAL_SNRS_DB = np.arange(0, 13, 1)

# Μεγέθη datasets (μπορείς να τα μεγαλώσεις για πιο "σοβαρά" αποτελέσματα)
N_BPSK_TRAIN, N_BPSK_VAL = 8_000, 2_000
N_QPSK_TRAIN, N_QPSK_VAL = 6_000, 2_000
N_TEST_BPSK, N_TEST_QPSK = 5_000, 3_000

EPOCHS_BPSK, EPOCHS_QPSK = 4, 5
BATCH_SIZE = 512
LR_BPSK, LR_QPSK = 5e-3, 7e-3
HID_BPSK, HID_QPSK = 16, 24
RANDOM_SEED = 7

# --------- Παράμετροι fading + uncertainty-aware detector ---------
PILOT_SNR_DB = 20.0          # SNR "πιλότων" για εκτίμηση καναλιού
N_QPSK_TRAIN_FAD = 8_000
N_QPSK_VAL_FAD = 2_000
EPOCHS_QPSK_FAD = 7
LR_QPSK_FAD = 7e-3
HID_QPSK_FAD = 32
LAMBDA_CONF = 0.7            # βάρος στο loss της confidence head

# -------------- Βοηθητικές Συναρτήσεις --------------
def db2lin(db):
    return 10.0**(db/10.0)

def awgn_sigma(EbN0_dB):
    EbN0_lin = db2lin(EbN0_dB)
    N0 = 1.0 / EbN0_lin  # Eb=1
    return np.sqrt(N0/2.0)  # per real dimension

def Qfunc(x):
    # Q-function μέσω erf (σταθερό αριθμητικά)
    import numpy as _np, math as _math
    x = _np.asarray(x, dtype=float)
    vec = _np.vectorize(lambda t: 0.5*(1.0 - _math.erf(t/_np.sqrt(2.0))))
    return vec(x)

def theory_ber_bit(EbN0_dB):
    EbN0 = db2lin(EbN0_dB)
    return Qfunc(np.sqrt(2*EbN0))

# ---------------- AWGN Generators ----------------
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
    Y = np.stack([b0, b1], axis=1)
    return X, Y

# ---------------- Fading Generator (νέο) ----------------
def gen_qpsk_fading(n_symbols, EbN0_dB, pilot_snr_db=PILOT_SNR_DB):
    """
    QPSK σε flat Rayleigh fading:
        y = h * s + n,  h ~ CN(0,1)

    Επιστρέφει:
      X      : [Re(y), Im(y), Re(h_hat), Im(h_hat)]  (είσοδος για MLP)
      Y      : [b0, b1] bits
      h      : πραγματικό complex κανάλι
      y      : λαμβανόμενα σύμβολα (complex)
      h_hat  : ατελής εκτίμηση καναλιού (complex)
    """
    bits = np.random.randint(0, 2, size=2*n_symbols, dtype=np.uint8)
    b0 = bits[0::2]
    b1 = bits[1::2]

    # QPSK σύμβολα με κανονικοποίηση ενέργειας
    I = 1 - 2*b0
    Q = 1 - 2*b1
    s = (I + 1j*Q) / np.sqrt(2.0)

    # Rayleigh flat fading: h ~ CN(0,1)
    h = (np.random.randn(n_symbols) + 1j*np.random.randn(n_symbols)) / np.sqrt(2.0)

    # AWGN στο κανάλι
    sigma = awgn_sigma(EbN0_dB)
    n = sigma * (np.random.randn(n_symbols) + 1j*np.random.randn(n_symbols))
    y = h * s + n

    # Εκτίμηση καναλιού μέσω "πιλότων" (imperfect CSI)
    sigma_p = awgn_sigma(pilot_snr_db)
    n_p = sigma_p * (np.random.randn(n_symbols) + 1j*np.random.randn(n_symbols))
    h_hat = h + n_p

    X = np.stack([np.real(y), np.imag(y),
                  np.real(h_hat), np.imag(h_hat)], axis=1)
    Y = np.stack([b0, b1], axis=1)

    return X.astype(np.float64), Y.astype(np.float64), h, y, h_hat

# -------------- Κλασικό MLP (για AWGN) --------------
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
        Yhat = 1.0/(1.0 + np.exp(-Z2))  # Sigmoid
        cache = (X, Z1, A1, Z2, Yhat)
        return Yhat, cache

    @staticmethod
    def bce_loss(Yhat, Y):
        eps = 1e-12
        Yhat = np.clip(Yhat, eps, 1-eps)
        return -np.mean(Y*np.log(Yhat) + (1-Y)*np.log(1-Yhat))

    def backward(self, cache, Y):
        X, Z1, A1, Z2, Yhat = cache
        m = X.shape[0]
        dZ2 = (Yhat - Y)/m
        dW2 = A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, Y, Xval=None, Yval=None, epochs=10, batch_size=256, verbose=True):
        n = X.shape[0]
        losses, val_losses = [], []
        for ep in range(epochs):
            idx = np.random.permutation(n)
            Xs, Ys = X[idx], Y[idx]
            for i in range(0, n, batch_size):
                xb = Xs[i:i+batch_size]
                yb = Ys[i:i+batch_size]
                Yhat, cache = self.forward(xb)
                self.backward(cache, yb)
            Yhat_all, _ = self.forward(X)
            loss = self.bce_loss(Yhat_all, Y)
            losses.append(float(loss))
            if Xval is not None:
                Yhat_val, _ = self.forward(Xval)
                vloss = self.bce_loss(Yhat_val, Yval)
                val_losses.append(float(vloss))
            if verbose:
                if Xval is not None:
                    print(f"[AWGN] Epoch {ep+1}/{epochs}: loss={loss:.4f} val_loss={vloss:.4f}")
                else:
                    print(f"[AWGN] Epoch {ep+1}/{epochs}: loss={loss:.4f}")
        return losses, val_losses

    def predict_bits(self, X):
        Yhat, _ = self.forward(X)
        return (Yhat >= 0.5).astype(np.uint8)

# -------------- Uncertainty-aware MLP (για fading) --------------
class MLPUncertainty:
    """
    MLP με 2 "κεφάλια":
      - Head 1: bits (2 outputs για QPSK)
      - Head 2: confidence (1 output: πόσο σίγουρο είναι ότι τα 2 bits είναι σωστά)

    Εκπαιδεύεται με:
      L_total = L_bits + λ * L_conf
    όπου:
      L_bits = BCE στα bits
      L_conf = BCE μεταξύ confidence και target "correctness indicator"
    """
    def __init__(self, input_dim, hidden=32, lr=1e-3, seed=1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, size=(input_dim, hidden))
        self.b1 = np.zeros((1, hidden))

        self.W_bits = rng.normal(0, 0.1, size=(hidden, 2))
        self.b_bits = np.zeros((1, 2))

        self.W_conf = rng.normal(0, 0.1, size=(hidden, 1))
        self.b_conf = np.zeros((1, 1))

        self.lr = lr

    @staticmethod
    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    @staticmethod
    def bce_loss(yhat, y):
        eps = 1e-12
        yhat = np.clip(yhat, eps, 1-eps)
        return -np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat))

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.maximum(0, Z1)

        Z_bits = A1 @ self.W_bits + self.b_bits
        Y_bits = self.sigmoid(Z_bits)

        Z_conf = A1 @ self.W_conf + self.b_conf
        C = self.sigmoid(Z_conf)

        cache = (X, Z1, A1, Z_bits, Y_bits, Z_conf, C)
        return Y_bits, C, cache

    def backward(self, cache, Y_true, C_target, lam_conf):
        X, Z1, A1, Z_bits, Y_bits, Z_conf, C = cache
        m = X.shape[0]

        # dL/dZ_bits για BCE + sigmoid
        dZ_bits = (Y_bits - Y_true) / m

        # dL/dZ_conf για BCE + sigmoid, με weight λ
        dZ_conf = lam_conf * (C - C_target) / m

        # Gradients heads
        dW_bits = A1.T @ dZ_bits
        db_bits = np.sum(dZ_bits, axis=0, keepdims=True)

        dW_conf = A1.T @ dZ_conf
        db_conf = np.sum(dZ_conf, axis=0, keepdims=True)

        # Backprop στο πρώτο layer (άθροισμα από τα 2 heads)
        dA1 = dZ_bits @ self.W_bits.T + dZ_conf @ self.W_conf.T
        dZ1 = dA1 * (Z1 > 0)

        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Gradient step
        self.W_bits -= self.lr * dW_bits
        self.b_bits -= self.lr * db_bits
        self.W_conf -= self.lr * dW_conf
        self.b_conf -= self.lr * db_conf
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict_bits_and_conf(self, X):
        Y_bits, C, _ = self.forward(X)
        bits_hat = (Y_bits >= 0.5).astype(np.uint8)
        return bits_hat, C

    def fit(self, X, Y, Xval=None, Yval=None,
            epochs=10, batch_size=256, lam_conf=0.5, verbose=True):
        """
        Εκπαίδευση με multi-task loss: bits + confidence
        """
        n = X.shape[0]
        losses, val_losses = [], []
        for ep in range(epochs):
            idx = np.random.permutation(n)
            Xs, Ys = X[idx], Y[idx]
            for i in range(0, n, batch_size):
                xb = Xs[i:i+batch_size]
                yb = Ys[i:i+batch_size]  # [batch, 2]

                Yhat_bits, C, cache = self.forward(xb)

                # Loss στα bits
                L_bits = self.bce_loss(Yhat_bits, yb)

                # Target για confidence: 1 αν και τα 2 bits σωστά, αλλιώς 0
                pred_bits = (Yhat_bits >= 0.5).astype(np.uint8)
                correct_mask = (pred_bits == yb).all(axis=1).astype(np.float64)
                C_target = correct_mask.reshape(-1,1)

                L_conf = self.bce_loss(C, C_target)
                L = L_bits + lam_conf * L_conf

                self.backward(cache, yb, C_target, lam_conf)

            # epoch losses (στο train set)
            Yhat_bits_all, C_all, _ = self.forward(X)
            L_bits_all = self.bce_loss(Yhat_bits_all, Y)
            pred_bits_all = (Yhat_bits_all >= 0.5).astype(np.uint8)
            correct_mask_all = (pred_bits_all == Y).all(axis=1).astype(np.float64)
            C_target_all = correct_mask_all.reshape(-1,1)
            L_conf_all = self.bce_loss(C_all, C_target_all)
            L_total = L_bits_all + lam_conf * L_conf_all
            losses.append(float(L_total))

            if Xval is not None:
                Yhat_bits_val, C_val, _ = self.forward(Xval)
                L_bits_val = self.bce_loss(Yhat_bits_val, Yval)
                pred_bits_val = (Yhat_bits_val >= 0.5).astype(np.uint8)
                correct_mask_val = (pred_bits_val == Yval).all(axis=1).astype(np.float64)
                C_target_val = correct_mask_val.reshape(-1,1)
                L_conf_val = self.bce_loss(C_val, C_target_val)
                L_total_val = L_bits_val + lam_conf * L_conf_val
                val_losses.append(float(L_total_val))

            if verbose:
                if Xval is not None:
                    print(f"[FADING+UNCERT] Epoch {ep+1}/{epochs}: "
                          f"train_loss={L_total:.4f} val_loss={L_total_val:.4f}")
                else:
                    print(f"[FADING+UNCERT] Epoch {ep+1}/{epochs}: train_loss={L_total:.4f}")
        return losses, val_losses

# -------------- Αξιολόγηση BER (AWGN) --------------
def evaluate_ber_bpsk(model, EbN0_dB_list, n_test=100_000):
    ber_mlp, ber_opt = [], []
    for eb in EbN0_dB_list:
        X, bits = gen_bpsk(n_test, eb)
        bhat_mlp = model.predict_bits(X).reshape(-1)
        ber_mlp.append(np.mean(bhat_mlp != bits))
        bhat_opt = (X.reshape(-1) < 0).astype(np.uint8)
        ber_opt.append(np.mean(bhat_opt != bits))
    return np.array(ber_mlp), np.array(ber_opt)

def evaluate_ber_qpsk(model, EbN0_dB_list, n_test=50_000):
    ber_mlp, ber_opt = [], []
    for eb in EbN0_dB_list:
        X, Y = gen_qpsk(n_test, eb)
        Yhat_mlp = model.predict_bits(X)
        ber_mlp.append(np.mean(Yhat_mlp != Y))
        b0_opt = (X[:,0] < 0).astype(np.uint8)
        b1_opt = (X[:,1] < 0).astype(np.uint8)
        Yopt = np.stack([b0_opt, b1_opt], axis=1)
        ber_opt.append(np.mean(Yopt != Y))
    return np.array(ber_mlp), np.array(ber_opt)

# -------------- Αξιολόγηση σε Fading + Uncertainty --------------
def evaluate_ber_qpsk_fading_uncert(model, EbN0_dB_list, n_test=50_000, pilot_snr_db=PILOT_SNR_DB):
    """
    Υπολογίζει:
      - BER του MLP uncertainty-aware δέκτη
      - BER κλασικού δέκτη με h_hat (imperfect CSI)
      - BER "oracle" δέκτη με perfect h
      - πίνακες με confidence & error για calibration analysis
    """
    ber_mlp, ber_opt_imperfect, ber_opt_oracle = [], [], []
    all_confidences = []
    all_errors = []

    for eb in EbN0_dB_list:
        X, Y, h, y, h_hat = gen_qpsk_fading(n_test, eb, pilot_snr_db=pilot_snr_db)

        # MLP-based detector (bits + confidence)
        bits_hat, C = model.predict_bits_and_conf(X)
        err_mask = (bits_hat != Y).any(axis=1).astype(np.uint8)

        ber_mlp.append(np.mean(err_mask))
        all_confidences.append(C.reshape(-1))
        all_errors.append(err_mask.reshape(-1))

        # Classical detector με h_hat (equalization)
        r_ih = y / (h_hat + 1e-8)
        b0_ih = (np.real(r_ih) < 0).astype(np.uint8)
        b1_ih = (np.imag(r_ih) < 0).astype(np.uint8)
        Y_ih = np.stack([b0_ih, b1_ih], axis=1)
        err_ih = (Y_ih != Y).any(axis=1).astype(np.uint8)
        ber_opt_imperfect.append(np.mean(err_ih))

        # Oracle detector με perfect h
        r_or = y / (h + 1e-8)
        b0_or = (np.real(r_or) < 0).astype(np.uint8)
        b1_or = (np.imag(r_or) < 0).astype(np.uint8)
        Y_or = np.stack([b0_or, b1_or], axis=1)
        err_or = (Y_or != Y).any(axis=1).astype(np.uint8)
        ber_opt_oracle.append(np.mean(err_or))

    all_confidences = np.concatenate(all_confidences)
    all_errors = np.concatenate(all_errors)

    return (np.array(ber_mlp),
            np.array(ber_opt_imperfect),
            np.array(ber_opt_oracle),
            all_confidences,
            all_errors)

# -------------- Calibration plot (confidence vs actual error) --------------
def plot_confidence_calibration(confidences, errors, bins=10):
    """
    Χωρίζει τα δείγματα σε "κουβάδες" ως προς το confidence
    και υπολογίζει πραγματικό error rate σε κάθε interval.
    """
    conf = confidences
    err = errors.astype(np.float64)

    edges = np.linspace(0, 1, bins+1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    avg_err = []
    avg_conf = []

    for i in range(bins):
        mask = (conf >= edges[i]) & (conf < edges[i+1])
        if np.sum(mask) == 0:
            avg_err.append(np.nan)
            avg_conf.append(np.nan)
        else:
            avg_err.append(np.mean(err[mask]))
            avg_conf.append(np.mean(conf[mask]))

    avg_err = np.array(avg_err)
    avg_conf = np.array(avg_conf)

    plt.figure(figsize=(6,5))
    plt.plot(avg_conf, avg_err, 'o-', label="MLP detector")
    plt.xlabel("Μέσο Confidence")
    plt.ylabel("Πραγματικό Error Rate")
    plt.title("Calibration: Confidence vs Actual Error")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("confidence_calibration.png", dpi=240, bbox_inches="tight")
    plt.show()

# -------------- Main --------------
def main():
    np.random.seed(RANDOM_SEED)

    # --------- AWGN: BPSK ---------
    Xtr, Ytr = gen_bpsk(N_BPSK_TRAIN, TRAIN_SNR_DB)
    Xval, Yval = gen_bpsk(N_BPSK_VAL, TRAIN_SNR_DB)
    Ytr = Ytr.reshape(-1,1).astype(np.float64)
    Yval = Yval.reshape(-1,1).astype(np.float64)

    mlp_bpsk = MLP(input_dim=1, output_dim=1, hidden=HID_BPSK, lr=LR_BPSK, seed=1)
    losses_b, vlosses_b = mlp_bpsk.fit(Xtr, Ytr, Xval, Yval,
                                       epochs=EPOCHS_BPSK, batch_size=BATCH_SIZE, verbose=True)

    # --------- AWGN: QPSK ---------
    Xtr_q, Ytr_q = gen_qpsk(N_QPSK_TRAIN, TRAIN_SNR_DB)
    Xval_q, Yval_q = gen_qpsk(N_QPSK_VAL, TRAIN_SNR_DB)
    Ytr_q = Ytr_q.astype(np.float64)
    Yval_q = Yval_q.astype(np.float64)

    mlp_qpsk = MLP(input_dim=2, output_dim=2, hidden=HID_QPSK, lr=LR_QPSK, seed=2)
    losses_q, vlosses_q = mlp_qpsk.fit(Xtr_q, Ytr_q, Xval_q, Yval_q,
                                       epochs=EPOCHS_QPSK, batch_size=BATCH_SIZE, verbose=True)

    # --------- Fading: QPSK + Uncertainty-aware detector ---------
    Xtr_f, Ytr_f, _, _, _ = gen_qpsk_fading(N_QPSK_TRAIN_FAD, TRAIN_SNR_DB, pilot_snr_db=PILOT_SNR_DB)
    Xval_f, Yval_f, _, _, _ = gen_qpsk_fading(N_QPSK_VAL_FAD, TRAIN_SNR_DB, pilot_snr_db=PILOT_SNR_DB)

    mlp_unc = MLPUncertainty(input_dim=4, hidden=HID_QPSK_FAD,
                             lr=LR_QPSK_FAD, seed=3)

    losses_f, vlosses_f = mlp_unc.fit(Xtr_f, Ytr_f, Xval_f, Yval_f,
                                      epochs=EPOCHS_QPSK_FAD,
                                      batch_size=BATCH_SIZE,
                                      lam_conf=LAMBDA_CONF,
                                      verbose=True)

    # --------- AWGN BER ---------
    ber_mlp_b, ber_opt_b = evaluate_ber_bpsk(mlp_bpsk, EVAL_SNRS_DB, n_test=N_TEST_BPSK)
    ber_mlp_q, ber_opt_q = evaluate_ber_qpsk(mlp_qpsk, EVAL_SNRS_DB, n_test=N_TEST_QPSK)
    ber_theory = theory_ber_bit(EVAL_SNRS_DB)

    # --------- Fading BER + Uncertainty ---------
    (ber_mlp_fad,
     ber_opt_imperf,
     ber_opt_oracle,
     all_conf,
     all_err) = evaluate_ber_qpsk_fading_uncert(mlp_unc,
                                                EVAL_SNRS_DB,
                                                n_test=N_TEST_QPSK,
                                                pilot_snr_db=PILOT_SNR_DB)

    # --------- CSV με όλα τα BER ---------
    with open("ml_vs_opt_ber_full.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "EbN0_dB",
            "BER_BPSK_MLP", "BER_BPSK_Optimal",
            "BER_QPSK_MLP_AWGN", "BER_QPSK_Optimal_AWGN", "BER_Theory_Bit",
            "BER_QPSK_MLP_Fading", "BER_QPSK_Opt_ImperfectCSI", "BER_QPSK_Opt_Oracle"
        ])
        for i, eb in enumerate(EVAL_SNRS_DB):
            w.writerow([
                float(eb),
                float(ber_mlp_b[i]), float(ber_opt_b[i]),
                float(ber_mlp_q[i]), float(ber_opt_q[i]),
                float(ber_theory[i]),
                float(ber_mlp_fad[i]),
                float(ber_opt_imperf[i]),
                float(ber_opt_oracle[i])
            ])

    # --------- Plots AWGN ---------
    plt.figure(figsize=(8,6))
    plt.semilogy(EVAL_SNRS_DB, ber_theory, '-.', linewidth=1.8, label="Theory (bit)")
    plt.semilogy(EVAL_SNRS_DB, ber_opt_b, '-', linewidth=1.8, label="BPSK Optimal")
    plt.semilogy(EVAL_SNRS_DB, ber_mlp_b, 'o', markersize=4, label="BPSK MLP")
    plt.semilogy(EVAL_SNRS_DB, ber_opt_q, '--', linewidth=1.8, label="QPSK Optimal")
    plt.semilogy(EVAL_SNRS_DB, ber_mlp_q, 's', markersize=4, label="QPSK MLP")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("AWGN: ML-based Detector vs Optimal & Theory (BPSK & QPSK)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(loc="upper right")
    plt.ylim(1e-5, 1)
    plt.tight_layout()
    plt.savefig("ml_vs_opt_ber_awgn.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------- Plots Fading ---------
    plt.figure(figsize=(8,6))
    plt.semilogy(EVAL_SNRS_DB, ber_opt_oracle, '-.', linewidth=1.8, label="Oracle (perfect h)")
    plt.semilogy(EVAL_SNRS_DB, ber_opt_imperf, '--', linewidth=1.8, label="Classical (imperfect ĥ)")
    plt.semilogy(EVAL_SNRS_DB, ber_mlp_fad, 'o-', markersize=4, label="MLP Uncertainty Detector")
    plt.xlabel("Eb/N0 (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title(f"QPSK over Rayleigh Fading (Pilot SNR = {PILOT_SNR_DB} dB)")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(loc="upper right")
    plt.ylim(1e-4, 1)
    plt.tight_layout()
    plt.savefig("ml_vs_opt_ber_fading_uncert.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --------- Training Losses ---------
    plt.figure(figsize=(8,5))
    if losses_b: plt.plot(losses_b, label="BPSK train loss")
    if vlosses_b: plt.plot(vlosses_b, label="BPSK val loss")
    if losses_q: plt.plot(losses_q, label="QPSK AWGN train loss")
    if vlosses_q: plt.plot(vlosses_q, label="QPSK AWGN val loss")
    if losses_f: plt.plot(losses_f, label="QPSK Fading+Uncert train loss")
    if vlosses_f: plt.plot(vlosses_f, label="QPSK Fading+Uncert val loss")
    plt.xlabel("Epoch"); plt.ylabel("Total Loss")
    plt.title("Training / Validation Losses (AWGN + Fading+Uncertainty)")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_losses_all.png", dpi=240, bbox_inches="tight")
    plt.show()

    # --------- Calibration: Confidence vs Actual Error ---------
    plot_confidence_calibration(all_conf, all_err)

    print("Saved: ml_vs_opt_ber_full.csv, ml_vs_opt_ber_awgn.png, "
          "ml_vs_opt_ber_fading_uncert.png, training_losses_all.png, "
          "confidence_calibration.png")
    print("Done.")

if __name__ == "__main__":
    main()
