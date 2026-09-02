# Discrete-Time Systems & $z$-Domain Control

`ctrlpy` provides a high-performance, strictly typed discrete-time control systems engine centered on the complex $z$-plane, difference equation simulation, unit-circle stability classification, and continuous-to-discrete ($c2d$) conversions.

---

## 1. Mathematical Foundations

### The $z$-Transform & Discrete Transfer Functions

In digital control systems, continuous signals $e(t)$ and $u(t)$ are sampled at uniform discrete time intervals $t = k T_s$, where $T_s > 0$ is the **sampling period** (in seconds) and $f_s = 1/T_s$ is the **sampling frequency** (in Hz).

A discrete-time Linear Time-Invariant (LTI) Single-Input Single-Output (SISO) system is represented by its transfer function $H(z)$ in the forward shift operator $z = e^{s T_s}$:

$$H(z) = \frac{N(z)}{D(z)} = \frac{b_m z^m + b_{m-1} z^{m-1} + \cdots + b_1 z + b_0}{a_n z^n + a_{n-1} z^{n-1} + \cdots + a_1 z + a_0}$$

Alternatively, discrete systems are frequently formulated in terms of the backward shift (delay) operator $z^{-1}$:

$$H(z^{-1}) = \frac{b_0 + b_1 z^{-1} + \cdots + b_m z^{-m}}{a_0 + a_1 z^{-1} + \cdots + a_n z^{-n}}$$

`ctrlpy` natively supports both representations via the `DiscreteTransferFunction` (alias `dtf`) class.

```python
import ctrlpy as cp

# Define H(z) = (0.04837 z + 0.04526) / (z^2 - 1.718 z + 0.7408) with Ts = 0.1 s
H = cp.dtf([0.04837, 0.04526], [1.0, -1.718, 0.7408], dt=0.1)

# Or in backward delay operator z^-1:
H_delay = cp.dtf([0.04837, 0.04526], [1.0, -1.718, 0.7408], dt=0.1, var="z^-1")
```

---

## 2. Unit-Circle Stability Classification

In the continuous $s$-plane, asymptotic stability requires all poles to lie in the Open Left-Half Plane ($\mathrm{Re}(p_i) < 0$). Under the mapping $z = e^{s T_s}$, the imaginary axis $\mathrm{Re}(s) = 0$ is mapped to the **Unit Circle** $|z| = 1$:

$$\mathrm{Re}(s) < 0 \iff |z| = |e^{(\sigma + j\omega) T_s}| = e^{\sigma T_s} < 1 \quad (\text{since } \sigma < 0)$$

| Stability Status | Mathematical Condition | System Behavior |
| :--- | :--- | :--- |
| **Strictly Stable** | $|p_i| < 1, \quad \forall i=1,\dots,n$ | Impulse response decays exponentially to zero ($y[k] \to 0$). |
| **Marginally Stable** | $|p_i| \le 1$ and all poles on $\|p_i\|=1$ are simple (multiplicity 1). | System exhibits sustained, bounded oscillations or constant offset. |
| **Unstable** | Any $|p_i| > 1$, or multiple (repeated) poles on $\|p_i\|=1$. | Response diverges exponentially or polynomially ($k \cdot e^{j\Omega k}$). |

```python
# Check stability properties
print(f"Poles: {H.poles()}")
print(f"Zeros: {H.zeros()}")
print(f"Is Stable: {H.is_stable()}")
print(f"Is Marginally Stable: {H.is_marginally_stable()}")
print(f"Stability Status: {H.stability()}")
```

---

## 3. Continuous-to-Discrete Discretization (`c2d`)

`ctrlpy` implements 5 analytical continuous-to-discrete conversion methods via `cp.c2d(sys, dt, method=...)`:

### 1. Zero-Order Hold (ZOH)
Assumes that the continuous input $u(t)$ is held constant over each sampling interval: $u(t) = u[k]$ for $t \in [k T_s, (k+1) T_s)$:

$$H_{\text{zoh}}(z) = (1 - z^{-1}) \mathcal{Z}\left\{ \mathcal{L}^{-1}\left\{ \frac{G(s)}{s} \right\} \right\}$$

### 2. First-Order Hold (FOH)
Assumes linear extrapolation/interpolation between consecutive sample points:

$$H_{\text{foh}}(z) = \frac{(z-1)^2}{T_s z} \mathcal{Z}\left\{ \mathcal{L}^{-1}\left\{ \frac{G(s)}{s^2} \right\} \right\}$$

### 3. Tustin (Bilinear Transform)
Applies the trapezoidal numerical integration substitution:

$$s \leftarrow \frac{2}{T_s} \frac{z - 1}{z + 1}$$

Tustin maps the entire left-half $s$-plane into the interior of the unit circle, preserving minimum-phase and stability properties.

### 4. Tustin with Frequency Pre-Warping
To eliminate frequency distortion at a critical design frequency $\omega_{\text{warp}}$ (such as a crossover or notch frequency), the bilinear transform is pre-warped:

$$s \leftarrow \frac{\omega_{\text{warp}}}{\tan\left(\frac{\omega_{\text{warp}} T_s}{2}\right)} \frac{z - 1}{z + 1}$$

This ensures that $H(e^{j \omega_{\text{warp}} T_s}) = G(j \omega_{\text{warp}})$ exactly in both magnitude and phase.

### 5. Matched Pole-Zero Method
Directly maps finite poles and zeros via $z_i = e^{s_i T_s}$, adds $(r - 1)$ zeros at $z = -1$ for relative degree $r = n - m$, and matches the steady-state DC gain $H(1) = G(0)$.

```python
# Continuous plant G(s) = 10 / (s^2 + 3s + 2)
G = cp.tf([10.0], [1.0, 3.0, 2.0])
Ts = 0.1

# Discretization comparisons
H_zoh = cp.c2d(G, dt=Ts, method="zoh")
H_foh = cp.c2d(G, dt=Ts, method="foh")
H_tustin = cp.c2d(G, dt=Ts, method="tustin")
H_prewarp = cp.c2d(G, dt=Ts, method="tustin", prewarp_frequency=2.0)
H_matched = cp.c2d(G, dt=Ts, method="matched")
```

---

## 4. Difference Equation Time-Domain Simulation

Given $H(z) = \frac{\sum_{j=0}^m b_j z^{-j}}{\sum_{i=0}^n a_i z^{-i}}$ with $a_0 = 1$, the discrete difference equation is solved iteratively:

$$y[k] = \sum_{j=0}^m b_j u[k-j] - \sum_{i=1}^n a_i y[k-i]$$

`ctrlpy` provides:
- `cp.discrete_step_response(sys, T=None, n_steps=None)` or `H.step()`
- `cp.discrete_impulse_response(sys, T=None, n_steps=None)` or `H.impulse()`
- `cp.discrete_forced_response(sys, U, T=None)` or `H.forced_response(U)`

```python
# Simulate step response for 50 steps
res = H_zoh.step(n_steps=50)

print(f"Steady-State Value: {res.steady_state_value():.4f}")
print(f"Rise Time (10%-90%): {res.rise_time():.4f} s")
print(f"Settling Time (2%):  {res.settling_time():.4f} s")
print(f"Percent Overshoot:   {res.overshoot():.2f} %")
```

---

## 5. Discrete Frequency Response & Bode Analysis

In discrete systems, the frequency response is evaluated along the unit circle $z = e^{j \omega T_s}$ up to the **Nyquist folding frequency** $\omega_N = \frac{\pi}{T_s} = \pi f_s$:

$$H(e^{j \omega T_s}) = \frac{N(e^{j \omega T_s})}{D(e^{j \omega T_s})}, \quad \omega \in \left(0, \frac{\pi}{T_s}\right]$$

```python
# Compute discrete Bode data
bdata = H_zoh.bode(n_points=300)

# Evaluate at specific frequencies
w_eval, resp = H_zoh.freqresp(omega=[0.1, 1.0, 5.0, 10.0])
```

---

## 6. Complex $z$-Plane Pole-Zero Maps

Visualize poles and zeros relative to the unit circle $|z| = 1$ with static (Matplotlib) or interactive (Plotly) maps:

```python
# Static Matplotlib Pole-Zero Map
fig, ax = cp.plot_pzmap(H_zoh)

# Interactive Plotly Pole-Zero Map with tooltips (damping, natural frequency, magnitude)
fig_interactive = cp.iplot_pzmap(H_zoh)
fig_interactive.show()
```
