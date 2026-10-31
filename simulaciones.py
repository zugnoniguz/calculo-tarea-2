"""Simulaciones para la tarea 2: métodos de Euler aplicados a gestión de vulnerabilidades."""

import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from euler import euler, euler_mejorado

FIG_DIR = "figuras"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.autolayout": True,
})


def run(f, x0, t0, tf, h, metodo=euler):
    ts, xs, _ = metodo(f, x0, t0, tf, h)
    return np.array(ts), np.array(xs)


def errores(ts_num, xs_num, exacta):
    ref = np.array([exacta(t) for t in ts_num])
    return np.abs(ref - xs_num)


def fmt(x, d=4):
    return f"{x:.{d}f}"


def sci(x, d=2):
    """Formato científico apto para LaTeX: 4.54e-05 -> 4,54 \times 10^{-5}."""
    if x == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(x))))
    mant = x / 10 ** exp
    return f"${mant:.{d}f} \\times 10^{{{exp}}}$"


def guardar(nombre):
    """Guarda la figura actual como SVG y como PDF (este último para LaTeX)."""
    plt.savefig(f"{FIG_DIR}/{nombre}.svg")
    plt.savefig(f"{FIG_DIR}/{nombre}.pdf")


# ---------------------------------------------------------------------------
# Caso 1: crecimiento lineal  x' = 1, x(0) = 0,  x(t) = t
# ---------------------------------------------------------------------------
f1 = lambda t, x: 1.0
exacta1 = lambda t: t

plt.figure(figsize=(7, 4.2))
tt = np.linspace(0, 10, 400)
plt.plot(tt, exacta1(tt), "k-", lw=2, label="Solución de referencia $x(t)=t$")
for h, color in [(1.0, "tab:red"), (0.5, "tab:orange"), (0.1, "tab:blue")]:
    ts, xs = run(f1, 0.0, 0.0, 10.0, h)
    plt.plot(ts, xs, "o-", ms=3.5, lw=1.2, color=color, label=f"Euler, $h={h:g}$")
plt.xlabel("Tiempo $t$")
plt.ylabel("$x(t)$")
plt.title("Caso 1: crecimiento lineal $x'(t)=1$, $x(0)=0$")
plt.legend(loc="upper left", fontsize=9)
guardar("caso1_lineal")
plt.close()

print("== Caso 1: crecimiento lineal ==")
for h in [1.0, 0.5, 0.1]:
    ts, xs = run(f1, 0.0, 0.0, 10.0, h)
    err = errores(ts, xs, exacta1)
    print(f"  h={h}: pasos={len(ts)-1}, x_final={xs[-1]:.4f}, error_max={err.max():.4e}")

# ---------------------------------------------------------------------------
# Caso 2: decrecimiento exponencial  x' = -x, x(0) = 1,  x(t) = e^{-t}
# ---------------------------------------------------------------------------
f2 = lambda t, x: -x
exacta2 = lambda t: math.exp(-t)

hs_caso2 = [1.0, 0.5, 0.1, 0.01]

plt.figure(figsize=(7, 4.2))
plt.plot(tt, np.exp(-tt), "k-", lw=2, label="Solución de referencia $x(t)=e^{-t}$")
colors2 = ["tab:red", "tab:orange", "tab:blue", "tab:green"]
for h, color in zip(hs_caso2, colors2):
    ts, xs = run(f2, 1.0, 0.0, 10.0, h)
    plt.plot(ts, xs, "o-", ms=3.5 if h >= 0.1 else 2, lw=1.2, color=color,
             label=f"Euler, $h={h:g}$")
plt.xlabel("Tiempo $t$")
plt.ylabel("$x(t)$")
plt.title("Caso 2: decrecimiento exponencial $x'(t)=-x$, $x(0)=1$")
plt.legend(loc="upper right", fontsize=9)
guardar("caso2_exponencial")
plt.close()

# ---------------------------------------------------------------------------
# Actividad 2: error numérico
# ---------------------------------------------------------------------------
hs_error = [1.0, 0.5, 0.1, 0.05, 0.01]

print("\n== Actividad 2: error (caso 2) ==")
tabla_error = []
for h in hs_error:
    ts, xs = run(f2, 1.0, 0.0, 10.0, h)
    err = errores(ts, xs, exacta2)
    tabla_error.append((h, len(ts) - 1, err[-1], err.max()))
    print(f"  h={h}: pasos={len(ts)-1}, error_final={err[-1]:.6f}, error_max={err.max():.6f}")

plt.figure(figsize=(7, 4.2))
for h, color in zip(hs_error, colors2 + ["tab:purple"]):
    ts, xs = run(f2, 1.0, 0.0, 10.0, h)
    err = errores(ts, xs, exacta2)
    plt.plot(ts, err, "o-", ms=3.5 if h >= 0.1 else 2, lw=1.2, color=color,
             label=f"$h={h:g}$")
plt.xlabel("Tiempo $t$")
plt.ylabel("Error absoluto $E_n=|x(t_n)-x_n|$")
plt.title("Error absoluto de Euler vs. tiempo (caso 2)")
plt.legend(loc="upper left", fontsize=9)
guardar("error_vs_t")
plt.close()

# ---------------------------------------------------------------------------
# Actividad 3: Euler mejorado
# ---------------------------------------------------------------------------
print("\n== Actividad 3: Euler vs Euler mejorado ==")
tabla_mejorado = []
plt.figure(figsize=(7, 4.2))
plt.plot(tt, np.exp(-tt), "k-", lw=2, label="Solución de referencia $e^{-t}$")
for h, color in zip([1.0, 0.5, 0.1], ["tab:red", "tab:orange", "tab:blue"]):
    ts_e, xs_e = run(f2, 1.0, 0.0, 10.0, h, euler)
    err_e = errores(ts_e, xs_e, exacta2)
    tabla_mejorado.append(("Euler", h, xs_e[-1], err_e[-1], len(ts_e) - 1))

    ts_m, xs_m = run(f2, 1.0, 0.0, 10.0, h, euler_mejorado)
    err_m = errores(ts_m, xs_m, exacta2)
    tabla_mejorado.append(("Euler mejorado", h, xs_m[-1], err_m[-1], len(ts_m) - 1))

    if h == 1.0:
        plt.plot(ts_e, xs_e, "s--", ms=4, lw=1.2, color=color, label=f"Euler, $h=1$")
        plt.plot(ts_m, xs_m, "^-", ms=4, lw=1.2, color=color,
                 label=f"Euler mejorado, $h=1$")
    print(f"  h={h}: Euler final={xs_e[-1]:.6f} err={err_e[-1]:.2e} | "
          f"Mejorado final={xs_m[-1]:.6f} err={err_m[-1]:.2e}")

plt.xlabel("Tiempo $t$")
plt.ylabel("$x(t)$")
plt.title("Euler vs. Euler mejorado, $h=1$, $x'(t)=-x$, $x(0)=1$")
plt.legend(loc="upper right", fontsize=9)
guardar("euler_vs_mejorado")
plt.close()

# ---------------------------------------------------------------------------
# Actividad 4: gestión de vulnerabilidades — organizaciones A, B, C
# ---------------------------------------------------------------------------
orgs = [("A", 1000, 0.1), ("B", 1000, 0.3), ("C", 1000, 0.7)]
V0 = 1000.0

print("\n== Actividad 4: organizaciones ==")
tabla_orgs = []
for nombre, v0, k in orgs:
    fk = (lambda t, v, k=k: -k * v)
    ref = (lambda t, k=k: v0 * math.exp(-k * t))
    for h in [1.0, 0.5, 0.1]:
        ts, vs = run(fk, v0, 0.0, 20.0, h)
        err = errores(ts, vs, ref)
        for t_amb in [5.0, 10.0, 15.0]:
            i = int(round(t_amb / h))
            tabla_orgs.append((nombre, k, h, t_amb, vs[i], ref(t_amb), abs(ref(t_amb) - vs[i])))
    print(f"  Org {nombre} (k={k}): h=1 -> V(5)={tabla_orgs[-7][4]:.1f} vs ref {tabla_orgs[-7][5]:.1f}")

# Figura: las tres organizaciones con h=1 y referencia
plt.figure(figsize=(7, 4.2))
tt20 = np.linspace(0, 20, 500)
for (nombre, v0, k), color in zip(orgs, ["tab:blue", "tab:orange", "tab:red"]):
    fk = (lambda t, v, k=k: -k * v)
    ref = (lambda t, k=k: v0 * np.exp(-k * t))
    ts, vs = run(fk, v0, 0.0, 20.0, 1.0)
    plt.plot(tt20, ref(tt20), "-", lw=2, color=color,
             label=f"Referencia $V(t)=1000\\,e^{{-{k}t}}$ (Org. {nombre})")
    plt.plot(ts, vs, "o", ms=4, color=color, label=f"Euler $h=1$ (Org. {nombre})")
plt.xlabel("Tiempo $t$ (días)")
plt.ylabel("Equipos vulnerables $V(t)$")
plt.title("Remediación de vulnerabilidades: organizaciones A, B y C ($h=1$)")
plt.legend(loc="upper right", fontsize=7.5)
guardar("orgs_abc")
plt.close()

# Figura: efecto del paso para la organización B
plt.figure(figsize=(7, 4.2))
k = 0.3
fk = lambda t, v: -k * v
ref = lambda t: V0 * np.exp(-k * t)
plt.plot(tt20, ref(tt20), "k-", lw=2, label="Referencia $1000\\,e^{-0.3t}$")
for h, color in zip([1.0, 0.5, 0.1], ["tab:red", "tab:orange", "tab:blue"]):
    ts, vs = run(fk, V0, 0.0, 20.0, h)
    plt.plot(ts, vs, "o-", ms=3.5 if h >= 0.5 else 2, lw=1.2, color=color,
             label=f"Euler, $h={h:g}$")
plt.xlabel("Tiempo $t$ (días)")
plt.ylabel("Equipos vulnerables $V(t)$")
plt.title("Organización B ($k=0.3$): efecto del tamaño del paso")
plt.legend(loc="upper right", fontsize=9)
guardar("org_b_paso")
plt.close()

# ---------------------------------------------------------------------------
# Modelo discreto (Actividad 5) — verificación numérica de hk > 1
# ---------------------------------------------------------------------------
print("\n== Actividad 5: modelo discreto ==")
k_disc = 0.3
plt.figure(figsize=(7, 4.2))
for hk, color in zip([0.3, 1.5, 2.5], ["tab:blue", "tab:orange", "tab:red"]):
    h = hk / k_disc
    fk = lambda t, v: -k_disc * v
    ts, vs = run(fk, V0, 0.0, 30.0, h)
    plt.plot(ts, vs, "o-", ms=4, lw=1.2, color=color, label=f"$hk={hk:g}$")
plt.axhline(0, color="k", lw=0.8)
plt.xlabel("Tiempo $t$")
plt.ylabel("Equipos vulnerables $V_n$")
plt.title("Modelo discreto $V_{n+1}=(1-hk)V_n$ con $k=0.3$")
plt.legend(loc="best", fontsize=9)
guardar("discreto_hk")
plt.close()

for hk in [0.3, 0.9, 1.0, 1.5, 2.5]:
    r = 1 - hk
    print(f"  hk={hk}: r={r:+.1f}, V1={V0*r:.1f}, V2={V0*r**2:.1f}")

# ---------------------------------------------------------------------------
# Actividad 6: modelo con aparición de nuevas vulnerabilidades  V' = λ - kV
# ---------------------------------------------------------------------------
print("\n== Actividad 6: modelo con λ ==")
k_lam = 0.3
lambdas = [0.0, 50.0, 100.0, 200.0]

plt.figure(figsize=(7, 4.2))
tt20 = np.linspace(0, 20, 500)
for lam, color in zip(lambdas, ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
    fl = (lambda t, v, lam=lam: lam - k_lam * v)
    ts, vs = run(fl, V0, 0.0, 20.0, 0.1)
    ref = (lambda t, lam=lam: lam / k_lam + (V0 - lam / k_lam) * np.exp(-k_lam * t))
    plt.plot(tt20, ref(tt20), "-", lw=1.8, color=color,
             label=f"$\\lambda={lam:g}$ (referencia $V^*={lam / k_lam:.0f}$)")
    plt.plot(ts[::5], vs[::5], "o", ms=3, color=color, alpha=0.7)
plt.xlabel("Tiempo $t$ (días)")
plt.ylabel("Equipos vulnerables $V(t)$")
plt.title(" Modelo con nuevas vulnerabilidades: $V'(t)=\\lambda - 0.3\\,V(t)$, $V(0)=1000$")
plt.legend(loc="best", fontsize=9)
guardar("lambda_modelo")
plt.close()

for lam in lambdas:
    fl = lambda t, v, lam=lam: lam - k_lam * v
    ts, vs = run(fl, V0, 0.0, 20.0, 0.1)
    ref = lambda t, lam=lam: lam / k_lam + (V0 - lam / k_lam) * math.exp(-k_lam * t)
    print(f"  λ={lam}: V(20)={vs[-1]:.1f}, ref={ref(20):.1f}, V*={lam/k_lam:.1f}")

# ---------------------------------------------------------------------------
# Tablas LaTeX
# ---------------------------------------------------------------------------
def latex_tabla_error():
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Error absoluto del método de Euler para $x'(t)=-x$, $x(0)=1$, $0\le t\le 10$.}",
        r"\label{tab:error}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"Tamaño de paso $h$ & Cantidad de pasos & Error final $E_N$ & Error máximo $E_{\max}$ \\",
        r"\midrule",
    ]
    for h, pasos, ef, em in tabla_error:
        h_s = f"{h:g}".replace(".", ",")
        lines.append(f"${h_s}$ & {pasos} & {sci(ef)} & {sci(em)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_tabla_mejorado():
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Comparación entre Euler y Euler mejorado para $x'(t)=-x$, $x(0)=1$, $0\le t\le 10$.}",
        r"\label{tab:mejorado}",
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"Método & $h$ & Valor final & Error final & Pasos \\",
        r"\midrule",
    ]
    for metodo, h, vf, ef, pasos in tabla_mejorado:
        h_s = f"{h:g}".replace(".", ",")
        lines.append(f"{metodo} & ${h_s}$ & ${vf:.6f}$ & {sci(ef)} & {pasos} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_tabla_orgs():
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Equipos vulnerables según Euler ($h=1$) vs.\\ solución de referencia $V(t)=1000\,e^{-kt}$.}",
        r"\label{tab:orgs}",
        r"\begin{tabular}{@{}lllrrr@{}}",
        r"\toprule",
        r"Organización & $k$ (día$^{-1}$) & $t$ (días) & Euler $V_n$ & Referencia $V(t)$ & Error $E_n$ \\",
        r"\midrule",
    ]
    for nombre, k, h, t_, vn, vr, er in tabla_orgs:
        if h != 1.0:
            continue
        k_s = f"{k:g}".replace(".", ",")
        t_s = f"{t_:g}".replace(".", ",")
        lines.append(f"{nombre} & ${k_s}$ & ${t_s}$ & ${vn:.1f}$ & ${vr:.1f}$ & ${er:.1f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def latex_tabla_orgs_h():
    """Error máximo de cada organización según h."""
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Error máximo de Euler en $0\le t\le 20$ según el tamaño de paso, para cada organización.}",
        r"\label{tab:orgs-h}",
        r"\begin{tabular}{@{}lrrr@{}}",
        r"\toprule",
        r"$h$ & Organización A ($k=0{,}1$) & Organización B ($k=0{,}3$) & Organización C ($k=0{,}7$) \\",
        r"\midrule",
    ]
    ermax = {1.0: [], 0.5: [], 0.1: []}
    for nombre, v0, k in orgs:
        fk = (lambda t, v, k=k: -k * v)
        ref = (lambda t, k=k: v0 * math.exp(-k * t))
        for h in [1.0, 0.5, 0.1]:
            ts, vs = run(fk, v0, 0.0, 20.0, h)
            ermax[h].append(errores(ts, vs, ref).max())
    for h in [1.0, 0.5, 0.1]:
        h_s = f"{h:g}".replace(".", ",")
        cells = " & ".join(f"${e:.1f}$" for e in ermax[h])
        lines.append(f"${h_s}$ & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


os.makedirs("tablas", exist_ok=True)
with open("tablas/error.tex", "w") as fh:
    fh.write(latex_tabla_error())
with open("tablas/mejorado.tex", "w") as fh:
    fh.write(latex_tabla_mejorado())
with open("tablas/orgs.tex", "w") as fh:
    fh.write(latex_tabla_orgs())
with open("tablas/orgs_h.tex", "w") as fh:
    fh.write(latex_tabla_orgs_h())

print("\nFiguras y tablas generadas.")
