from typing import Callable


def euler(f: Callable[[float, float], float], x0: float, t0: float, tf: float, h: float):
    xn = x0
    tn = t0
    ts: list[float] = []
    xs: list[float] = []
    while tn <= tf:
        ts.append(tn)
        xs.append(xn)
        xn = xn + h * f(tn, xn)
        tn = tn + h

    return ts, xs


def euler_mejorado(f: Callable[[float, float], float], x0: float, t0: float, tf: float, h: float):
    xn = x0
    tn = t0
    ts: list[float] = []
    xs: list[float] = []
    while tn <= tf:
        ts.append(tn)
        xs.append(xn)
        xbarn1 = xn + h * f(tn, xn)
        v = f(tn, xn) + f(tn + h, xbarn1)
        xn = xn + (h / 2) * v
        tn = tn + h

    return ts, xs
