from collections.abc import Callable

DerivativeFunction = Callable[[float, float], float]

def _build_time_grid(t0: float, tf: float, h: float) -> list[float]:
    """
    Construye la secuencia tn para el intervalo provisto.

    Se exige que el intervalo sea divisible por el tamaño del paso para que
    la simulación termine exactamente en final_time, sin agregar un paso parcial.
    """
    interval_count_float = (tf - t0) / h
    interval_count = round(interval_count_float)

    if abs(interval_count_float - interval_count) > 1e-9:
        raise ValueError(
            """The time interval must be divisible by step_size.
            Choose a step_size that reaches final_time exactly."""
        )

    return [
        t0 + index * h
        for index in range(interval_count + 1)
    ]


def euler(
    f: DerivativeFunction,
    x0: float,
    t0: float,
    tf: float,
    h: float,
) -> tuple[list[float], list[float], int]:
    """
    Aproxima una ecuación x'(t) = f(t, x) mediante el método de Euler.

    Returns:
        times: instantes simulados.
        values: aproximaciones de la variable en esos instantes.
        step_count: cantidad de iteraciones realizadas.
    """
    times = _build_time_grid(t0, tf, h)

    values = [float(x0)]
    for index in range(len(times) - 1):
        tn = times[index]
        xn = values[-1]

        xn1 = xn + h * f(tn, xn)
        values.append(xn1)

    return times, values, len(times) - 1


def euler_improved(
    f: DerivativeFunction,
    x0: float,
    t0: float,
    tf: float,
    h: float,
) -> tuple[list[float], list[float], int]:
    """
    Aproxima una ecuación diferencial x'(t) = f(t, x) con Euler mejorado.

    Primero predice el siguiente valor con Euler y luego promedia la pendiente
    inicial con la pendiente evaluada en el punto predicho.
    """
    times = _build_time_grid(t0, tf, h)

    values = [x0]
    for index in range(len(times) - 1):
        tn = times[index]
        xn = values[-1]

        initial_slope = f(tn, xn)
        xnbar = xn + h * initial_slope
        predicted_slope = f(tn + h, xnbar)

        average_slope = (initial_slope + predicted_slope) / 2
        xn1 = xn + h * average_slope
        values.append(xn1)

    return times, values, len(times) - 1
