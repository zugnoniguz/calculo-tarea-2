from collections.abc import Callable

DerivativeFunction = Callable[[float, float], float]

def _validate_inputs(
    initial_value: float,
    initial_time: float,
    final_time: float,
    step_size: float,
) -> None:
    """Valida los parámetros comunes de los métodos numéricos."""
    if step_size <= 0:
        raise ValueError("step_size must be greater than zero.")
    if final_time < initial_time:
        raise ValueError("final_time must be greater than or equal to initial_time.")
    if initial_value != initial_value:  # Detecta NaN sin depender de librerías externas.
        raise ValueError("initial_value cannot be NaN.")


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


def runge_kutta_4(
    derivative: DerivativeFunction,
    initial_value: float,
    initial_time: float,
    final_time: float,
    step_size: float,
) -> tuple[list[float], list[float], int]:
    """
    Aproxima una ecuación diferencial mediante Runge-Kutta clásico de orden 4.

    En cada paso se calculan cuatro pendientes: al inicio, en dos puntos
    intermedios y al final del intervalo.
    """
    _validate_inputs(initial_value, initial_time, final_time, step_size)
    times = _build_time_grid(initial_time, final_time, step_size)

    values = [float(initial_value)]
    for index in range(len(times) - 1):
        current_time = times[index]
        current_value = values[-1]

        # k1: pendiente al comienzo del intervalo.
        k1 = derivative(current_time, current_value)

        # k2: pendiente en el punto medio, usando k1.
        k2 = derivative(
            current_time + step_size / 2,
            current_value + step_size * k1 / 2,
        )

        # k3: segunda estimación de la pendiente en el punto medio.
        k3 = derivative(
            current_time + step_size / 2,
            current_value + step_size * k2 / 2,
        )

        # k4: pendiente al final del intervalo.
        k4 = derivative(
            current_time + step_size,
            current_value + step_size * k3,
        )

        # Combinación ponderada de las cuatro pendientes.
        weighted_slope = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        next_value = current_value + step_size * weighted_slope
        values.append(next_value)

    return times, values, len(times) - 1
