"""
Métodos numéricos para aproximar soluciones de ecuaciones diferenciales.

Este módulo contiene las implementaciones reutilizables de:
- Método de Euler.
- Método de Euler mejorado (Heun).
"""

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


def _build_time_grid(initial_time: float, final_time: float, step_size: float) -> list[float]:
    """
    Construye una grilla temporal estable.

    Se exige que el intervalo sea divisible por el tamaño del paso para que
    la simulación termine exactamente en final_time, sin agregar un paso parcial.
    """
    interval_count_float = (final_time - initial_time) / step_size
    interval_count = round(interval_count_float)

    if abs(interval_count_float - interval_count) > 1e-9:
        raise ValueError(
            "The time interval must be divisible by step_size. "
            "Choose a step_size that reaches final_time exactly."
        )

    return [
        initial_time + index * step_size
        for index in range(interval_count + 1)
    ]


def euler(
    derivative: DerivativeFunction,
    initial_value: float,
    initial_time: float,
    final_time: float,
    step_size: float,
) -> tuple[list[float], list[float], int]:
    """
    Aproxima una ecuación x'(t) = f(t, x) mediante el método de Euler.

    Returns:
        times: instantes simulados.
        values: aproximaciones de la variable en esos instantes.
        step_count: cantidad de iteraciones realizadas.
    """
    _validate_inputs(initial_value, initial_time, final_time, step_size)
    times = _build_time_grid(initial_time, final_time, step_size)

    values = [float(initial_value)]
    for index in range(len(times) - 1):
        current_time = times[index]
        current_value = values[-1]

        # Actividad 1: se usa la pendiente al comienzo del intervalo.
        slope = derivative(current_time, current_value)
        next_value = current_value + step_size * slope
        values.append(next_value)

    return times, values, len(times) - 1


def euler_improved(
    derivative: DerivativeFunction,
    initial_value: float,
    initial_time: float,
    final_time: float,
    step_size: float,
) -> tuple[list[float], list[float], int]:
    """
    Aproxima una ecuación diferencial con Euler mejorado (método de Heun).

    Primero predice el siguiente valor con Euler y luego promedia la pendiente
    inicial con la pendiente evaluada en el punto predicho.
    """
    _validate_inputs(initial_value, initial_time, final_time, step_size)
    times = _build_time_grid(initial_time, final_time, step_size)

    values = [float(initial_value)]
    for index in range(len(times) - 1):
        current_time = times[index]
        current_value = values[-1]

        # Actividad 3: primera pendiente, calculada al inicio del intervalo.
        initial_slope = derivative(current_time, current_value)

        # Actividad 3: predicción provisoria del valor al final del intervalo.
        predicted_value = current_value + step_size * initial_slope

        # Actividad 3: segunda pendiente, calculada en el punto predicho.
        predicted_slope = derivative(
            current_time + step_size,
            predicted_value,
        )

        # Actividad 3: corrección usando el promedio de ambas pendientes.
        average_slope = (initial_slope + predicted_slope) / 2
        next_value = current_value + step_size * average_slope
        values.append(next_value)

    return times, values, len(times) - 1
