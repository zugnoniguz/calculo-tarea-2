"""
Definición de los modelos matemáticos utilizados en el proyecto.

Los modelos corresponden a los casos de verificación y a la aplicación
de gestión de vulnerabilidades descritos en la propuesta.
"""

import math


# ---------------------------------------------------------------------------
# Sección 1: casos de verificación del método (propuesta, sección 5).
# ---------------------------------------------------------------------------

def linear_growth_derivative(time: float, value: float) -> float:
    """Caso 1: x'(t) = 1."""
    return 1.0


def linear_growth_reference(time: float) -> float:
    """Solución de referencia del caso 1: x(t) = t."""
    return time


def exponential_decay_derivative(time: float, value: float) -> float:
    """Caso 2: x'(t) = -x(t)."""
    return -value


def exponential_decay_reference(time: float) -> float:
    """Solución de referencia del caso 2: x(t) = e^(-t)."""
    return math.exp(-time)


# ---------------------------------------------------------------------------
# Sección 2: gestión de vulnerabilidades (propuesta, secciones 8 y 10).
# ---------------------------------------------------------------------------

def vulnerability_remediation_derivative(
    time: float,
    vulnerable_devices: float,
    remediation_rate: float,
) -> float:
    """
    Actividad 4: V'(t) = -k V(t).

    remediation_rate (k) representa la rapidez del proceso de remediación.
    """
    return -remediation_rate * vulnerable_devices


def vulnerability_remediation_reference(
    time: float,
    initial_vulnerable_devices: float,
    remediation_rate: float,
) -> float:
    """Referencia provista en la propuesta: V(t) = V0 * e^(-k*t)."""
    return initial_vulnerable_devices * math.exp(-remediation_rate * time)


def vulnerability_with_new_devices_derivative(
    time: float,
    vulnerable_devices: float,
    remediation_rate: float,
    new_vulnerability_rate: float,
) -> float:
    """
    Actividad 6: V'(t) = lambda - k V(t).

    new_vulnerability_rate (lambda) representa la aparición simplificada
    de nuevos equipos vulnerables por unidad de tiempo.
    """
    return new_vulnerability_rate - remediation_rate * vulnerable_devices


def vulnerability_with_new_devices_reference(
    time: float,
    initial_vulnerable_devices: float,
    remediation_rate: float,
    new_vulnerability_rate: float,
) -> float:
    """
    Referencia matemática para comparar la simulación de la Actividad 6.

    Se incluye como apoyo opcional de validación; la consigna no exige
    resolver analíticamente esta ecuación diferencial.
    """
    equilibrium_value = new_vulnerability_rate / remediation_rate
    return equilibrium_value + (
        initial_vulnerable_devices - equilibrium_value
    ) * math.exp(-remediation_rate * time)
