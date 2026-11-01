import math

def linear_growth_derivative(t: float, x: float) -> float:
    """x'(t) = 1."""
    return 1.0


def linear_growth_reference(t: float) -> float:
    """x(t) = t."""
    return t


def exponential_decay_derivative(t: float, x: float) -> float:
    """x'(t) = -x(t)."""
    return -x


def exponential_decay_reference(t: float) -> float:
    """Solución de referencia del caso 2: x(t) = e^(-t)."""
    return math.exp(-t)


def vulnerability_remediation_derivative(
    t: float,
    v: float,
    k: float,
) -> float:
    """
    V'(t) = -k V(t).

    remediation_rate (k) representa la rapidez del proceso de remediación.
    """
    return -k * v


def vulnerability_remediation_reference(
    t: float,
    v0: float,
    k: float,
) -> float:
    """Solución referencia: V(t) = V0 * e^(-k*t)."""
    return v0 * math.exp(-k * t)


def vulnerability_with_new_devices_derivative(
    t: float,
    v: float,
    k: float,
    lambd: float,
) -> float:
    """
    V'(t) = lambda - k V(t).

    new_vulnerability_rate (lambda) representa la aparición simplificada
    de nuevos equipos vulnerables por unidad de tiempo.
    """
    return lambd - k * v
