import math


# ---------------------------------------------------------------------------
# Casos de verificación del método (propuesta, sección 5).
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Gestión de vulnerabilidades (propuesta, secciones 8 y 10).
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bonus: modelo de crecimiento logístico.
# ---------------------------------------------------------------------------

def logistic_growth_derivative(time: float, value: float) -> float:
    """Bonus: y'(t) = y(t) * (1 - y(t))."""
    return value * (1.0 - value)


def logistic_growth_reference(time: float, initial_value: float) -> float:
    """Solución exacta del modelo logístico para 0 < initial_value < 1."""
    if not 0.0 < initial_value < 1.0:
        raise ValueError("initial_value must be between 0 and 1 (exclusive).")

    return 1.0 / (
        1.0 + ((1.0 - initial_value) / initial_value) * math.exp(-time)
    )
