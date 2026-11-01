from pathlib import Path
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from metodos_numericos import euler, euler_improved
from modelos import (
    linear_growth_derivative,
    linear_growth_reference,
    exponential_decay_derivative,
    exponential_decay_reference,
    vulnerability_remediation_derivative,
    vulnerability_remediation_reference,
    vulnerability_with_new_devices_derivative,
)


OUTPUT_DIRECTORY = Path(__file__).resolve().parent
FIGURES_DIRECTORY = OUTPUT_DIRECTORY / "figuras"
TABLES_DIRECTORY = OUTPUT_DIRECTORY / "tablas"

FIGURES_DIRECTORY.mkdir(exist_ok=True)
TABLES_DIRECTORY.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def calculate_absolute_errors(
    times: list[float],
    approximate_values: list[float],
    reference_function,
) -> np.ndarray:
    """Calcula E_n = |x(t_n) - x_n| para cada instante simulado."""
    reference_values = np.array([reference_function(time) for time in times])
    return np.abs(reference_values - np.array(approximate_values))


def save_figure(file_name: str) -> None:
    """Guarda la figura actual en formato SVG."""
    plt.tight_layout()
    figure_path = FIGURES_DIRECTORY / f"{file_name}.svg"
    plt.savefig(figure_path)
    plt.close()


def save_csv(file_name: str, headers: list[str], rows: list[tuple]) -> None:
    """Guarda una tabla de resultados en formato CSV."""
    table_path = TABLES_DIRECTORY / file_name
    with table_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)


def plot_reference_and_approximations(
    reference_function,
    simulation_results: list[tuple[str, list[float], list[float]]],
    final_time: float,
    title: str,
    y_label: str,
    file_name: str,
) -> None:
    """Grafica una referencia y varias aproximaciones numéricas."""
    reference_times = np.linspace(0, final_time, 500)
    reference_values = [reference_function(time) for time in reference_times]

    plt.figure(figsize=(8, 4.8))
    plt.plot(
        reference_times,
        reference_values,
        color="black",
        linewidth=2,
        label="Solución de referencia",
    )

    for label, times, values in simulation_results:
        plt.plot(times, values, marker="o", markersize=2.5, label=label)

    plt.xlabel("Tiempo t")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    save_figure(file_name)


def run_task_1() -> None:
    """Ejecuta los dos casos de verificación solicitados en la sección 5."""
    final_time = 10.0

    # Caso 1: crecimiento lineal, para h = 1, 0.5 y 0.1.
    linear_results = []
    linear_rows = []
    for step_size in [1.0, 0.5, 0.1]:
        times, values, step_count = euler(
            linear_growth_derivative, 0.0, 0.0, final_time, step_size
        )
        errors = calculate_absolute_errors(times, values, linear_growth_reference)
        linear_results.append((f"Euler, h={step_size:g}", times, values))
        linear_rows.append((
            step_size, step_count, values[-1], errors[-1], errors.max()
        ))

    plot_reference_and_approximations(
        linear_growth_reference,
        linear_results,
        final_time,
        "Caso 1: crecimiento lineal",
        "x(t)",
        "crecimiento_lineal",
    )
    save_csv(
        "crecimiento_lineal.csv",
        ["step_size", "step_count", "final_value", "final_error", "maximum_error"],
        linear_rows,
    )

    # Caso 2: decrecimiento exponencial, para h = 1, 0.5, 0.1 y 0.01.
    exponential_results = []
    exponential_rows = []
    for step_size in [1.0, 0.5, 0.1, 0.01]:
        times, values, step_count = euler(
            exponential_decay_derivative, 1.0, 0.0, final_time, step_size
        )
        errors = calculate_absolute_errors(times, values, exponential_decay_reference)
        exponential_results.append((f"Euler, h={step_size:g}", times, values))
        exponential_rows.append((
            step_size, step_count, values[-1], errors[-1], errors.max()
        ))

    plot_reference_and_approximations(
        exponential_decay_reference,
        exponential_results,
        final_time,
        "Caso 2: decrecimiento exponencial",
        "x(t)",
        "decrecimiento_exponencial",
    )
    save_csv(
        "decrecimiento_exponencial.csv",
        ["step_size", "step_count", "final_value", "final_error", "maximum_error"],
        exponential_rows,
    )


def run_task_2() -> None:
    """Calcula y grafica el error absoluto para los pasos indicados."""
    final_time = 10.0
    step_sizes = [1.0, 0.5, 0.1, 0.05, 0.01]
    error_rows = []
    error_results = []

    for step_size in step_sizes:
        times, values, step_count = euler(
            exponential_decay_derivative, 1.0, 0.0, final_time, step_size
        )
        errors = calculate_absolute_errors(times, values, exponential_decay_reference)
        error_rows.append((
            step_size, step_count, errors[-1], errors.max()
        ))
        error_results.append((f"h={step_size:g}", times, errors.tolist()))

    plt.figure(figsize=(8, 4.8))
    for label, times, errors in error_results:
        plt.plot(times, errors, label=label)
    plt.xlabel("Tiempo t")
    plt.ylabel("Error absoluto")
    plt.title("Error absoluto de Euler en función del tiempo")
    plt.legend()
    save_figure("error_absoluto_vs_tiempo")

    save_csv(
        "error_absoluto_vs_tiempo.csv",
        ["step_size", "step_count", "final_error", "maximum_error"],
        error_rows,
    )


def run_task_3() -> None:
    """Compara ambos métodos para h = 1, 0.5 y 0.1."""
    final_time = 10.0
    step_sizes = [1.0, 0.5, 0.1]
    comparison_rows = []

    for method_name, method_function in [
        ("Euler", euler),
        ("Euler improved", euler_improved),
    ]:
        for step_size in step_sizes:
            times, values, step_count = method_function(
                exponential_decay_derivative, 1.0, 0.0, final_time, step_size
            )
            errors = calculate_absolute_errors(
                times, values, exponential_decay_reference
            )
            comparison_rows.append((
                method_name, step_size, values[-1], errors[-1], step_count
            ))

    # La gráfica permite comparar los dos métodos con el mismo paso h=1.
    reference_times = np.linspace(0, final_time, 500)
    plt.figure(figsize=(8, 4.8))
    plt.plot(
        reference_times,
        [exponential_decay_reference(time) for time in reference_times],
        color="black",
        linewidth=2,
        label="Solución de referencia",
    )

    for method_name, method_function in [
        ("Euler", euler),
        ("Euler improved", euler_improved),
    ]:
        times, values, _ = method_function(
            exponential_decay_derivative, 1.0, 0.0, final_time, 1.0
        )
        plt.plot(times, values, marker="o", label=f"{method_name}, h=1")

    plt.xlabel("Tiempo t")
    plt.ylabel("x(t)")
    plt.title("Euler frente a Euler mejorado")
    plt.legend()
    save_figure("euler_vs_euler_mejorado")

    save_csv(
        "euler_vs_euler_mejorado.csv",
        ["method", "step_size", "final_value", "final_error", "step_count"],
        comparison_rows,
    )


def run_task_4() -> None:
    """Simula las organizaciones A, B y C para los pasos solicitados."""
    initial_vulnerable_devices = 1000.0
    final_time = 20.0
    organizations = {
        "A": 0.1,
        "B": 0.3,
        "C": 0.7,
    }
    step_sizes = [1.0, 0.5, 0.1]
    requested_times = [5.0, 10.0, 15.0]
    result_rows = []

    # Gráfica principal solicitada: tres organizaciones con h=1 y sus referencias.
    reference_times = np.linspace(0, final_time, 500)
    plt.figure(figsize=(8, 4.8))

    for organization_name, remediation_rate in organizations.items():
        derivative = lambda time, value, rate=remediation_rate: (
            vulnerability_remediation_derivative(time, value, rate)
        )
        reference = lambda time, rate=remediation_rate: (
            vulnerability_remediation_reference(
                time, initial_vulnerable_devices, rate
            )
        )

        times, values, _ = euler(
            derivative, initial_vulnerable_devices, 0.0, final_time, 1.0
        )
        plt.plot(times, values, marker="o", markersize=2.5,
                 label=f"Euler, organización {organization_name}")
        plt.plot(reference_times, [reference(time) for time in reference_times],
                 linestyle="--", label=f"Referencia, organización {organization_name}")

        # Se calculan valores y errores en t=5, 10 y 15 para cada tamaño de paso.
        for step_size in step_sizes:
            times_for_step, values_for_step, step_count = euler(
                derivative,
                initial_vulnerable_devices,
                0.0,
                final_time,
                step_size,
            )
            errors = calculate_absolute_errors(
                times_for_step, values_for_step, reference
            )

            for requested_time in requested_times:
                time_index = round(requested_time / step_size)
                approximate_value = values_for_step[time_index]
                reference_value = reference(requested_time)
                result_rows.append((
                    organization_name,
                    remediation_rate,
                    step_size,
                    requested_time,
                    approximate_value,
                    reference_value,
                    abs(reference_value - approximate_value),
                    step_count,
                    errors.max(),
                ))

    plt.xlabel("Tiempo t (días)")
    plt.ylabel("Cantidad de equipos vulnerables V(t)")
    plt.title("Remediación en las organizaciones A, B y C")
    plt.legend(fontsize=8)
    save_figure("remediacion_organizaciones")

    save_csv(
        "remediacion_organizaciones.csv",
        [
            "organization", "remediation_rate_k", "step_size", "time",
            "euler_value", "reference_value", "absolute_error",
            "step_count", "maximum_error",
        ],
        result_rows,
    )

    # Gráfica adicional: efecto del tamaño del paso para la organización B.
    remediation_rate = organizations["B"]
    derivative = lambda time, value: (
        vulnerability_remediation_derivative(time, value, remediation_rate)
    )
    reference = lambda time: vulnerability_remediation_reference(
        time, initial_vulnerable_devices, remediation_rate
    )
    results = []
    for step_size in step_sizes:
        times, values, _ = euler(
            derivative, initial_vulnerable_devices, 0.0, final_time, step_size
        )
        results.append((f"Euler, h={step_size:g}", times, values))

    plot_reference_and_approximations(
        reference,
        results,
        final_time,
        "Organización B: efecto del tamaño del paso",
        "Equipos vulnerables V(t)",
        "organizacion_b_efecto_paso",
    )


def run_task_5() -> None:
    """
    Genera experimentos numéricos para distintos valores de h*k.

    La identificación de la sucesión geométrica y su expresión explícita
    deben explicarse en el informe; este código ilustra el comportamiento.
    """
    initial_vulnerable_devices = 1000.0
    remediation_rate = 0.3
    hk_values = [0.3, 0.9, 1.0, 1.5, 2.5]
    iteration_count = 12
    result_rows = []

    plt.figure(figsize=(8, 4.8))
    for hk_value in hk_values:
        step_size = hk_value / remediation_rate
        multiplier = 1 - hk_value
        vulnerable_devices = initial_vulnerable_devices
        values = [vulnerable_devices]

        for iteration in range(iteration_count):
            # Recurrencia V_(n+1) = (1-hk) * V_n.
            vulnerable_devices = multiplier * vulnerable_devices
            values.append(vulnerable_devices)
            result_rows.append((
                hk_value, iteration + 1, multiplier, vulnerable_devices
            ))

        plt.plot(range(iteration_count + 1), values, marker="o",
                 label=f"h*k={hk_value:g}")

    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Iteración n")
    plt.ylabel("Equipos vulnerables V_n")
    plt.title("Comportamiento del modelo discreto")
    plt.legend()
    save_figure("modelo_discreto")

    save_csv(
        "modelo_discreto.csv",
        ["h_times_k", "iteration", "sequence_multiplier", "vulnerable_devices"],
        result_rows,
    )


def run_task_6() -> None:
    """Simula V'(t) = lambda - kV(t) para los cuatro valores de lambda."""
    initial_vulnerable_devices = 1000.0
    remediation_rate = 0.3
    new_vulnerability_rates = [0.0, 50.0, 100.0, 200.0]
    final_time = 20.0
    step_size = 0.1
    result_rows = []

    plt.figure(figsize=(8, 4.8))

    for new_vulnerability_rate in new_vulnerability_rates:
        derivative = lambda time, value, rate=new_vulnerability_rate: (
            vulnerability_with_new_devices_derivative(
                time, value, remediation_rate, rate
            )
        )

        times, values, step_count = euler(
            derivative,
            initial_vulnerable_devices,
            0.0,
            final_time,
            step_size,
        )
        equilibrium_value = new_vulnerability_rate / remediation_rate

        plt.plot(times, values, label=f"Euler, lambda={new_vulnerability_rate:g}")

        result_rows.append((
            new_vulnerability_rate,
            values[-1],
            equilibrium_value,
            step_count,
        ))

    plt.xlabel("Tiempo t")
    plt.ylabel("Equipos vulnerables V(t)")
    plt.title("Aparición de nuevas vulnerabilidades")
    plt.legend(fontsize=8)
    save_figure("nuevas_vulnerabilidades")

    save_csv(
        "nuevas_vulnerabilidades.csv",
        [
            "new_vulnerability_rate_lambda", "euler_value_at_final_time",
            "equilibrium_value", "step_count",
        ],
        result_rows,
    )


def run_all_tasks() -> None:
    tasks = [
        ("Tarea 1: verificación de Euler", run_task_1),
        ("Tarea 2: error numérico", run_task_2),
        ("Tarea 3: Euler vs Euler mejorado", run_task_3),
        ("Tarea 4: gestión de vulnerabilidades", run_task_4),
        ("Tarea 5: modelo discreto", run_task_5),
        ("Tarea 6: nuevas vulnerabilidades", run_task_6),
    ]
    total = len(tasks)

    for index, (description, task) in enumerate(tasks, start=1):
        print(f"[{index}/{total}] {description}", flush=True)
        task()

    print(f"Experiments completed. Results saved in: {OUTPUT_DIRECTORY}", flush=True)


if __name__ == "__main__":
    run_all_tasks()
