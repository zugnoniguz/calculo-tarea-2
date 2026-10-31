# Simulación numérica de sistemas dinámicos aplicada a ciberseguridad

Código de apoyo para el proyecto de Cálculo Aplicado.

## Archivos

- `metodos_numericos.py`: implementaciones de Euler y Euler mejorado.
- `modelos.py`: funciones que representan los modelos matemáticos de la propuesta.
- `experimentos.py`: simulaciones, cálculos de error, tablas y gráficos de las actividades 1 a 6.
- `main.py`: archivo desde el que se ejecuta el proyecto.
- `figuras/`: se generan gráficos PNG y PDF.
- `tablas/`: se generan tablas de resultados CSV.

## Requisitos

Python 3.10 o superior y las bibliotecas:

```bash
python -m pip install numpy matplotlib
```

## Ejecución

Desde la carpeta del proyecto:

```bash
python main.py
```

Los resultados se guardan automáticamente en las carpetas `figuras/` y `tablas/`.

## Relación con la propuesta

| Archivo / función | Parte de la propuesta |
|---|---|
| `metodos_numericos.py` - `euler` | Sección 4 y Actividad 1 |
| `metodos_numericos.py` - `euler_improved` | Sección 7 y Actividad 3 |
| `experimentos.py` - `run_activity_1` | Sección 5: casos de verificación |
| `experimentos.py` - `run_activity_2` | Sección 6: error numérico |
| `experimentos.py` - `run_activity_3` | Sección 7: Euler mejorado |
| `experimentos.py` - `run_activity_4` | Sección 8: gestión de vulnerabilidades |
| `experimentos.py` - `run_activity_5` | Sección 9: modelo discreto |
| `experimentos.py` - `run_activity_6` | Sección 10: nuevas vulnerabilidades |

Las secciones 11 a 13 de la propuesta son principalmente de investigación,
interpretación y discusión escrita. No se resuelven con estos programas:
deben desarrollarse en el informe y respaldarse con fuentes cuando corresponda.

## Convenciones

- El código y los nombres de variables están en inglés.
- Los comentarios explicativos están en español.
- Los nombres buscan describir el significado de los datos y parámetros.
- Las funciones de los métodos numéricos devuelven tres elementos:
  tiempos, valores aproximados y cantidad de pasos.
