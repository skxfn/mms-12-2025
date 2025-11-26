from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


Number = float | int


@dataclass
class SimulationConfig:
    steps: int = 400
    discard: int = 100
    tolerance: float = 1e-4


@dataclass
class HostParasiteParams:
    growth_rate: float = 2.5  # b
    attack_rate: float = 0.8  # a
    conversion_rate: float = 1.2  # c

    def as_tuple(self) -> Tuple[float, float, float]:
        return self.growth_rate, self.attack_rate, self.conversion_rate


def iterate_map(
    map_fn: Callable[[float], float],
    x0: float,
    steps: int,
) -> np.ndarray:
    values = np.empty(steps, dtype=float)
    values[0] = x0
    for i in range(1, steps):
        values[i] = map_fn(values[i - 1])
    return values


def exponential_model(r: float, x0: float, config: SimulationConfig) -> np.ndarray:
    return iterate_map(lambda x: r * x, x0, config.steps)


def logistic_model(r: float, x0: float, config: SimulationConfig) -> np.ndarray:
    return iterate_map(lambda x: r * x * (1.0 - x), x0, config.steps)


def moran_model(r: float, x0: float, config: SimulationConfig) -> np.ndarray:
    return iterate_map(lambda x: x * np.exp(r * (1.0 - x)), x0, config.steps)


def nicholson_bailey_model(
    params: HostParasiteParams,
    x0: float,
    y0: float,
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    b, a, c = params.as_tuple()
    hosts = np.empty(config.steps, dtype=float)
    parasites = np.empty(config.steps, dtype=float)
    hosts[0], parasites[0] = x0, y0

    for i in range(1, config.steps):
        hosts[i] = b * hosts[i - 1] * np.exp(-a * parasites[i - 1])
        infections = 1.0 - np.exp(-a * parasites[i - 1])
        parasites[i] = c * hosts[i - 1] * infections
    return hosts, parasites


def detect_attractor(
    sequence: Sequence[float],
    config: SimulationConfig,
) -> str:
    tail = np.array(sequence[-config.discard :])
    if np.any(np.isnan(tail)) or np.any(tail < 0) or np.any(~np.isfinite(tail)):
        return "нестабильно"

    rounded = np.round(tail, 3)
    unique = np.unique(rounded)

    if unique.size == 1:
        return "устойчиво"
    if unique.size <= 8:
        return f"период {unique.size}"
    return "хаос"


def scan_parameter_space(
    generator: Callable[[float], np.ndarray],
    param_values: Iterable[float],
    config: SimulationConfig,
) -> list[tuple[float, str]]:
    summary = []
    for value in param_values:
        data = generator(value)
        state = detect_attractor(data, config)
        summary.append((value, state))
    return summary


def plot_time_series(ax: plt.Axes, data: np.ndarray, title: str, color: str) -> None:
    ax.plot(data, color=color, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("x_t")
    ax.grid(alpha=0.2)


def visualize_core_models(config: SimulationConfig) -> None:
    x0 = 0.2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_time_series(
        axes[0],
        exponential_model(r=1.05, x0=x0, config=config),
        "Экспоненциальный рост (r=1.05)",
        "tab:green",
    )
    plot_time_series(
        axes[1],
        logistic_model(r=3.3, x0=x0, config=config),
        "Логистическая модель (r=3.3)",
        "tab:blue",
    )
    plot_time_series(
        axes[2],
        moran_model(r=2.5, x0=x0, config=config),
        "Модель Морана (r=2.5)",
        "tab:orange",
    )
    fig.suptitle("Сравнение базовых моделей роста")
    fig.tight_layout()


def visualize_host_parasite(config: SimulationConfig) -> None:
    params = HostParasiteParams()
    hosts, parasites = nicholson_bailey_model(params, 0.6, 0.2, config)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(hosts, label="хозяева", color="tab:blue")
    axes[0].plot(parasites, label="паразиты", color="tab:red")
    axes[0].set_title("Численности во времени")
    axes[0].legend()
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("Популяция")
    axes[0].grid(alpha=0.2)

    axes[1].plot(hosts, parasites, color="tab:purple", linewidth=1.2)
    axes[1].set_title("Фазовый портрет (x_t, y_t)")
    axes[1].set_xlabel("хозяева")
    axes[1].set_ylabel("паразиты")
    axes[1].grid(alpha=0.2)

    fig.suptitle("Модель Никольсона–Бейли")
    fig.tight_layout()


def visualize_logistic_transitions(config: SimulationConfig) -> None:
    r_values = np.linspace(2.4, 4.0, 80)
    summary = scan_parameter_space(
        lambda r: logistic_model(r=r, x0=0.2, config=config),
        r_values,
        config,
    )
    labels = {"устойчиво": 0, "хаос": 2}
    y = [labels.get(state, 1) for _, state in summary]

    fig, ax = plt.subplots(figsize=(10, 3))
    cmap = {0: "tab:green", 1: "tab:orange", 2: "tab:red"}
    colors = [cmap[val] for val in y]
    ax.scatter(r_values, y, c=colors, s=30)
    ax.set_title("Изменение поведения логистической модели при росте r")
    ax.set_xlabel("r")
    ax.set_yticks([0, 1, 2], ["устойчиво", "период", "хаос"])
    ax.grid(alpha=0.2)
    fig.tight_layout()

    text_rows = "\n".join(f"r={r:.2f}: {state}" for r, state in summary[::10])
    print("Примеры состояний логистической модели:\n", text_rows)


def run_all(config: SimulationConfig, show: bool = True) -> None:
    visualize_core_models(config)
    visualize_host_parasite(config)
    visualize_logistic_transitions(config)
    if show:
        plt.show()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Лабораторная работа №4: исследование моделей роста популяций."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=400,
        help="число итераций во временном ряде (default: 400)",
    )
    parser.add_argument(
        "--discard",
        type=int,
        default=100,
        help="длина хвоста для анализа устойчивости (default: 100)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="не вскрывать окна matplotlib (полезно при тестах)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = SimulationConfig(steps=args.steps, discard=args.discard)
    run_all(config, show=not args.no_show)


if __name__ == "__main__":
    main()

