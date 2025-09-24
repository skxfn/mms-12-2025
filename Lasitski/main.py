import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception:  # matplotlib may be unavailable in some environments
    plt = None  # type: ignore


# Константы по условию
FLIPS_PER_EXPERIMENT: int = 100
DEFAULT_MONTE_CARLO_RUNS: int = 1000
SERIES_TARGET_LENGTH: int = 5
CONF_LEVEL: float = 0.95


def simulate_single_experiment(p_head: float, flips: int, rng: random.Random) -> (int, int):
    """Провести один эксперимент: подбросить монету flips раз.

    Возвращает количество орлов и длину максимальной серии орлов.
    """
    heads_count = 0
    current_run = 0
    longest_run = 0
    for _ in range(flips):
        is_head = rng.random() < p_head
        if is_head:
            heads_count += 1
            current_run += 1
            if current_run > longest_run:
                longest_run = current_run
        else:
            current_run = 0
    return heads_count, longest_run


def run_monte_carlo(
    p_head: float,
    runs: int,
    flips: int,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Запустить серию экспериментов, вернуть:
    - список количеств орлов по экспериментам
    - список длин максимальных серий орлов по экспериментам
    """
    rng = random.Random(seed)
    heads_list: List[int] = []
    longest_runs: List[int] = []
    for _ in range(runs):
        heads, longest_run = simulate_single_experiment(p_head, flips, rng)
        heads_list.append(heads)
        longest_runs.append(longest_run)
    return heads_list, longest_runs


def mean(values: List[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def probability(condition_list: List[bool]) -> float:
    return sum(1 for x in condition_list if x) / len(condition_list) if condition_list else 0.0


def probability_heads_greater_than(heads_list: List[int], threshold: int) -> float:
    return probability([h > threshold for h in heads_list])


def interval_probabilities(heads_list: List[int]) -> Dict[Tuple[int, int], float]:
    """Вероятности попадания числа орлов в интервалы
    [0,10), [10,20), ..., [80,90), [90,100]
    """
    bins: List[Tuple[int, int]] = [
        (0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
        (50, 60), (60, 70), (70, 80), (80, 90), (90, 100),
    ]
    counts: Dict[Tuple[int, int], int] = {b: 0 for b in bins}
    total = len(heads_list)
    for h in heads_list:
        for left, right in bins:
            if (left <= h < right) or (left == 90 and right == 100 and left <= h <= right):
                counts[(left, right)] += 1
                break
    return {k: v / total for k, v in counts.items()}


def empirical_interval(heads_list: List[int], conf_level: float) -> Tuple[int, int]:
    """Эмпирический предсказательный интервал по квантилям (equal-tailed).

    Возвращает целочисленные границы [L, U].
    """
    if not heads_list:
        return 0, 0
    sorted_vals = sorted(heads_list)
    n = len(sorted_vals)
    alpha = 1.0 - conf_level
    low_idx = max(0, int(alpha / 2 * n))
    high_idx = min(n - 1, int((1 - alpha / 2) * n) - 1)
    return sorted_vals[low_idx], sorted_vals[high_idx]


def probability_of_run_at_least(heads_list: List[int], longest_runs: List[int], target_len: int) -> float:
    return probability([r >= target_len for r in longest_runs])


def plot_expected_heads_vs_p(flips: int) -> None:
    if plt is None:
        print("Matplotlib недоступен для построения графиков.")
        return
    ps = [i / 20 for i in range(21)]  # 0.00..1.00 шаг 0.05
    ys = [flips * p for p in ps]
    plt.style.use('seaborn-v0_8') if 'seaborn-v0_8' in plt.style.available else None
    fig = plt.figure(figsize=(8, 5))
    plt.plot(ps, ys, linewidth=2.5, color='#1f77b4')
    plt.xlabel('Вероятность орла p')
    plt.ylabel('Ожидаемое число орлов')
    plt.title('Ожидаемое число орлов при n = %d' % flips)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'expected_heads_vs_p.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def plot_interval_width_vs_p(runs: int, flips: int, conf_level: float, seed: int) -> None:
    if plt is None:
        print("Matplotlib недоступен для построения графиков.")
        return
    ps = [i / 20 for i in range(21)]  # 0.00..1.00 шаг 0.05
    widths: List[int] = []
    for p in ps:
        heads_list, _ = run_monte_carlo(p, runs, flips, seed)
        low, high = empirical_interval(heads_list, conf_level)
        widths.append(high - low)
    plt.style.use('seaborn-v0_8') if 'seaborn-v0_8' in plt.style.available else None
    fig = plt.figure(figsize=(8, 5))
    plt.plot(ps, widths, linewidth=2.5, color='#ff7f0e')
    plt.xlabel('Вероятность орла p')
    plt.ylabel('Ширина предсказательного интервала')
    plt.title('Ширина %d%% предсказательного интервала (n = %d, MC = %d)'
              % (int(conf_level * 100), flips, runs))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'interval_width_vs_p.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def plot_run_probability_vs_p(runs: int, flips: int, target_len: int, seed: int) -> None:
    if plt is None:
        print("Matplotlib недоступен для построения графиков.")
        return
    ps = [i / 20 for i in range(21)]  # 0.00..1.00 шаг 0.05
    probs: List[float] = []
    for p in ps:
        heads_list, longest_runs = run_monte_carlo(p, runs, flips, seed)
        prob = probability_of_run_at_least(heads_list, longest_runs, target_len)
        probs.append(prob)
    plt.style.use('seaborn-v0_8') if 'seaborn-v0_8' in plt.style.available else None
    fig = plt.figure(figsize=(8, 5))
    plt.plot(ps, probs, linewidth=2.5, color='#2ca02c')
    plt.xlabel('Вероятность орла p')
    plt.ylabel(f'Вероятность серии ≥ {target_len}')
    plt.title('Вероятность наличия серии орлов (n = %d, MC = %d)'
              % (flips, runs))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'run_prob_ge_{target_len}_vs_p.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def plot_max_run_length_vs_p(runs: int, flips: int, seed: int) -> None:
    if plt is None:
        print("Matplotlib недоступен для построения графиков.")
        return
    ps = [i / 20 for i in range(21)]  # 0.00..1.00 шаг 0.05
    expected_max_runs: List[float] = []
    for p in ps:
        _, longest_runs = run_monte_carlo(p, runs, flips, seed)
        expected_max_runs.append(mean(longest_runs))
    plt.style.use('seaborn-v0_8') if 'seaborn-v0_8' in plt.style.available else None
    fig = plt.figure(figsize=(8, 5))
    plt.plot(ps, expected_max_runs, linewidth=2.5, color='#d62728')
    plt.xlabel('Вероятность орла p')
    plt.ylabel('Средняя длина максимальной серии')
    plt.title('Длина максимальной серии (ожидание), n = %d, MC = %d' % (flips, runs))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'max_run_length_vs_p.png'
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Сохранено: {out_path}")


def print_task_results_for_p(p: float, runs: int, flips: int, conf_level: float, seed: int) -> None:
    heads_list, longest_runs = run_monte_carlo(p, runs, flips, seed)

    # Задание 1: среднее число орлов
    avg_heads = mean(heads_list)

    # Задание 2: P(орлов > 60)
    prob_more_than_60 = probability_heads_greater_than(heads_list, 60)

    # Задание 3: вероятности по интервалам
    intervals = interval_probabilities(heads_list)

    # Задание 4: предсказательный интервал с уровнем CONF_LEVEL
    low, high = empirical_interval(heads_list, conf_level)

    # Задание 5: вероятность хотя бы одной серии из SERIES_TARGET_LENGTH орлов
    prob_series = probability_of_run_at_least(heads_list, longest_runs, SERIES_TARGET_LENGTH)

    print("ЗАДАНИЕ 1:")
    print(f"Среднее число орлов за {flips} бросков: {avg_heads:.3f}")
    print("*" * 90)

    print("ЗАДАНИЕ 2:")
    print(f"Вероятность получить > 60 орлов: {prob_more_than_60:.4f}")
    print("*" * 90)

    print("ЗАДАНИЕ 3:")
    print("Вероятности по интервалам числа орлов:")
    for (l, r), prob in intervals.items():
        right_br = ']' if (l, r) == (90, 100) else ')'
        print(f"[{l}, {r}{right_br}: {prob:.4f}")
    print("*" * 90)

    print("ЗАДАНИЕ 4:")
    print(f"{int(conf_level*100)}% предсказательный интервал: [{low}, {high}] (ширина {high - low})")
    print("*" * 90)

    print("ЗАДАНИЕ 5:")
    print(f"Вероятность наличия серии длиной ≥ {SERIES_TARGET_LENGTH}: {prob_series:.4f}")
    print("*" * 90)


def main() -> None:
    # Базовые параметры по умолчанию
    p_default = 0.5
    runs_default = DEFAULT_MONTE_CARLO_RUNS
    flips_default = FLIPS_PER_EXPERIMENT
    seed_default = 42

    print_task_results_for_p(
        p=p_default,
        runs=runs_default,
        flips=flips_default,
        conf_level=CONF_LEVEL,
        seed=seed_default,
    )

    # Небольшое меню для графиков (Задание 6)
    while True:
        print("ГРАФИКИ (Задание 6):")
        print("1) Ожидаемое число орлов (теория)")
        print("2) Ширина предсказательного интервала (MC)")
        print(f"3) Вероятность наличия серии из {SERIES_TARGET_LENGTH} орлов (MC)")
        print("4) Средняя длина максимальной серии (MC)")
        print("0) Выход")
        try:
            choice_str = input("Введите номер графика: ").strip()
            if choice_str == '':
                continue
            choice = int(choice_str)
        except Exception:
            print("Некорректный ввод. Повторите.")
            continue

        if choice == 0:
            break
        elif choice == 1:
            plot_expected_heads_vs_p(flips_default)
        elif choice == 2:
            plot_interval_width_vs_p(runs_default, flips_default, CONF_LEVEL, seed_default)
        elif choice == 3:
            plot_run_probability_vs_p(runs_default, flips_default, SERIES_TARGET_LENGTH, seed_default)
        elif choice == 4:
            plot_max_run_length_vs_p(runs_default, flips_default, seed_default)
        else:
            print("Нет такого пункта меню. Повторите.")


if __name__ == "__main__":
    main()


