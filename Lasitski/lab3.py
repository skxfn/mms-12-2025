from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
import math
import statistics


# Матрица выплат W (см. labs/2.md)
# Строки: мой ход (0 = сотрудничать, 1 = предать)
# Столбцы: ход соперника (0 = сотрудничать, 1 = предать)
W: List[List[int]] = [
    [3, 0],  # I play 0
    [5, 1],  # I play 1
]


class Strategy:

    name: str

    def reset(self) -> None:
        return

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        raise NotImplementedError


# Всегда предаёт (Defect).
class Alex(Strategy):
    name = "Alex"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        return 1


# Всегда сотрудничает (Cooperate).
class Bob(Strategy):
    name = "Bob"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        return 0


# «Око за око» (Tit-for-Tat): начинает с сотрудничества, затем повторяет ход соперника.
class Clara(Strategy):
    name = "Clara"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        if not opponent_history:
            return 0
        return opponent_history[-1]


# Анти Tit-for-Tat: начинает с сотрудничества, затем играет противоположно последнему ходу соперника.
class Denis(Strategy):
    name = "Denis"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        if not opponent_history:
            return 0
        return 1 - opponent_history[-1]


# Периодическое предательство: предаёт раз в 20 ходов, иначе сотрудничает.
class Emma(Strategy):
    name = "Emma"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        return 1 if (round_index % 20 == 0) else 0


# «Жёсткий триггер» (Grim Trigger): сотрудничает до первого предательства соперника, затем всегда предаёт.
class Frida(Strategy):
    name = "Frida"

    def __init__(self) -> None:
        self._opponent_defected_ever = False

    def reset(self) -> None:
        self._opponent_defected_ever = False

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        if opponent_history and opponent_history[-1] == 1:
            self._opponent_defected_ever = True
        # Жёсткий триггер: сотрудничаем, пока соперник не предаст; затем всегда предаём
        return 1 if self._opponent_defected_ever else 0


# «Павлов» (Win-Stay, Lose-Shift): если предыдущая выплата >= 3 — повторяет ход, иначе меняет.
class George(Strategy):
    name = "George"

    def __init__(self) -> None:
        self._last_move: int | None = None

    def reset(self) -> None:
        self._last_move = None

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        # Павлов (Win-Stay, Lose-Shift): если получили >= 3, повторяем ход, иначе меняем
        if not my_history:
            self._last_move = 0
            return 0

        last_my = my_history[-1]
        last_op = opponent_history[-1]
        last_payoff = W[last_my][last_op]
        if last_payoff >= 3:
            assert self._last_move is not None
            return self._last_move
        new_move = 1 - last_my
        self._last_move = new_move
        return new_move


@dataclass
class MatchResult:
    history_a: List[int]
    history_b: List[int]
    score_a: int
    score_b: int
    max_dominating_run_a: int
    max_dominating_run_b: int


def play_match(strategy_a: Strategy, strategy_b: Strategy, rounds: int = 200) -> MatchResult:
    strategy_a.reset()
    strategy_b.reset()

    history_a: List[int] = []
    history_b: List[int] = []
    score_a = 0
    score_b = 0

    current_dom_run_a = 0
    current_dom_run_b = 0
    max_dom_run_a = 0
    max_dom_run_b = 0

    for r in range(1, rounds + 1):
        move_a = strategy_a.choose(history_a, history_b, r)
        move_b = strategy_b.choose(history_b, history_a, r)

        history_a.append(move_a)
        history_b.append(move_b)

        payoff_a = W[move_a][move_b]
        payoff_b = W[move_b][move_a]
        score_a += payoff_a
        score_b += payoff_b

        # Обновляем счётчики доминирующих серий
        if payoff_a == 5 and payoff_b == 0:
            current_dom_run_a += 1
            current_dom_run_b = 0
        elif payoff_b == 5 and payoff_a == 0:
            current_dom_run_b += 1
            current_dom_run_a = 0
        else:
            current_dom_run_a = 0
            current_dom_run_b = 0

        if current_dom_run_a > max_dom_run_a:
            max_dom_run_a = current_dom_run_a
        if current_dom_run_b > max_dom_run_b:
            max_dom_run_b = current_dom_run_b

    return MatchResult(
        history_a=history_a,
        history_b=history_b,
        score_a=score_a,
        score_b=score_b,
        max_dominating_run_a=max_dom_run_a,
        max_dominating_run_b=max_dom_run_b,
    )


def _geometric_count(success_probability: float, max_len: int = 20) -> int:
    count = 1
    while count < max_len and random.random() > success_probability:
        count += 1
    return count


# Случайная стратегия: сотрудничает и предаёт с вероятностью 0.5.
class Hank(Strategy):
    name = "Hank"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        return 1 if random.random() < 0.5 else 0


# «Почти всегда сотрудничает»: сотрудничает с вероятностью 0.9, иначе предаёт.
class Ivan(Strategy):
    name = "Ivan"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        return 0 if random.random() < 0.9 else 1


# Мягкий Tit-for-Tat: отвечает предательством на предательство, но прощает с p = 0.25.
class Jack(Strategy):
    name = "Jack"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        if not opponent_history:
            return 0
        last_op = opponent_history[-1]
        if last_op == 0:
            return 0
        # forgive with probability 0.25
        return 0 if random.random() < 0.25 else 1


# Нойзовый копировщик: повторяет последний ход соперника, но с p = 0.25 переворачивает его.
class Kevin(Strategy):
    name = "Kevin"

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        if not opponent_history:
            base = 0
        else:
            base = opponent_history[-1]
        # flip with probability 0.25
        if random.random() < 0.25:
            return 1 - base
        return base


# Периодическая стратегия: предаёт каждые P ходов, где P ~ Uniform{1..50}.
class Lucas(Strategy):
    name = "Lucas"

    def __init__(self) -> None:
        self._period = random.randint(1, 50)

    def reset(self) -> None:
        self._period = random.randint(1, 50)

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        return 1 if (round_index % self._period == 0) else 0


# Блочная стратегия: чередует блоки одинаковых ходов случайной длины (0..20).
class Max(Strategy):
    name = "Max"

    def __init__(self) -> None:
        self._current_move = 0
        self._remaining = random.randint(0, 20)

    def reset(self) -> None:
        self._current_move = 0
        self._remaining = random.randint(0, 20)

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        if self._remaining <= 0:
            self._current_move = 1 - self._current_move
            self._remaining = random.randint(0, 20)
        self._remaining -= 1
        return self._current_move


# Наказание геометрической длины: после предательства соперника наказывает случайное число ходов; иначе сотрудничает.
class Natan(Strategy):
    name = "Natan"

    def __init__(self) -> None:
        self._punish_left = 0

    def reset(self) -> None:
        self._punish_left = 0

    def choose(self, my_history: List[int], opponent_history: List[int], round_index: int) -> int:
        if self._punish_left > 0:
            self._punish_left -= 1
            return 1
        if opponent_history and opponent_history[-1] == 1:
            # punish for a geometric number of steps (p=0.2)
            self._punish_left = _geometric_count(0.2, max_len=10)
            self._punish_left -= 1
            return 1
        return 0


@dataclass
class SampleStats:
    mean: float
    median: float
    variance: float
    mode_estimate: float


def _mode_estimate(values: List[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values[0])
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        return float(vmin)
    k = max(1, int(math.log2(1 + n)))
    width = (vmax - vmin) / k if k > 0 else (vmax - vmin)
    if width == 0:
        return float(values[0])
    # assign to bins [vmin + i*width, vmin + (i+1)*width)
    bins: List[List[float]] = [[] for _ in range(k)]
    for x in values:
        idx = int((x - vmin) / width)
        if idx >= k:
            idx = k - 1
        bins[idx].append(x)
    best_bin = max(bins, key=lambda b: len(b))
    if not best_bin:
        return float(values[0])
    return float(sum(best_bin) / len(best_bin))


def _compute_stats(values: List[float]) -> SampleStats:
    if not values:
        return SampleStats(0.0, 0.0, 0.0, 0.0)
    mean_v = statistics.fmean(values)
    median_v = statistics.median(values)
    variance_v = statistics.pvariance(values) if len(values) > 1 else 0.0
    mode_v = _mode_estimate(values)
    return SampleStats(mean_v, median_v, variance_v, mode_v)


def simulate_pair(a_cls: type[Strategy], b_cls: type[Strategy], rounds: int, simulations: int) -> Tuple[SampleStats, SampleStats]:
    scores_a: List[float] = []
    scores_b: List[float] = []
    for _ in range(simulations):
        a = a_cls()
        b = b_cls()
        result = play_match(a, b, rounds=rounds)
        scores_a.append(float(result.score_a))
        scores_b.append(float(result.score_b))
    return _compute_stats(scores_a), _compute_stats(scores_b)


def run_tournament(rounds: int = 200, simulations: int = 1000):
    strategies: List[type[Strategy]] = [
        type(Alex()),
        type(Bob()),
        type(Clara()),
        type(Denis()),
        type(Emma()),
        type(Frida()),
        type(George()),
        Hank,
        Ivan,
        Jack,
        Kevin,
        Lucas,
        Max,
        Natan,
    ]

    names: List[str] = [cls().name for cls in strategies]
    # pairwise_stats[nameA][nameB] -> SampleStats for A against B
    pairwise_stats: Dict[str, Dict[str, SampleStats]] = {name: {} for name in names}

    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            a_cls = strategies[i]
            b_cls = strategies[j]
            stats_a, stats_b = simulate_pair(a_cls, b_cls, rounds=rounds, simulations=simulations)
            name_a = a_cls().name
            name_b = b_cls().name
            pairwise_stats[name_a][name_b] = stats_a
            pairwise_stats[name_b][name_a] = stats_b

    # aggregate mean per strategy across opponents (mean of means)
    aggregate_mean: Dict[str, float] = {}
    for name in names:
        opps = [pairwise_stats[name][other].mean for other in names if other != name]
        aggregate_mean[name] = statistics.fmean(opps) if opps else 0.0

    return names, pairwise_stats, aggregate_mean, rounds, simulations


def save_results_markdown(path: str, names: List[str], pairwise_stats: Dict[str, Dict[str, SampleStats]], aggregate_mean: Dict[str, float], rounds: int, simulations: int) -> None:
    def format_matrix(value_getter) -> List[str]:
        order = names
        rows: List[str] = []
        rows.append("| Стратегия | " + " | ".join(order) + " |")
        rows.append("| --- | " + " | ".join(["---:" for _ in order]) + " |")
        for r in order:
            cells: List[str] = []
            for c in order:
                if r == c:
                    cells.append("-")
                else:
                    cells.append(value_getter(pairwise_stats[r][c]))
            rows.append(f"| {r} | " + " | ".join(cells) + " |")
        rows.append("")
        return rows

    lines: List[str] = []
    lines.append("# Результаты турнира: Лабораторная №3")
    lines.append("")
    lines.append(f"Параметры симуляции: раундов = {rounds}, симуляций на пару = {simulations}")
    lines.append("")

    lines.append("## Средние очки по стратегиям (среднее по всем соперникам)")
    lines.append("")
    lines.append("| Стратегия | Среднее |")
    lines.append("| --- | ---: |")
    for name in sorted(names, key=lambda n: aggregate_mean[n], reverse=True):
        lines.append(f"| {name} | {aggregate_mean[name]:.2f} |")
    lines.append("")

    lines.append("## Матрица средних очков (A против B)")
    lines.extend(format_matrix(lambda s: f"{s.mean:.2f}"))

    lines.append("## Матрица медианных очков (A против B)")
    lines.extend(format_matrix(lambda s: f"{s.median:.2f}"))

    lines.append("## Матрица моды (оценка по интервалам) (A против B)")
    lines.extend(format_matrix(lambda s: f"{s.mode_estimate:.2f}"))

    lines.append("## Матрица дисперсий (A против B)")
    lines.extend(format_matrix(lambda s: f"{s.variance:.2f}"))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_pairwise_ascii_table():
    return ""


def main() -> None:
    names, pairwise_stats, aggregate_mean, rounds, simulations = run_tournament(rounds=200, simulations=1000)
    save_results_markdown(
        path="lab3_results.md",
        names=names,
        pairwise_stats=pairwise_stats,
        aggregate_mean=aggregate_mean,
        rounds=rounds,
        simulations=simulations,
    )
    print("Mean scores (avg across opponents):")
    for name in sorted(names, key=lambda n: aggregate_mean[n], reverse=True):
        print(f"  {name:>7}: {aggregate_mean[name]:.2f}")


if __name__ == "__main__":
    main()


