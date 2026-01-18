# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 20:31:33 2026

@author: bapti
"""

import os
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import pulp
import matplotlib.pyplot as plt


# Data model
@dataclass
class Flight:
    arrival: int
    dep_sched: int
    t_clean: int
    t_fuel: int
    t_board: int

    @property
    def service_time(self) -> int:
        return self.t_clean + self.t_fuel + self.t_board


def generate_flights(
    n: int,
    seed: int = 42,
    horizon: int = 720,
    buffer_min: int = 5,
    buffer_max: int = 30
) -> List[Flight]:
    random.seed(seed)
    flights: List[Flight] = []
    for _ in range(n):
        arrival = random.randint(0, horizon)
        t_clean = random.randint(15, 30)
        t_fuel = random.randint(5, 15)
        t_board = random.randint(20, 40)
        total = t_clean + t_fuel + t_board
        dep_sched = arrival + total + random.randint(buffer_min, buffer_max)
        flights.append(Flight(arrival, dep_sched, t_clean, t_fuel, t_board))
    flights.sort(key=lambda f: f.arrival)
    return flights


def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

# Metrics
def compute_metrics(schedule: List[Dict], flights: List[Flight]) -> Dict[str, float]:
    n = len(schedule)
    avg_turnaround = sum(s["finish"] - flights[s["i"]].arrival for s in schedule) / n
    delayed_flights = sum(1 for s in schedule if s["delay"] > 1e-9)
    total_delay = sum(s["delay"] for s in schedule)
    avg_delay = total_delay / n
    return {
        "avg_turnaround": avg_turnaround,
        "delayed_flights": delayed_flights,
        "total_delay": total_delay,
        "avg_delay": avg_delay
    }


# FCFS baseline (consistent: 1 crew per flight for whole service)
def schedule_fcfs(flights: List[Flight], num_gates: int, num_crews: int) -> List[Dict]:
    gate_free = [0] * num_gates
    crew_free = [0] * num_crews

    schedule = []
    for i, f in enumerate(flights):
        best = None  # (start, gate, crew)
        for g in range(num_gates):
            for c in range(num_crews):
                start = max(f.arrival, gate_free[g], crew_free[c])
                if best is None or start < best[0]:
                    best = (start, g, c)

        start, g, c = best
        finish = start + f.service_time
        delay = max(0, finish - f.dep_sched)

        gate_free[g] = finish
        crew_free[c] = finish

        schedule.append({"i": i, "start": start, "finish": finish, "gate": g, "crew": c, "delay": delay})

    return schedule


# MILP (feasible + lexicographic objective)
def solve_milp(
    flights: List[Flight],
    num_gates: int,
    num_crews: int,
    time_limit_sec: int = 30,
    msg: bool = False
) -> Tuple[str, List[Dict]]:
    """
    Two-stage (lexicographic) objective:
      Stage 1: minimize total delay sum_i delay_i
      Stage 2: subject to optimal total delay, minimize total completion time sum_i f_i
    """
    n = len(flights)
    G = range(num_gates)
    C = range(num_crews)

    # Big-M: choose safely large to avoid accidental constraint activation issues
    M = 100000

    model = pulp.LpProblem("GroundOpsScheduling", pulp.LpMinimize)

    x = pulp.LpVariable.dicts("x", (range(n), G), cat="Binary")  # gate assignment
    y = pulp.LpVariable.dicts("y", (range(n), C), cat="Binary")  # crew assignment
    s = pulp.LpVariable.dicts("s", range(n), lowBound=0, cat="Continuous")
    ftime = pulp.LpVariable.dicts("f", range(n), lowBound=0, cat="Continuous")
    delay = pulp.LpVariable.dicts("delay", range(n), lowBound=0, cat="Continuous")

    # Constraints: assignment + timing + tardiness
    for i in range(n):
        model += pulp.lpSum(x[i][g] for g in G) == 1, f"AssignGate_{i}"
        model += pulp.lpSum(y[i][c] for c in C) == 1, f"AssignCrew_{i}"

        model += s[i] >= flights[i].arrival, f"Arrival_{i}"
        model += ftime[i] == s[i] + flights[i].service_time, f"FinishDef_{i}"
        model += delay[i] >= ftime[i] - flights[i].dep_sched, f"Tardiness_{i}"

    # Non-overlap on gates
    for i in range(n):
        for j in range(i + 1, n):
            for g in G:
                o = pulp.LpVariable(f"og_{i}_{j}_{g}", cat="Binary")  # 1 => i before j on gate g
                # active only if both are on gate g (x_i_g = x_j_g = 1)
                model += (
                    s[j] >= ftime[i] - M * (1 - o) - M * (2 - x[i][g] - x[j][g]),
                    f"GateNoOv1_{i}_{j}_{g}",
                )
                model += (
                    s[i] >= ftime[j] - M * (o) - M * (2 - x[i][g] - x[j][g]),
                    f"GateNoOv2_{i}_{j}_{g}",
                )

    # Non-overlap on crews
    for i in range(n):
        for j in range(i + 1, n):
            for c in C:
                o = pulp.LpVariable(f"oc_{i}_{j}_{c}", cat="Binary")  # 1 => i before j on crew c
                model += (
                    s[j] >= ftime[i] - M * (1 - o) - M * (2 - y[i][c] - y[j][c]),
                    f"CrewNoOv1_{i}_{j}_{c}",
                )
                model += (
                    s[i] >= ftime[j] - M * (o) - M * (2 - y[i][c] - y[j][c]),
                    f"CrewNoOv2_{i}_{j}_{c}",
                )

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit_sec)

    # ---- Stage 1: minimize total delay
    model.setObjective(pulp.lpSum(delay[i] for i in range(n)))
    model.solve(solver)
    status1 = pulp.LpStatus[model.status]
    if status1 not in ("Optimal", "Not Solved"):
        return status1, []

    best_total_delay = sum((pulp.value(delay[i]) or 0.0) for i in range(n))

    # ---- Stage 2: fix total delay (within epsilon) and minimize total completion time
    EPS = 1e-4
    model += pulp.lpSum(delay[i] for i in range(n)) <= best_total_delay + EPS, "FixTotalDelay"
    model.setObjective(pulp.lpSum(ftime[i] for i in range(n)))
    model.solve(solver)
    status2 = pulp.LpStatus[model.status]
    status = status2

    if status not in ("Optimal", "Not Solved"):
        return status, []

    # Extract schedule
    schedule: List[Dict] = []
    for i in range(n):
        start = float(pulp.value(s[i]) or 0.0)
        finish = float(pulp.value(ftime[i]) or 0.0)
        dly = float(pulp.value(delay[i]) or 0.0)

        # argmax assignment
        gate_vals = [(g, float(pulp.value(x[i][g]) or 0.0)) for g in G]
        crew_vals = [(c, float(pulp.value(y[i][c]) or 0.0)) for c in C]
        gate = max(gate_vals, key=lambda t: t[1])[0]
        crew = max(crew_vals, key=lambda t: t[1])[0]

        schedule.append({"i": i, "start": start, "finish": finish, "gate": gate, "crew": crew, "delay": max(0.0, dly)})

    return status, schedule


# Genetic Algorithm (permutation -> decode with same resource logic)
def decode_order_to_schedule(order: List[int], flights: List[Flight], num_gates: int, num_crews: int) -> List[Dict]:
    gate_free = [0] * num_gates
    crew_free = [0] * num_crews

    schedule = []
    for idx in order:
        f = flights[idx]
        best = None
        for g in range(num_gates):
            for c in range(num_crews):
                start = max(f.arrival, gate_free[g], crew_free[c])
                if best is None or start < best[0]:
                    best = (start, g, c)

        start, g, c = best
        finish = start + f.service_time
        dly = max(0, finish - f.dep_sched)

        gate_free[g] = finish
        crew_free[c] = finish

        schedule.append({"i": idx, "start": start, "finish": finish, "gate": g, "crew": c, "delay": dly})
    return schedule


def ga_optimize(
    flights: List[Flight],
    num_gates: int,
    num_crews: int,
    pop_size: int = 60,
    generations: int = 120,
    mutation_rate: float = 0.2,
    lambda_delay: float = 1000.0,
    seed: int = 123
) -> List[Dict]:
    random.seed(seed)
    n = len(flights)

    def fitness(order: List[int]) -> float:
        sch = decode_order_to_schedule(order, flights, num_gates, num_crews)
        return sum(s["finish"] + lambda_delay * s["delay"] for s in sch)

    def tournament_select(pop, fits):
        a, b = random.sample(range(len(pop)), 2)
        return pop[a] if fits[a] < fits[b] else pop[b]

    def ordered_crossover(p1, p2):
        n_ = len(p1)
        a, b = sorted(random.sample(range(n_), 2))
        child = [-1] * n_
        child[a:b+1] = p1[a:b+1]
        fill = [gene for gene in p2 if gene not in child]
        j = 0
        for i_ in range(n_):
            if child[i_] == -1:
                child[i_] = fill[j]
                j += 1
        return child

    population = [random.sample(range(n), n) for _ in range(pop_size)]

    best_order = None
    best_fit = float("inf")

    for _ in range(generations):
        fits = [fitness(ind) for ind in population]

        for ind, fval in zip(population, fits):
            if fval < best_fit:
                best_fit = fval
                best_order = ind[:]

        selected = [tournament_select(population, fits) for _ in range(pop_size)]

        new_pop = []
        for i in range(0, pop_size, 2):
            p1 = selected[i]
            p2 = selected[i + 1] if i + 1 < pop_size else selected[i]
            c1 = ordered_crossover(p1, p2)
            c2 = ordered_crossover(p2, p1)
            new_pop.extend([c1, c2])

        population = new_pop[:pop_size]

        for i in range(pop_size):
            if random.random() < mutation_rate:
                a, b = random.sample(range(n), 2)
                population[i][a], population[i][b] = population[i][b], population[i][a]

    if best_order is None:
        best_order = list(range(n))
    return decode_order_to_schedule(best_order, flights, num_gates, num_crews)


# Reporting utilities
def save_summary_csv(summary: Dict[str, Dict[str, float]], path: str = "results/summary.csv"):
    lines = ["method,avg_turnaround,delayed_flights,total_delay,avg_delay\n"]
    for m, metrics in summary.items():
        lines.append(
            f"{m},{metrics['avg_turnaround']:.4f},{int(metrics['delayed_flights'])},{metrics['total_delay']:.4f},{metrics['avg_delay']:.4f}\n"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def plot_results(summary: Dict[str, Dict[str, float]], delays_per_method: Dict[str, List[float]]):
    methods = list(summary.keys())

    # Bar: avg turnaround
    plt.figure()
    plt.bar(methods, [summary[m]["avg_turnaround"] for m in methods])
    plt.ylabel("Average turnaround time (min)")
    plt.title("Average Turnaround Time by Method")
    plt.tight_layout()
    plt.savefig("figures/avg_turnaround.png", dpi=200)
    plt.close()

    # Bar: delayed flights
    plt.figure()
    plt.bar(methods, [summary[m]["delayed_flights"] for m in methods])
    plt.ylabel("Number of delayed flights")
    plt.title("Delayed Flights by Method")
    plt.tight_layout()
    plt.savefig("figures/delayed_flights.png", dpi=200)
    plt.close()

    # Histogram: delays distribution
    plt.figure()
    for m in methods:
        plt.hist(delays_per_method[m], bins=15, alpha=0.6, label=m)
    plt.xlabel("Delay (min)")
    plt.ylabel("Count")
    plt.title("Delay Distribution by Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/delay_hist.png", dpi=200)
    plt.close()


# Main
def main():
    ensure_dirs()

    # Parameters (match your report)
    N = 50
    NUM_GATES = 5
    NUM_CREWS = 5
    SEED_DATA = 42

    # GA penalty (used only inside GA fitness)
    LAMBDA_GA = 1000.0

    flights = generate_flights(N, seed=SEED_DATA)

    # FCFS
    fcfs_schedule = schedule_fcfs(flights, NUM_GATES, NUM_CREWS)
    fcfs_metrics = compute_metrics(fcfs_schedule, flights)

    # MILP (lexicographic)
    milp_status, milp_schedule = solve_milp(
        flights, NUM_GATES, NUM_CREWS,
        time_limit_sec=30,
        msg=False
    )
    if milp_status not in ("Optimal", "Not Solved"):
        milp_metrics = {"avg_turnaround": float("nan"), "delayed_flights": float("nan"), "total_delay": float("nan"), "avg_delay": float("nan")}
    else:
        milp_metrics = compute_metrics(milp_schedule, flights)

    # GA
    ga_schedule = ga_optimize(
        flights, NUM_GATES, NUM_CREWS,
        pop_size=60,
        generations=120,
        mutation_rate=0.2,
        lambda_delay=LAMBDA_GA,
        seed=123
    )
    ga_metrics = compute_metrics(ga_schedule, flights)

    summary = {
        "FCFS": fcfs_metrics,
        "MILP": milp_metrics,
        "GA": ga_metrics
    }

    # Print results
    print("\n=== Summary Metrics ===")
    print(f"Data seed: {SEED_DATA} | Gates: {NUM_GATES} | Crews: {NUM_CREWS}")
    print(f"MILP status: {milp_status}\n")
    for m in summary:
        print(
            f"{m}: avg_turnaround={summary[m]['avg_turnaround']:.2f} "
            f"| delayed_flights={int(summary[m]['delayed_flights']) if summary[m]['delayed_flights']==summary[m]['delayed_flights'] else 'NA'} "
            f"| total_delay={summary[m]['total_delay']:.2f}"
        )

    # Save CSV + figures
    save_summary_csv(summary, "results/summary.csv")

    delays_per_method = {
        "FCFS": [s["delay"] for s in fcfs_schedule],
        "MILP": [s["delay"] for s in milp_schedule] if milp_status in ("Optimal", "Not Solved") else [],
        "GA": [s["delay"] for s in ga_schedule]
    }
    plot_results(summary, delays_per_method)

    print("\nSaved: results/summary.csv")
    print("Saved figures: figures/avg_turnaround.png, figures/delayed_flights.png, figures/delay_hist.png")


if __name__ == "__main__":
    main()