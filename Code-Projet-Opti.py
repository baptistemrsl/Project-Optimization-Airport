# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 19:24:00 2025

@author: bapti
"""

import random
import pulp

# 1. Data Generation
random.seed(42)
N = 50  # Define resource counts
NUM_GATES = 5
NUM_CREWS = 5  

flights = []
for i in range(N):
    arrival = random.randint(0, 720)
    t_clean = random.randint(15, 30)     
    t_fuel = random.randint(5, 15)       
    t_board = random.randint(20, 40)     
    total_turnaround = t_clean + t_fuel + t_board
    sched_dep = arrival + total_turnaround + random.randint(5, 30)
    flights.append({
        "arrival": arrival,
        "dep_sched": sched_dep,
        "t_clean": t_clean,
        "t_fuel": t_fuel,
        "t_board": t_board
    })

flights.sort(key=lambda f: f["arrival"])

# 2. Baseline Heuristic Scheduling (First-Come-First-Served)
def schedule_baseline(flights, num_gates, num_crews):
    """
    Schedule flights using a greedy FCFS approach.
    Returns a list of results with actual start and finish times and delays for each flight.
    """
    gate_free = [0] * num_gates
    crew_free = [0] * num_crews
    results = []
    for f in flights:
        arr = f["arrival"]; dep_sched = f["dep_sched"]
        t_clean = f["t_clean"]; t_fuel = f["t_fuel"]; t_board = f["t_board"]
        earliest_gate = min(range(num_gates), key=lambda g: gate_free[g] if gate_free[g] >= arr else arr)
        start_service = max(arr, gate_free[earliest_gate])  
        gate_free[earliest_gate] = start_service  
        crew_idx = min(range(num_crews), key=lambda c: crew_free[c])
        start_clean = max(start_service, crew_free[crew_idx])
        end_clean = start_clean + t_clean
        crew_free[crew_idx] = end_clean  
        crew_idx2 = min(range(num_crews), key=lambda c: crew_free[c])
        start_fuel = max(end_clean, crew_free[crew_idx2])
        end_fuel = start_fuel + t_fuel
        crew_free[crew_idx2] = end_fuel
        crew_idx3 = min(range(num_crews), key=lambda c: crew_free[c])
        start_board = max(end_fuel, crew_free[crew_idx3])
        end_board = start_board + t_board
        crew_free[crew_idx3] = end_board
        finish_time = end_board
        actual_dep = max(dep_sched, finish_time)
        gate_free[earliest_gate] = actual_dep  
        delay = max(0, finish_time - dep_sched)
        results.append({
            "arrival": arr,
            "start_service": start_service,
            "finish_time": finish_time,
            "sched_dep": dep_sched,
            "actual_dep": actual_dep,
            "delay": delay
        })
    return results

baseline_schedule = schedule_baseline(flights, NUM_GATES, NUM_CREWS)
avg_turnaround = sum(rec["finish_time"] - rec["arrival"] for rec in baseline_schedule) / N
delayed_flights = sum(1 for rec in baseline_schedule if rec["delay"] > 0)
print(f"Baseline avg turnaround: {avg_turnaround:.1f} min, Delayed flights: {delayed_flights}/{N}")

# 3. MILP Model using PuLP
model = pulp.LpProblem("GroundOpsScheduling", pulp.LpMinimize)

x = pulp.LpVariable.dicts('x', (range(N), range(NUM_GATES)), lowBound=0, upBound=1, cat='Binary')
y = pulp.LpVariable.dicts('y', (range(N), range(NUM_CREWS)), lowBound=0, upBound=1, cat='Binary')
s = pulp.LpVariable.dicts('s', range(N), lowBound=0, cat='Continuous')
f_time = pulp.LpVariable.dicts('f', range(N), lowBound=0, cat='Continuous')

model += pulp.lpSum([f_time[i] for i in range(N)]), "TotalFinishTime"

for i in range(N):
    model += pulp.lpSum(x[i][g] for g in range(NUM_GATES)) == 1, f"AssignGate_{i}"
    model += pulp.lpSum(y[i][c] for c in range(NUM_CREWS)) == 1, f"AssignCrew_{i}"
    model += s[i] >= flights[i]["arrival"], f"Arrival_{i}"
    model += f_time[i] == s[i] + flights[i]["t_clean"] + flights[i]["t_fuel"] + flights[i]["t_board"], f"FinishDef_{i}"
    model += f_time[i] <= flights[i]["dep_sched"], f"DepWindow_{i}"

M = 10000  
for i in range(N):
    for j in range(i+1, N):
        if flights[i]["arrival"] < flights[j]["dep_sched"] and flights[j]["arrival"] < flights[i]["dep_sched"]:
            for g in range(NUM_GATES):
                z_var = pulp.LpVariable(f"z_{i}_{j}_g{g}", lowBound=0, upBound=1, cat='Binary')
                model += z_var >= x[i][g] + x[j][g] - 1, f"OrderBinActivate_{i}_{j}_g{g}"
                model += s[j] >= f_time[i] - M * (1 - z_var), f"GateOrder1_{i}_{j}_g{g}"
                model += s[i] >= f_time[j] - M * z_var, f"GateOrder2_{i}_{j}_g{g}"

for i in range(N):
    for j in range(i+1, N):
        if flights[i]["arrival"] < flights[j]["dep_sched"] and flights[j]["arrival"] < flights[i]["dep_sched"]:
            for c in range(NUM_CREWS):
                w_var = pulp.LpVariable(f"w_{i}_{j}_c{c}", lowBound=0, upBound=1, cat='Binary')
                model += w_var >= y[i][c] + y[j][c] - 1, f"CrewBinActivate_{i}_{j}_c{c}"
                model += s[j] >= f_time[i] - M * (1 - w_var), f"CrewOrder1_{i}_{j}_c{c}"
                model += s[i] >= f_time[j] - M * w_var, f"CrewOrder2_{i}_{j}_c{c}"

# Solve MILP (CBC solver by default)
model.solve(pulp.PULP_CBC_CMD(msg=False))
print(f"MILP solver status: {pulp.LpStatus[model.status]}")
if pulp.LpStatus[model.status] == 'Optimal':
    milp_schedule = []
    for i in range(N):
        start = pulp.value(s[i]); finish = pulp.value(f_time[i])
        gate_assigned = [g for g in range(NUM_GATES) if pulp.value(x[i][g]) > 0.5][0]
        crew_assigned = [c for c in range(NUM_CREWS) if pulp.value(y[i][c]) > 0.5][0]
        delay = max(0, finish - flights[i]["dep_sched"])
        milp_schedule.append({
            "start": start,
            "finish": finish,
            "gate": gate_assigned,
            "crew": crew_assigned,
            "delay": delay
        })
    avg_turnaround_milp = sum(rec["finish"] - flights[i]["arrival"] for i, rec in enumerate(milp_schedule)) / N
    delays_milp = sum(1 for rec in milp_schedule if rec["delay"] > 1e-6)
    print(f"MILP avg turnaround: {avg_turnaround_milp:.1f} min, Delayed flights: {delays_milp}/{N}")

# 4. Genetic Algorithm Approach
import math

def evaluate_schedule(order):
    gate_free = [0] * NUM_GATES
    crew_free = [0] * NUM_CREWS
    total_finish_time = 0
    delays = 0
    for i in order:
        arr = flights[i]["arrival"]; dep_sched = flights[i]["dep_sched"]
        t_clean = flights[i]["t_clean"]; t_fuel = flights[i]["t_fuel"]; t_board = flights[i]["t_board"]
        start_time = arr
        gate_idx = min(range(NUM_GATES), key=lambda g: gate_free[g])
        if gate_free[gate_idx] > start_time:
            start_time = gate_free[gate_idx]
        crew_idx = min(range(NUM_CREWS), key=lambda c: crew_free[c])
        if crew_free[crew_idx] > start_time:
            start_time = max(start_time, crew_free[crew_idx])
        gate_free[gate_idx] = start_time  
        crew_free[crew_idx] = start_time  
        finish_time = start_time + t_clean
        crew_free[crew_idx] = finish_time  
        finish_time += t_fuel
        finish_time += t_board
        gate_free[gate_idx] = finish_time
        crew_free[crew_idx] = finish_time
        total_finish_time += finish_time
        if finish_time > dep_sched:
            delays += 1  
    avg_finish = total_finish_time / len(order)
    fitness = avg_finish + (delays * 1000)  
    return fitness

POP_SIZE = 50
GENERATIONS = 100
MUTATION_RATE = 0.2

population = [random.sample(range(N), N) for _ in range(POP_SIZE)]

best_order = None
best_fitness = float('inf')
for gen in range(GENERATIONS):
    fitness_values = [evaluate_schedule(order) for order in population]
    for idx, fit in enumerate(fitness_values):
        if fit < best_fitness:
            best_fitness = fit
            best_order = population[idx][:]
    selected = []
    for _ in range(POP_SIZE):
        a, b = random.randrange(POP_SIZE), random.randrange(POP_SIZE)
        winner = a if fitness_values[a] < fitness_values[b] else b
        selected.append(population[winner])
    new_pop = []
    for i in range(0, POP_SIZE, 2):
        parent1 = selected[i]; parent2 = selected[i+1] if i+1 < POP_SIZE else selected[i]
        cut1, cut2 = sorted([random.randrange(N) for _ in range(2)])
        child1 = [-1]*N; child2 = [-1]*N
        child1[cut1:cut2+1] = parent1[cut1:cut2+1]
        child2[cut1:cut2+1] = parent2[cut1:cut2+1]
        fill_idx1 = (cut2+1) % N; fill_idx2 = (cut2+1) % N
        for j in range(N):
            gene = parent2[(cut2+1+j) % N]
            if gene not in child1:
                child1[fill_idx1] = gene
                fill_idx1 = (fill_idx1+1) % N
            gene2 = parent1[(cut2+1+j) % N]
            if gene2 not in child2:
                child2[fill_idx2] = gene2
                fill_idx2 = (fill_idx2+1) % N
        new_pop.extend([child1, child2])
    population = new_pop
    for i in range(POP_SIZE):
        if random.random() < MUTATION_RATE:
            a, b = random.sample(range(N), 2)
            population[i][a], population[i][b] = population[i][b], population[i][a]

if best_order is not None:
    ga_schedule = schedule_baseline([flights[i] for i in best_order], NUM_GATES, NUM_CREWS)
    avg_turnaround_ga = sum(rec["finish_time"] - rec["arrival"] for rec in ga_schedule) / N
    delays_ga = sum(1 for rec in ga_schedule if rec["delay"] > 0)
    print(f"GA avg turnaround: {avg_turnaround_ga:.1f} min, Delayed flights: {delays_ga}/{N}")
