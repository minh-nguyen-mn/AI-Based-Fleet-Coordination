# AI-Based Fleet Coordination

This project implements and evaluates multiple AI-based task allocation strategies for autonomous fleet coordination in dynamic environments.

Implemented approaches:

- Greedy nearest-vehicle allocation
- Auction-based decentralised bidding allocation
- Tabular Q-learning allocation

The project evaluates coordination efficiency under dynamically arriving tasks using reproducible experiments and detailed performance logging.

---

# Project Structure

| File | Description |
|---|---|
| `DataGenerationDynamic.py` | Dynamic environment and task generation |
| `fleet_greedy_allocationDynamic.py` | Greedy baseline allocation algorithm |
| `fleet_auction_allocationDynamic.py` | Auction-based allocation implementation |
| `fleet_qlearning_allocationDynamic.py` | Tabular Q-learning allocation implementation |
| `fleet_ComparisonDynamicGiven.py` | Main experiment runner and comparison script |
| `reproducibility_utils.py` | Global seed control and reproducibility utilities |
| `aggregate_summary.csv` | Aggregated metrics across all experiment runs |
| `comparison_metrics.png` | Average performance comparison plots |
| `comparison_distributions.png` | Distribution visualisations across runs |
| `comparison_trends.png` | Trend analysis visualisations |
| `Report_AI_Based_Fleet_Coordination.pdf` | Final report |
| `requirements.txt` | Required Python packages |

---

# Implemented Methods

## 1. Greedy Allocation

The greedy baseline assigns each task to the nearest available vehicle with sufficient battery capacity.

Characteristics:
- Fully local decision-making
- Distance-based assignment
- No global optimisation
- Low computational complexity

---

## 2. Auction-Based Allocation

Vehicles independently compute bids for tasks using:
- travel time,
- task urgency,
- nearby future demand,
- nearby idle vehicles,
- and vehicle idle time.

The system uses:
- decentralised local bid computation,
- followed by centralised greedy bid clearing.

The method is therefore a hybrid decentralised-centralised allocation mechanism.

---

## 3. Q-Learning Allocation

A shared tabular Q-learning policy is used across vehicles.

The implementation includes:
- discretised environment states,
- multiple allocation strategies as actions,
- Boltzmann (softmax) exploration,
- experience replay,
- and reward shaping based on efficiency and urgency.

The learned policy attempts to balance:
- efficiency,
- urgency handling,
- spatial clustering,
- and reduced idle behaviour.

---

# Experimental Setup

The system evaluates all methods under identical dynamically generated environments.

Key experiment properties:
- Dynamic task arrivals
- Multiple autonomous vehicles
- Battery-constrained assignments
- Repeated stochastic simulations
- Fully reproducible seeded execution

The comparison is performed over:
- 1000 independent runs

---

# Performance Metrics

The following metrics are evaluated:

- Tasks completed
- Tasks not serviced
- Throughput
- Engagement time
- Energy per unit workload
- Idle time
- Normalised engagement

Energy consumption is approximated using total engagement time.

Throughput is defined as:
- the sum of serviced task durations.

---

# Reproducibility

The project includes deterministic seed control through:

- NumPy seed management
- Python random seed management
- Fixed experiment configuration

This ensures:
- reproducible results across reruns,
- stable benchmarking,
- and consistent metric outputs.

---

# Running the Experiments

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the comparison experiments
```bash
python fleet_ComparisonDynamicGiven.py
```

This will:

- run all allocation methods,
- execute repeated simulations,
- generate comparison figures,
- and produce aggregated CSV summaries.

---

# Output Files

Running the experiments generates:

| Output | Description |
|---|---|
| `aggregate_summary.csv` |	Aggregated experiment metrics |
| `comparison_metrics.png` | Average metric comparison plots |
| `comparison_distributions.png` | Metric distribution visualisations |
| `comparison_trends.png` | Metric trend plots |

---

# Key Findings

Observed results show:

- Auction-based allocation achieves the best overall throughput and efficiency.
- Q-learning improves adaptability but is limited by tabular discretisation.
- Greedy allocation performs worst under clustered dynamic task distributions.

---

# Additional Notes

The submitted folder contains:

- source code,
- experiment outputs,
- generated figures,
- and the final report.

For a more comprehensive and continuously updated version of the implementation, including additional experiment details and repository history, see:

https://github.com/minh-nguyen-mn/AI-Based-Fleet-Coordination