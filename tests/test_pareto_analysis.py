from evaluation.pareto_analysis import ParetoPoint, pareto_frontier


def test_pareto_frontier_drops_dominated_points() -> None:
    points = [
        ParetoPoint("best", macro_f1=0.8, latency_ms=100.0, memory_mb=50.0, deployment_feasibility=0.9),
        ParetoPoint("dominated", macro_f1=0.6, latency_ms=150.0, memory_mb=60.0, deployment_feasibility=0.7),
        ParetoPoint("tradeoff", macro_f1=0.7, latency_ms=80.0, memory_mb=70.0, deployment_feasibility=0.8),
    ]

    frontier = pareto_frontier(points)

    assert [point.model for point in frontier] == ["best", "tradeoff"]
