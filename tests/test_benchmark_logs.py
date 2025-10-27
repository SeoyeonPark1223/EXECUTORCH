import json
import os
import pytest
from benchmark.run_benchmark import run_benchmark

@pytest.mark.parametrize("model, repeat", [("resnet18.pte", 5)])
def test_benchmark_logs(tmp_path, model, repeat):
    """
    1. run_benchmark() -> JSON result
    2. Validate log
    """

    # 1. run_benchmark() -> JSON result
    log = run_benchmark(model, repeat=repeat)

    # 2. Validate log
    assert isinstance(log, dict)
    assert "model_name" in log
    assert "latency_ms_avg" in log
    assert "repeat" in log
    assert isinstance(json.dumps(log), str)