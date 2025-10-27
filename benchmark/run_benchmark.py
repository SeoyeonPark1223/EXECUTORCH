import argparse
import json
import os
import time
import torch
import numpy as np
from executorch.runtime import Runtime


def run_benchmark(model: str, repeat: int):
    """
    Core benchmark function

    1. Load ExecuTorch 모델 (resnet18.pte)
    2. repeat param value만큼 추론 반복 실행
    3. latency_ms_avg 계산
    """

    model_dir = "./model/model_outputs/"
    model_path = model_dir + model
    example_input = torch.randn(1, 3, 224, 224)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 1. Load ExecuTorch
    runtime = Runtime.get()
    method = runtime.load_program(model_path).load_method("forward")

    # 2. Validate ExecuTorch
    latencies = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = method.execute([example_input])
        latencies.append((time.perf_counter() - start) * 1000)

    latency_ms_avg = np.mean(latencies)

    # 3. JSON log
    log = {
        "model_name": model,
        "latency_ms_avg": round(latency_ms_avg, 2),
        "repeat": repeat,
    }

    # 6. Print console
    print(json.dumps(log, indent=2))
    return log


def main():
    """
    CLI entry point

    입력 예시: run_bench --model resnet18.pte --repeat 5
    """
    parser = argparse.ArgumentParser(description="ExecuTorch Model Benchmark CLI")
    parser.add_argument("--model", type=str, required=True, help="Path to .pte model file")
    parser.add_argument("--repeat", type=int, default=5, help="Number of repetitions")

    args = parser.parse_args()
    run_benchmark(args.model, args.repeat)

if __name__ == "__main__":
    main()