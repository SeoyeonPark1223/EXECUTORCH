import torch
import time
import os
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from executorch.runtime import Runtime
from model.export_resnet18 import export_resnet18

def test_resnet18_equivalence(tmp_path):
    """
    Pytorch (ResNet18) 모델과 ExecuTorch 모델 (resnet18.pte) 결과 비교
    - 평균 절대 오차 (mean absolute difference)
    - 실행시간 (평균, 최대 latency)
    """

    # 0. Common Values
    num_inference = 30
    example_input = torch.randn(1, 3, 224, 224)

    # 1-1. Load ResNet18
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

    # 1-2. Validate ResNet18
    torch_latencies = []
    for _ in range(num_inference):
        start = time.perf_counter() # second 단위
        with torch.no_grad(): # inference mode
            torch_output = model(example_input)
        torch_latencies.append((time.perf_counter() - start) * 1000) # millisecond 단위

    torch_output_np = torch_output.detach().cpu().numpy() # torch inference result
    torch_avg = np.mean(torch_latencies) # torch 평균 latency
    torch_max = np.max(torch_latencies)  # torch max latency

    # 2-1. Load ExecuTorch
    execu_path = export_resnet18()

    if not os.path.exists(execu_path):
        raise FileNotFoundError(f"Model not found: {execu_path}")

    runtime = Runtime.get()
    method = runtime.load_program(execu_path).load_method("forward")

    # 2-2. Validate ExecuTorch
    execu_latencies = []
    for _ in range(num_inference):
        start = time.perf_counter()
        execu_outputs = method.execute([example_input])
        execu_latencies.append((time.perf_counter() - start) * 1000)

    execu_output_np = execu_outputs[0].detach().cpu().numpy() # type(execu_outputs) -> <class 'list'>
    execu_avg = np.mean(execu_latencies)
    execu_max = np.max(execu_latencies)

    # 3. Compare Mean Absolute Difference
    mad = np.mean(np.abs(torch_output_np - execu_output_np))

    # 4. Print results
    print(f"\nmodel_name: resnet18")
    print(f"mean_absolute_difference: {mad:.8f}")
    print(f"pytorch_latency: avg {torch_avg:.2f} ms, max {torch_max:.2f} ms")
    print(f"executorch_latency: avg {execu_avg:.2f} ms, max {execu_max:.2f} ms")


if __name__ == "__main__":
    test_resnet18_equivalence()