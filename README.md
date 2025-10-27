# Executorch

## Docs
1. [Executorch 오픈소스 분석](https://important-chauffeur-cbe.notion.site/ExecuTorch-2976d26f836b8087a09ff9e90930d98e)
2. [모델 실행 검증 도구](https://important-chauffeur-cbe.notion.site/2996d26f836b80a0afeac3ac596c61ac)
3. [실행 로깅 유틸리티](https://important-chauffeur-cbe.notion.site/2996d26f836b805bace7cc2bc7f058ad)
---

## Quick Start
### 0. Recommended Env
```bash
# craete venv with Python 3.10
conda create -n python310 python=3.10

# activate venv
conda activate python310
# verify python version 
python --version
```

### 1. Git Clone
```bash
git clone https://github.com/SeoyeonPark1223/EXECUTORCH.git
cd EXECUTORCH
```

### 2. Installation
#### requirement.txt
```bash
pip install -r requirements.txt
```
#### pyproject.toml for run_bench cli
```bash
pip install .
```

### 3. Test
#### Run Pytest
```bash
pytest -s tests/
```
#### Run CLI Benchmark (example)
```bash
run_bench --help
run_bench --model {model_name} --repeat {repeat_num}
```

## Test Examples
#### Pytest 1:
```bash
pytest -s tests/test_resnet18_equivalence()
```
Example output: 
```
model_name: resnet18
mean_absolute_difference: 0.00000121
pytorch_latency: avg 14.77 ms, max 30.64 ms
executorch_latency: avg 11.44 ms, max 14.74 ms
====== 1 passed, 9 warnings in 10.07s ======
```
#### Pytest 2:
```bash
pytest -s tests/test_benchmark_logs.py
```
Example output:
```
{
"model_name": "resnet18.pte",
"latency_ms_avg": 13.76,
"repeat": 5
}
====== 1 passed, 1 warning in 2.69s ======
```
#### CLI:
```bash
run_bench --model resnet18.pte --repeat 5
```
Example output:
```json
{
"model_name": "resnet18.pte",
"latency_ms_avg": 12.98,
"repeat": 5
}
```

---
### Reference
- [Executorch GitHub](https://github.com/pytorch/executorch)
- [PyTorch Docs](https://docs.pytorch.org/executorch)
- [Introducing ExecuTorch from PyTorch Edge (YouTube)](https://www.youtube.com/watch?v=9U9MNbNcu-w)