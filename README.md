# 一个简易的视觉大模型评测脚本

这个项目用于在 `MathVision` 和 `MathVista` 数据集上做视觉推理评测，包含数据下载、单样本测试、批量推理和结果评估。

## 项目结构

```text
project/
|- scripts/
|  |- download_datasets.py
|  |- call_api.py
|  |- run_inference.py
|  |- run_inference_sc.py
|  |- run_inference_verify_retry.py
|  |- evaluate.py
|  |- inference_common.py
|  `- answer_utils.py
|- data/
|- outputs/
|- api_config.example.json
|- requirements.txt
`- README.md
```

## 脚本说明

- `download_datasets.py`：下载并整理数据集
- `call_api.py`：测试单条样本
- `run_inference.py`：普通批量推理
- `run_inference_sc.py`：self-consistency 推理
- `evaluate.py`：评估预测结果

## 环境安装

安装依赖：

```bash
pip install -r requirements.txt
```

## 配置接口

先复制配置文件：

```bash
copy api_config.example.json api_config.json
```

然后填写你自己的接口信息：

```json
{
  "api_key": "your_api_key_here",
  "base_url": "http://127.0.0.1:8000/v1",
  "model": "Qwen2.5-VL-3B-Instruct"
}
```

如果你使用其他 OpenAI-compatible 服务，只需要修改 `base_url` 和 `model`。

## 使用流程

### 1. 下载数据

下载 `MathVision`：

```bash
python scripts/download_datasets.py --dataset mathvision --split testmini
```

下载 `MathVista`：

```bash
python scripts/download_datasets.py --dataset mathvista --split testmini
```

数据会保存在：

```text
data/<dataset>/<split>/
```

### 2. 测试单条样本

```bash
python scripts/call_api.py --input data/mathvision/testmini/records.jsonl --sample-index 0
```

### 3. 普通批量推理

```bash
python scripts/run_inference.py --input data/mathvision/testmini/records.jsonl --output outputs/mathvision_testmini_predictions.jsonl
```

只跑前 20 条：

```bash
python scripts/run_inference.py --input data/mathvision/testmini/records.jsonl --output outputs/mathvision_testmini_predictions.jsonl --limit 20
```

从指定位置继续跑：

```bash
python scripts/run_inference.py --input data/mathvision/testmini/records.jsonl --output outputs/mathvision_testmini_predictions.jsonl --start-index 101 --append
```

### 4. Self-consistency 推理

```bash
python scripts/run_inference_sc.py --input data/mathvision/testmini/records.jsonl --output outputs/mathvision_sc_predictions.jsonl --num-samples 4
```

### 5. 评估结果

规则评估：

```bash
python scripts/evaluate.py outputs/mathvision_testmini_predictions.jsonl
```

使用 LLM 评估：

```bash
python scripts/evaluate.py outputs/mathvision_testmini_predictions.jsonl --eval-mode llm
```

评估后会生成：

```text
outputs/*_eval_*.json
outputs/*_badcases_*.jsonl
```
