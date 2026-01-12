# 🔥 mlops-research-template


```text
mlops-research-template/
├── src/
│   ├── train.py
│   └── __init__.py
├── configs/
│   └── default.yaml
├── requirements.txt
├── README.md
├── .gitignore
└── pyproject.toml   # אופציונלי (מומלץ)
```

## Research project template for working with the MLOps platform.

### What this repo is for
- Model training & experimentation
- Logging experiments to MLflow
- Reproducible research via configs

### What this repo is NOT
- Infrastructure
- Docker images
- Deployment logic

Those are owned by the **mlops-platform** repository.

---

## Setup (local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure MLflow platform is running:

```
http://localhost:5000
```
---

## Run training

```bash
python src/train.py
```

Or with custom config:

```bash
CONFIG_PATH=configs/default.yaml python src/train.py
```

```bash
docker run --rm \
  --network docker-compose_default \
  -v "../mlops-research-template:/workspace" \
  -w /workspace \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  mlops-training:latest
```
---

## MLflow conventions

### Experiments

* One experiment per project or research topic

### Params

* Hyperparameters only

### Metrics

* Final metrics (accuracy, loss, etc.)

### Tags (required)

| Tag       | Description               |
| --------- | ------------------------- |
| project   | Project name              |
| owner     | Research owner            |
| stage     | research / staging / prod |
| framework | sklearn / pytorch / etc   |

---

## Typical workflow

1. Modify config
2. Run training
3. Inspect results in MLflow UI
4. Iterate

---

## Questions?

Contact the MLOps team.

