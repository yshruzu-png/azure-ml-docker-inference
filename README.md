# Dockerized ML Inference Service on Azure Machine Learning

A scikit-learn Iris classifier served by FastAPI, packaged as a Docker image, pushed to Azure Container Registry, and deployed as an Azure ML managed online endpoint (custom container).

## Architecture

```
train.py → model.pkl → FastAPI /predict → Docker image (linux/amd64)
   → Azure Container Registry → Azure ML environment
   → online endpoint → blue deployment → HTTPS REST API
```

## Tech stack

Python 3.10 · scikit-learn · FastAPI · Uvicorn · Docker · Azure Container Registry · Azure Machine Learning (CLI v2) · Managed identity with AcrPull

## Project structure

| File | Purpose |
| --- | --- |
| `train.py` | Trains a RandomForest on the Iris dataset, saves `model.pkl` |
| `inference.py` | FastAPI app: `GET /` health check, `POST /predict` |
| `requirements.txt` | Pinned dependencies (same versions for training and serving) |
| `Dockerfile` | Builds the inference image on `python:3.10-slim` |
| `azure/environment.yml` | Azure ML environment pointing to the ACR image, with port and routes |
| `azure/endpoint.yml` | Managed online endpoint with key auth |
| `azure/deployment.yml` | `blue` deployment, Standard_DS2_v2, liveness/readiness probes |
| `azure/sample-request.json` | Sample input |

## Run locally

```bash
# Train inside the serving image so model.pkl matches the container's libraries
docker run --rm -v "$PWD":/app -w /app python:3.10-slim \
  sh -c "pip install -r requirements.txt && python train.py"

docker build --platform linux/amd64 -t iris-ml-service:1.0 .
docker run -d -p 8000:8000 --name iris-service iris-ml-service:1.0

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" --data-binary @azure/sample-request.json
# {"prediction":0}   -> Iris Setosa
```

Swagger UI: http://localhost:8000/docs

## Deploy to Azure ML

```bash
az acr login --name <ACR_NAME>
docker tag iris-ml-service:1.0 <ACR_NAME>.azurecr.io/iris-ml-service:1.0
docker push <ACR_NAME>.azurecr.io/iris-ml-service:1.0

az ml environment create --file azure/environment.yml -g <RG> -w <WORKSPACE>
az ml online-endpoint create --file azure/endpoint.yml -g <RG> -w <WORKSPACE>
# Grant the endpoint's managed identity AcrPull on the registry, then:
az ml online-deployment create --file azure/deployment.yml -g <RG> -w <WORKSPACE> --all-traffic

az ml online-endpoint invoke -n iris-docker-endpoint \
  --request-file azure/sample-request.json -g <RG> -w <WORKSPACE>
```

## Results

The same image returned `{"prediction": 0}` locally, through the Azure CLI, through REST with a bearer key, and from the Azure ML Studio Test tab.

## Lessons learned

- **Apple Silicon:** build with `--platform linux/amd64`, or the Azure deployment fails with `exec format error`.
- **Version drift:** training and serving must use the same scikit-learn version, so the model is trained inside the serving image.
- **Permissions:** the endpoint's managed identity needs `AcrPull` on the registry before deployment.
- **Cost:** delete the resource group after testing; the endpoint VM bills hourly.
