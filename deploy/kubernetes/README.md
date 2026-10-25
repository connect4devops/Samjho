# Samjho Kubernetes Deployment

This directory contains the Kubernetes manifests to deploy Samjho into a production environment.

## Prerequisites
- A Kubernetes cluster (e.g., kind, minikube for local testing, or a cloud provider's managed Kubernetes).
- `kubectl` configured to interact with your cluster.

## Deployment Steps

1. **Configure Secrets:**
   Copy the example secret file and add your actual JarvisLabs API Token:
   ```bash
   cp secret.example.yaml secret.yaml
   # Edit secret.yaml and add your JARVIS_API_TOKEN
   ```
   *Note: Do not commit `secret.yaml` to version control.*

2. **Apply Manifests:**
   Deploy the application using `kubectl` (Kustomize is supported natively):
   ```bash
   kubectl apply -k .
   ```
   Or apply directly:
   ```bash
   kubectl apply -f namespace.yaml
   kubectl apply -f configmap.yaml
   kubectl apply -f secret.yaml
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   ```

3. **Verify Deployment:**
   Check the status of the pods and the service:
   ```bash
   kubectl get pods -n samjho
   kubectl get svc -n samjho
   ```

## Local Testing with Docker

To build and run Samjho locally using Docker:

1. Build the image:
   ```bash
   docker build -t samjho:latest .
   ```

2. Run the container:
   ```bash
   docker run -p 8090:8090 --env-file .env samjho:latest
   ```

Access the application at `http://localhost:8090`.

## Testing with local Kubernetes (minikube / kind)

### kind
```bash
kind create cluster
docker build -t samjho:latest .
kind load docker-image samjho:latest
kubectl apply -k deploy/kubernetes/
```

### minikube
```bash
minikube start
eval $(minikube docker-env)
docker build -t samjho:latest .
kubectl apply -k deploy/kubernetes/
```

Port forward to access locally:
```bash
kubectl port-forward svc/samjho -n samjho 8090:80
```
Then navigate to `http://localhost:8090`.

## Architecture

```
User Request
    │
    ▼
Ingress (samjho.example.com)
    │
    ▼
Service (samjho)
    │
    ▼
Deployment (samjho) ──► JarvisLabs Serverless API
```

## Scaling and Health

The application leverages the existing `/api/health` endpoint for liveness and readiness probes.
Because the actual AI inference happens remotely (via JarvisLabs Serverless API), scaling up the local pods will not increase GPU throughput. The `Deployment` is configured to handle concurrent HTTP connections and API requests efficiently.
