"""
Kubernetes Fundamentals
========================
From "what is K8s" to Pods, Deployments, Services, ConfigMaps,
and the core workflow of deploying containerized applications at scale.
"""

TOPIC_NAME = "☸️ Kubernetes Fundamentals"
CATEGORY = "Orchestration"

# ─────────────────────────────────────────────────────────────────────────────
# THEORY
# ─────────────────────────────────────────────────────────────────────────────

THEORY = """
## Kubernetes (K8s) — Container Orchestration

### What is Kubernetes?
Kubernetes is an open-source container orchestration platform that automates
**deploying, scaling, and managing** containerized applications. Think of Docker
as building/running a single container; Kubernetes manages fleets of them.

### The Problem K8s Solves

```
    Without K8s                              With K8s
    ───────────                              ────────

    "My container crashed" 😰               Auto-restarts crashed containers ✅
    "Traffic spike — need more" 😰          Auto-scales replicas up/down ✅
    "How to update without downtime?" 😰    Rolling updates, zero downtime ✅
    "Which server runs what?" 😰            K8s decides placement for you ✅
    "Config & secrets everywhere" 😰        Centralized ConfigMaps & Secrets ✅
```

### Key Terminology

| Term              | Definition                                                            |
|-------------------|-----------------------------------------------------------------------|
| **Cluster**       | A set of machines (nodes) running K8s. The whole system.              |
| **Node**          | A single machine (VM or physical) in the cluster.                     |
| **Pod**           | Smallest deployable unit. Wraps one or more containers.               |
| **Deployment**    | Manages a set of identical Pods (replicas, rolling updates).          |
| **Service**       | Stable network endpoint to reach a set of Pods (load balancing).      |
| **Namespace**     | Virtual sub-cluster for isolation (dev, staging, prod).               |
| **ConfigMap**     | Store non-sensitive config as key-value pairs.                        |
| **Secret**        | Store sensitive data (passwords, tokens) — base64 encoded.            |
| **Ingress**       | Manages external HTTP/HTTPS access to Services.                       |
| **PV / PVC**      | PersistentVolume / Claim — storage that outlives Pods.               |
| **kubectl**       | The CLI tool to interact with K8s clusters.                          |

### Architecture Overview

```
    ┌─────────────────── Kubernetes Cluster ───────────────────┐
    │                                                          │
    │   ┌─────────── Control Plane ──────────┐                │
    │   │                                     │                │
    │   │  ┌──────────┐   ┌──────────────┐   │                │
    │   │  │ API Server│   │  Scheduler    │   │                │
    │   │  └─────┬────┘   └──────────────┘   │                │
    │   │        │                            │                │
    │   │  ┌─────┴──────┐  ┌──────────────┐  │                │
    │   │  │ Controller  │  │    etcd       │  │                │
    │   │  │  Manager    │  │ (state store) │  │                │
    │   │  └────────────┘  └──────────────┘  │                │
    │   └─────────────────────────────────────┘                │
    │                    │                                     │
    │         ┌──────────┼──────────┐                         │
    │         ▼          ▼          ▼                         │
    │   ┌──────────┐ ┌──────────┐ ┌──────────┐              │
    │   │  Node 1   │ │  Node 2   │ │  Node 3   │             │
    │   │           │ │           │ │           │             │
    │   │ ┌───────┐ │ │ ┌───────┐ │ │ ┌───────┐ │             │
    │   │ │ Pod A  │ │ │ │ Pod B  │ │ │ │ Pod C  │ │            │
    │   │ │ Pod D  │ │ │ │ Pod E  │ │ │ │ Pod F  │ │            │
    │   │ └───────┘ │ │ └───────┘ │ │ └───────┘ │             │
    │   │           │ │           │ │           │             │
    │   │ ┌───────┐ │ │ ┌───────┐ │ │ ┌───────┐ │             │
    │   │ │kubelet │ │ │ │kubelet │ │ │ │kubelet │ │            │
    │   │ │kube-   │ │ │ │kube-   │ │ │ │kube-   │ │            │
    │   │ │proxy   │ │ │ │proxy   │ │ │ │proxy   │ │            │
    │   │ └───────┘ │ │ └───────┘ │ │ └───────┘ │             │
    │   └──────────┘ └──────────┘ └──────────┘              │
    │                                                          │
    └──────────────────────────────────────────────────────────┘

    Control Plane:  Makes decisions (scheduling, scaling, healing)
    Nodes:          Run the actual workloads (Pods)
    etcd:           Distributed key-value store (single source of truth)
    kubelet:        Agent on each node that talks to the API server
    kube-proxy:     Networking rules for Services on each node
```

### The K8s Mental Model: Desired vs Actual State

```
    YOU declare:  "I want 3 replicas of my app running"   (desired state)
                        │
                        ▼
    K8s observes: "Currently 2 are running"                (actual state)
                        │
                        ▼
    K8s acts:     "Starting 1 more Pod"                    (reconciliation)
                        │
                        ▼
    Loop:         Continuously watches and reconciles       (control loop)
```

> This is the **declarative** model: you tell K8s WHAT you want,
> not HOW to do it. K8s figures out the rest.

### Docker → K8s Object Mapping

| Docker Concept             | Kubernetes Equivalent              |
|----------------------------|------------------------------------|
| `docker run`               | Pod                                |
| `docker-compose` services  | Deployment + Service               |
| Port mapping `-p 80:80`    | Service (ClusterIP, NodePort, LB)  |
| Volumes `-v`               | PersistentVolumeClaim              |
| Environment `-e`           | ConfigMap / Secret                 |
| `docker network`           | Namespace + NetworkPolicy          |
| `docker-compose up`        | `kubectl apply -f`                 |
"""

# ─────────────────────────────────────────────────────────────────────────────
# COMMAND REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

COMMANDS = """
### kubectl Core Commands

| Command                                     | What it does                              |
|---------------------------------------------|-------------------------------------------|
| `kubectl get pods`                          | List Pods in current namespace            |
| `kubectl get pods -A`                       | List Pods across ALL namespaces           |
| `kubectl get deployments`                   | List Deployments                          |
| `kubectl get services`                      | List Services                             |
| `kubectl get all`                           | List all major resources                  |
| `kubectl get nodes`                         | List cluster nodes                        |
| `kubectl apply -f file.yaml`               | Create/update resources from YAML         |
| `kubectl delete -f file.yaml`              | Delete resources defined in YAML          |
| `kubectl describe pod <name>`              | Detailed info about a Pod                 |
| `kubectl logs <pod>`                        | View Pod logs                             |
| `kubectl logs -f <pod>`                     | Follow Pod logs (live)                    |
| `kubectl exec -it <pod> -- bash`           | Shell into a Pod                          |
| `kubectl port-forward <pod> 8080:80`       | Forward local port to Pod                 |
| `kubectl scale deploy <name> --replicas=5` | Scale replicas up/down                    |
| `kubectl rollout status deploy/<name>`     | Watch a rolling update                    |
| `kubectl rollout undo deploy/<name>`       | Rollback to previous version              |
| `kubectl config get-contexts`              | List available cluster contexts           |
| `kubectl config use-context <ctx>`         | Switch to a different cluster             |

### Useful Shortcuts & Flags

| Flag / Trick                     | What it does                                 |
|----------------------------------|----------------------------------------------|
| `-o wide`                        | Show extra columns (node, IP)                |
| `-o yaml`                        | Output full YAML of a resource               |
| `-o json`                        | Output full JSON                             |
| `-n <namespace>`                 | Target a specific namespace                  |
| `--watch` / `-w`                 | Watch for changes in real-time               |
| `-l app=myapp`                   | Filter by label                              |
| `--dry-run=client -o yaml`      | Generate YAML without applying               |
| `kubectl explain pod.spec`       | Docs for any field in a resource spec        |
"""

# ─────────────────────────────────────────────────────────────────────────────
# OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "1. Pod — The Smallest Unit": {
        "description": "A Pod wraps one or more containers. Rarely created directly — usually via a Deployment.",
        "language": "yaml",
        "code": '''# pod.yaml — Basic Pod definition
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  labels:
    app: my-app
    env: dev
spec:
  containers:
    - name: app
      image: python:3.11-slim
      command: ["python", "-m", "http.server", "8000"]
      ports:
        - containerPort: 8000
      resources:
        requests:              # Minimum guaranteed
          memory: "64Mi"
          cpu: "100m"          # 100 millicores = 0.1 CPU
        limits:                # Maximum allowed
          memory: "128Mi"
          cpu: "250m"

# ── Usage ──
# kubectl apply -f pod.yaml
# kubectl get pods
# kubectl describe pod my-app
# kubectl logs my-app
# kubectl delete pod my-app
'''
    },

    "2. Deployment — Managed Replicas": {
        "description": "A Deployment manages a set of identical Pod replicas with rolling updates and self-healing",
        "language": "yaml",
        "code": '''# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web-app
spec:
  replicas: 3                        # Run 3 identical Pods
  selector:
    matchLabels:
      app: web-app                   # Must match template labels
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1                    # 1 extra Pod during update
      maxUnavailable: 0              # Never go below desired count
  template:                          # ← This IS the Pod spec
    metadata:
      labels:
        app: web-app
    spec:
      containers:
        - name: web
          image: myregistry/web-app:1.0.0
          ports:
            - containerPort: 8000
          env:
            - name: ENV
              value: "production"
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-secrets
                  key: password
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "500m"
          livenessProbe:             # Restart if unhealthy
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 15
          readinessProbe:            # Don't send traffic until ready
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5

# ── Usage ──
# kubectl apply -f deployment.yaml
# kubectl get deploy
# kubectl rollout status deploy/web-app
# kubectl scale deploy web-app --replicas=5
# kubectl set image deploy/web-app web=myregistry/web-app:2.0.0
# kubectl rollout undo deploy/web-app       # Rollback
'''
    },

    "3. Service — Stable Networking": {
        "description": "A Service gives Pods a stable IP and DNS name, with load balancing across replicas",
        "language": "yaml",
        "code": '''# ── ClusterIP (internal only, default) ──
apiVersion: v1
kind: Service
metadata:
  name: web-app-service
spec:
  type: ClusterIP
  selector:
    app: web-app              # Routes to Pods with this label
  ports:
    - port: 80                # Service port (what other pods use)
      targetPort: 8000        # Container port
      protocol: TCP

# Other Pods reach this at: web-app-service:80
# Or: web-app-service.<namespace>.svc.cluster.local:80

---
# ── NodePort (expose on every node's IP) ──
apiVersion: v1
kind: Service
metadata:
  name: web-app-nodeport
spec:
  type: NodePort
  selector:
    app: web-app
  ports:
    - port: 80
      targetPort: 8000
      nodePort: 30080         # Access at <NodeIP>:30080

---
# ── LoadBalancer (cloud provider LB) ──
apiVersion: v1
kind: Service
metadata:
  name: web-app-lb
spec:
  type: LoadBalancer
  selector:
    app: web-app
  ports:
    - port: 80
      targetPort: 8000
  # Cloud provider assigns an external IP automatically

# ── Usage ──
# kubectl apply -f service.yaml
# kubectl get svc
# kubectl describe svc web-app-service
'''
    },

    "4. ConfigMap & Secret": {
        "description": "Externalize configuration and sensitive data from your container images",
        "language": "yaml",
        "code": '''# ── ConfigMap — non-sensitive config ──
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  APP_ENV: "production"
  LOG_LEVEL: "info"
  MAX_WORKERS: "4"
  config.json: |
    {
      "feature_flags": {
        "new_ui": true,
        "beta_api": false
      }
    }

---
# ── Secret — sensitive data (base64 encoded) ──
apiVersion: v1
kind: Secret
metadata:
  name: db-secrets
type: Opaque
data:
  username: YWRtaW4=          # echo -n "admin" | base64
  password: cEBzc3cwcmQ=     # echo -n "p@ssw0rd" | base64

---
# ── Using them in a Deployment ──
# (add to container spec)
#
#   env:
#     - name: APP_ENV
#       valueFrom:
#         configMapKeyRef:
#           name: app-config
#           key: APP_ENV
#     - name: DB_PASSWORD
#       valueFrom:
#         secretKeyRef:
#           name: db-secrets
#           key: password
#
#   # Or mount as files:
#   volumeMounts:
#     - name: config-volume
#       mountPath: /app/config
#
#   volumes:
#     - name: config-volume
#       configMap:
#         name: app-config

# ── Create secret from CLI ──
# kubectl create secret generic db-secrets \\
#     --from-literal=username=admin \\
#     --from-literal=password=p@ssw0rd
'''
    },

    "5. Namespace — Environment Isolation": {
        "description": "Namespaces provide virtual sub-clusters for team/environment isolation",
        "language": "yaml",
        "code": '''# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: staging
  labels:
    env: staging

---
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    env: production

# ── Deploy to a specific namespace ──
# kubectl apply -f deployment.yaml -n staging
# kubectl apply -f deployment.yaml -n production

# ── Set default namespace for your context ──
# kubectl config set-context --current --namespace=staging

# ── List resources in a namespace ──
# kubectl get all -n staging
# kubectl get all -n production

# ── Resource Quota (limit namespace usage) ──
---
apiVersion: v1
kind: ResourceQuota
metadata:
  name: staging-quota
  namespace: staging
spec:
  hard:
    pods: "20"
    requests.cpu: "4"
    requests.memory: "8Gi"
    limits.cpu: "8"
    limits.memory: "16Gi"
'''
    },

    "6. Ingress — External HTTP Access": {
        "description": "Route external HTTP/HTTPS traffic to internal Services with path-based or host-based rules",
        "language": "yaml",
        "code": '''# ingress.yaml
# Requires an Ingress Controller (nginx, traefik, etc.)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - myapp.example.com
      secretName: myapp-tls
  rules:
    # Host-based routing
    - host: myapp.example.com
      http:
        paths:
          # Path-based routing
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend-service
                port:
                  number: 80
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: backend-service
                port:
                  number: 80

# ── Traffic flow ──
#
#   Internet
#       │
#       ▼
#   Load Balancer (cloud)
#       │
#       ▼
#   Ingress Controller (nginx pod)
#       │
#       ├── myapp.example.com/     → frontend-service:80
#       └── myapp.example.com/api  → backend-service:80
'''
    },

    "7. Full Stack Example — Deploy a Web App": {
        "description": "Complete YAML to deploy an app with database, service, and ingress",
        "language": "bash",
        "code": '''# ── Step 1: Create namespace ──
kubectl create namespace myapp

# ── Step 2: Create secrets ──
kubectl create secret generic db-creds \\
    --from-literal=POSTGRES_USER=app \\
    --from-literal=POSTGRES_PASSWORD=secretpass \\
    --from-literal=POSTGRES_DB=myappdb \\
    -n myapp

# ── Step 3: Apply all manifests ──
kubectl apply -f k8s/ -n myapp
# (assuming k8s/ folder has deployment.yaml, service.yaml, ingress.yaml)

# ── Step 4: Verify everything ──
kubectl get all -n myapp
kubectl get ingress -n myapp

# ── Step 5: Watch rollout ──
kubectl rollout status deploy/web-app -n myapp

# ── Step 6: Check logs ──
kubectl logs -f deploy/web-app -n myapp

# ── Step 7: Debug if needed ──
kubectl describe pod -l app=web-app -n myapp
kubectl exec -it deploy/web-app -n myapp -- sh

# ── Step 8: Scale up ──
kubectl scale deploy web-app --replicas=5 -n myapp

# ── Step 9: Update image (rolling update) ──
kubectl set image deploy/web-app \\
    web=myregistry/web-app:2.0.0 -n myapp

# ── Step 10: Rollback if broken ──
kubectl rollout undo deploy/web-app -n myapp
'''
    },

    "8. Local Development — Minikube / kind": {
        "description": "Set up a local K8s cluster for development and testing",
        "language": "bash",
        "code": '''# ════════════════════════════════════════
# Option A: Minikube (VM or container)
# ════════════════════════════════════════

# Install (macOS)
brew install minikube

# Start cluster
minikube start --cpus=4 --memory=8g --driver=docker

# Enable useful addons
minikube addons enable ingress
minikube addons enable dashboard
minikube addons enable metrics-server

# Open dashboard
minikube dashboard

# Use minikube's Docker daemon (build images directly)
eval $(minikube docker-env)
docker build -t myapp:local .

# Access a LoadBalancer service
minikube tunnel    # Runs in background, assigns external IPs

# Stop / Delete
minikube stop
minikube delete


# ════════════════════════════════════════
# Option B: kind (K8s IN Docker)
# ════════════════════════════════════════

# Install (macOS)
brew install kind

# Create cluster
kind create cluster --name dev

# Load local image into kind
kind load docker-image myapp:local --name dev

# Delete cluster
kind delete cluster --name dev


# ════════════════════════════════════════
# Verify either setup
# ════════════════════════════════════════
kubectl cluster-info
kubectl get nodes
kubectl get pods -A
'''
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def get_content():
    """Return all content for this tutorial module."""
    return {
        "theory": THEORY,
        "commands": COMMANDS,
        "operations": OPERATIONS,
        "category": CATEGORY,
    }
