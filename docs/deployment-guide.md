# Deployment Guide

> Comprehensive deployment guide for the Metis platform covering local development, staging, and production environments.

## 📋 Overview

This guide covers the complete deployment strategy for the Metis platform, from local development setup to production deployment. The platform uses a containerized microservices architecture with Kubernetes orchestration.

**Deployment Environments:**
- **Local Development**: Docker Compose for rapid development
- **Staging**: Kubernetes cluster for testing and validation
- **Production**: Multi-region Kubernetes deployment with high availability

---

## 🏗️ Infrastructure Architecture

### **Deployment Topology**
```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                        │
├─────────────────────────────────────────────────────────────────┤
│  🌐 LOAD BALANCER (AWS ALB)                                   │
│  ├─ SSL Termination                                            │
│  ├─ Health Checks                                              │
│  └─ Traffic Distribution                                       │
├─────────────────────────────────────────────────────────────────┤
│  ☸️ KUBERNETES CLUSTER (EKS)                                  │
│  ├─ Core App Services (Go)                                     │
│  ├─ ML Services (Python/FastAPI)                               │
│  ├─ Temporal Workers                                           │
│  └─ Supporting Services                                        │
├─────────────────────────────────────────────────────────────────┤
│  💾 DATA LAYER                                                │
│  ├─ RDS PostgreSQL (Multi-AZ)                                  │
│  ├─ ElastiCache Redis (Cluster Mode)                           │
│  ├─ S3 (Multi-region replication)                              │
│  └─ RabbitMQ (Clustered)                                       │
├─────────────────────────────────────────────────────────────────┤
│  📊 MONITORING & LOGGING                                      │
│  ├─ Prometheus + Grafana                                       │
│  ├─ ELK Stack (Elasticsearch, Logstash, Kibana)               │
│  └─ AWS CloudWatch                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🐳 Local Development Setup

### **Prerequisites**
```bash
# Required tools
- Docker Desktop 4.20+
- Docker Compose 2.20+
- Go 1.21+
- Python 3.11+
- Node.js 18+ (for frontend)
- kubectl 1.28+
- Helm 3.12+
```

### **Environment Setup**
```bash
# Clone repository
git clone https://github.com/metis/platform.git
cd platform

# Copy environment template
cp .env.example .env.local

# Edit environment variables
vim .env.local
```

### **Environment Variables**
```bash
# .env.local
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=metis_dev
POSTGRES_USER=metis
POSTGRES_PASSWORD=dev_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# S3 Configuration (LocalStack)
AWS_ENDPOINT_URL=http://localhost:4566
AWS_ACCESS_KEY_ID=test
AWS_SECRET_ACCESS_KEY=test
AWS_REGION=us-east-1
S3_BUCKET=metis-dev-bucket

# RabbitMQ Configuration
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=metis
RABBITMQ_PASSWORD=dev_password

# Temporal Configuration
TEMPORAL_HOST=localhost
TEMPORAL_PORT=7233

# ML Service Configuration
ML_SERVICE_URL=http://localhost:8001
MODEL_STORAGE_PATH=/app/models

# Application Configuration
APP_ENV=development
LOG_LEVEL=debug
JWT_SECRET=dev_jwt_secret_key
ENCRYPTION_KEY=dev_encryption_key_32_chars
```

### **Docker Compose Setup**
```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  # Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: metis_dev
      POSTGRES_USER: metis
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init.sql

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # LocalStack (AWS Services)
  localstack:
    image: localstack/localstack:latest
    environment:
      SERVICES: s3,sqs,sns
      DEBUG: 1
      DATA_DIR: /tmp/localstack/data
    ports:
      - "4566:4566"
    volumes:
      - localstack_data:/tmp/localstack

  # RabbitMQ
  rabbitmq:
    image: rabbitmq:3-management
    environment:
      RABBITMQ_DEFAULT_USER: metis
      RABBITMQ_DEFAULT_PASS: dev_password
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

  # Temporal
  temporal:
    image: temporalio/auto-setup:latest
    environment:
      DB: postgresql
      DB_PORT: 5432
      POSTGRES_USER: metis
      POSTGRES_PWD: dev_password
      POSTGRES_SEEDS: postgres
    ports:
      - "7233:7233"
      - "8233:8233"
    depends_on:
      - postgres

  # Core App Service
  core-app:
    build:
      context: .
      dockerfile: cmd/core/Dockerfile.dev
    environment:
      - DATABASE_URL=postgres://metis:dev_password@postgres:5432/metis_dev
      - REDIS_URL=redis://redis:6379
      - RABBITMQ_URL=amqp://metis:dev_password@rabbitmq:5672/
    ports:
      - "8080:8080"
    volumes:
      - ./cmd/core:/app/cmd/core
      - ./pkg:/app/pkg
    depends_on:
      - postgres
      - redis
      - rabbitmq

  # ML Service
  ml-service:
    build:
      context: .
      dockerfile: cmd/ml/Dockerfile.dev
    environment:
      - DATABASE_URL=postgres://metis:dev_password@postgres:5432/metis_dev
      - REDIS_URL=redis://redis:6379
      - MODEL_STORAGE_PATH=/app/models
    ports:
      - "8001:8000"
    volumes:
      - ./cmd/ml:/app/cmd/ml
      - ./models:/app/models
    depends_on:
      - postgres
      - redis

volumes:
  postgres_data:
  redis_data:
  localstack_data:
  rabbitmq_data:
```

### **Development Commands**
```bash
# Start all services
make dev-up

# Stop all services
make dev-down

# View logs
make dev-logs

# Run database migrations
make dev-migrate

# Seed development data
make dev-seed

# Run tests
make test

# Build and run specific service
make dev-build-core
make dev-run-core
```

### **Makefile**
```makefile
# Makefile
.PHONY: dev-up dev-down dev-logs dev-migrate dev-seed test

# Development environment
dev-up:
	docker-compose -f docker-compose.dev.yml up -d
	@echo "Development environment started"
	@echo "Core App: http://localhost:8080"
	@echo "ML Service: http://localhost:8001"
	@echo "RabbitMQ Management: http://localhost:15672"

dev-down:
	docker-compose -f docker-compose.dev.yml down

dev-logs:
	docker-compose -f docker-compose.dev.yml logs -f

dev-migrate:
	docker-compose -f docker-compose.dev.yml exec core-app go run cmd/migrate/main.go

dev-seed:
	docker-compose -f docker-compose.dev.yml exec core-app go run cmd/seed/main.go

# Testing
test:
	go test ./...
	cd cmd/ml && python -m pytest

test-integration:
	docker-compose -f docker-compose.test.yml up --abort-on-container-exit

# Building
build-core:
	docker build -f cmd/core/Dockerfile -t metis/core-app:latest .

build-ml:
	docker build -f cmd/ml/Dockerfile -t metis/ml-service:latest .

build-all: build-core build-ml

# Linting and formatting
lint:
	golangci-lint run
	cd cmd/ml && black . && isort . && flake8

format:
	go fmt ./...
	cd cmd/ml && black . && isort .
```

---

## ☸️ Kubernetes Deployment

### **Cluster Setup**

#### **EKS Cluster Configuration**
```yaml
# cluster-config.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: metis-production
  region: us-west-2
  version: "1.28"

nodeGroups:
  - name: core-services
    instanceType: m5.large
    minSize: 2
    maxSize: 10
    desiredCapacity: 3
    volumeSize: 50
    ssh:
      allow: false
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        ebs: true
        efs: true
        albIngress: true

  - name: ml-services
    instanceType: m5.xlarge
    minSize: 1
    maxSize: 5
    desiredCapacity: 2
    volumeSize: 100
    ssh:
      allow: false
    labels:
      workload-type: ml-intensive

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver

cloudWatch:
  clusterLogging:
    enableTypes: ["*"]
```

#### **Create Cluster**
```bash
# Create EKS cluster
eksctl create cluster -f cluster-config.yaml

# Install AWS Load Balancer Controller
kubectl apply -k "github.com/aws/eks-charts/stable/aws-load-balancer-controller//crds?ref=master"

helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=metis-production \
  --set serviceAccount.create=false \
  --set serviceAccount.name=aws-load-balancer-controller
```

### **Namespace Configuration**
```yaml
# namespaces.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: metis-production
  labels:
    name: metis-production
---
apiVersion: v1
kind: Namespace
metadata:
  name: metis-staging
  labels:
    name: metis-staging
---
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
  labels:
    name: monitoring
```

### **ConfigMaps and Secrets**

#### **Application Configuration**
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: metis-config
  namespace: metis-production
data:
  APP_ENV: "production"
  LOG_LEVEL: "info"
  POSTGRES_HOST: "metis-postgres.cluster-xyz.us-west-2.rds.amazonaws.com"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "metis_production"
  REDIS_HOST: "metis-redis.cache.amazonaws.com"
  REDIS_PORT: "6379"
  RABBITMQ_HOST: "metis-rabbitmq.mq.us-west-2.amazonaws.com"
  RABBITMQ_PORT: "5672"
  TEMPORAL_HOST: "temporal-frontend.metis-production.svc.cluster.local"
  TEMPORAL_PORT: "7233"
  S3_BUCKET: "metis-production-storage"
  AWS_REGION: "us-west-2"
```

#### **Secrets Management**
```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: metis-secrets
  namespace: metis-production
type: Opaque
data:
  POSTGRES_PASSWORD: <base64-encoded-password>
  REDIS_PASSWORD: <base64-encoded-password>
  RABBITMQ_PASSWORD: <base64-encoded-password>
  JWT_SECRET: <base64-encoded-jwt-secret>
  ENCRYPTION_KEY: <base64-encoded-encryption-key>
  AWS_ACCESS_KEY_ID: <base64-encoded-access-key>
  AWS_SECRET_ACCESS_KEY: <base64-encoded-secret-key>
```

### **Application Deployments**

#### **Core App Service**
```yaml
# core-app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: core-app
  namespace: metis-production
  labels:
    app: core-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: core-app
  template:
    metadata:
      labels:
        app: core-app
    spec:
      containers:
      - name: core-app
        image: metis/core-app:v1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          value: "postgres://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@$(POSTGRES_HOST):$(POSTGRES_PORT)/$(POSTGRES_DB)"
        envFrom:
        - configMapRef:
            name: metis-config
        - secretRef:
            name: metis-secrets
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: core-app-service
  namespace: metis-production
spec:
  selector:
    app: core-app
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

#### **ML Service**
```yaml
# ml-service-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: metis-production
  labels:
    app: ml-service
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      nodeSelector:
        workload-type: ml-intensive
      containers:
      - name: ml-service
        image: metis/ml-service:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_STORAGE_PATH
          value: "/app/models"
        envFrom:
        - configMapRef:
            name: metis-config
        - secretRef:
            name: metis-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: ml-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: ml-service
  namespace: metis-production
spec:
  selector:
    app: ml-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### **Ingress Configuration**
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: metis-ingress
  namespace: metis-production
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:us-west-2:123456789:certificate/abc123
    alb.ingress.kubernetes.io/ssl-policy: ELBSecurityPolicy-TLS-1-2-2017-01
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: '443'
spec:
  rules:
  - host: api.metis.com
    http:
      paths:
      - path: /api/v1/ml
        pathType: Prefix
        backend:
          service:
            name: ml-service
            port:
              number: 80
      - path: /
        pathType: Prefix
        backend:
          service:
            name: core-app-service
            port:
              number: 80
```

---

## 🚀 CI/CD Pipeline

### **GitHub Actions Workflow**
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  AWS_REGION: us-west-2
  EKS_CLUSTER_NAME: metis-production

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Go
      uses: actions/setup-go@v3
      with:
        go-version: 1.21

    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: 3.11

    - name: Run Go tests
      run: |
        go mod download
        go test ./...

    - name: Run Python tests
      run: |
        cd cmd/ml
        pip install -r requirements.txt
        python -m pytest

    - name: Run linting
      run: |
        golangci-lint run
        cd cmd/ml && black --check . && isort --check . && flake8

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}

    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1

    - name: Build and push Core App image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: metis/core-app
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -f cmd/core/Dockerfile -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

    - name: Build and push ML Service image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: metis/ml-service
        IMAGE_TAG: ${{ github.sha }}
      run: |
        docker build -f cmd/ml/Dockerfile -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}

    - name: Update kubeconfig
      run: |
        aws eks update-kubeconfig --region ${{ env.AWS_REGION }} --name ${{ env.EKS_CLUSTER_NAME }}

    - name: Deploy to Kubernetes
      env:
        IMAGE_TAG: ${{ github.sha }}
      run: |
        # Update image tags in deployment files
        sed -i "s|metis/core-app:.*|metis/core-app:$IMAGE_TAG|g" k8s/core-app-deployment.yaml
        sed -i "s|metis/ml-service:.*|metis/ml-service:$IMAGE_TAG|g" k8s/ml-service-deployment.yaml

        # Apply configurations
        kubectl apply -f k8s/

        # Wait for rollout to complete
        kubectl rollout status deployment/core-app -n metis-production
        kubectl rollout status deployment/ml-service -n metis-production
```

---

## 📊 Monitoring & Logging

### **Prometheus Configuration**
```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    rule_files:
      - "metis_rules.yml"

    scrape_configs:
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)

      - job_name: 'metis-core-app'
        static_configs:
          - targets: ['core-app-service.metis-production.svc.cluster.local:80']
        metrics_path: /metrics

      - job_name: 'metis-ml-service'
        static_configs:
          - targets: ['ml-service.metis-production.svc.cluster.local:80']
        metrics_path: /metrics

    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager.monitoring.svc.cluster.local:9093
```

### **Grafana Dashboards**
```json
{
  "dashboard": {
    "title": "Metis Platform Overview",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{service}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "5xx errors"
          }
        ]
      }
    ]
  }
}
```

---

## 🔒 Security Considerations

### **Network Security**
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: metis-network-policy
  namespace: metis-production
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: metis-production
    - podSelector: {}
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: metis-production
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### **Pod Security Standards**
```yaml
# pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: metis-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

---

## 🚨 Disaster Recovery

### **Backup Strategy**
```bash
#!/bin/bash
# backup-script.sh

# Database backup
kubectl exec -n metis-production deployment/postgres -- pg_dump -U metis metis_production > backup-$(date +%Y%m%d).sql

# Upload to S3
aws s3 cp backup-$(date +%Y%m%d).sql s3://metis-backups/database/

# Kubernetes configuration backup
kubectl get all -n metis-production -o yaml > k8s-backup-$(date +%Y%m%d).yaml
aws s3 cp k8s-backup-$(date +%Y%m%d).yaml s3://metis-backups/k8s/

# Clean up local files
rm backup-$(date +%Y%m%d).sql k8s-backup-$(date +%Y%m%d).yaml
```

### **Recovery Procedures**
```bash
#!/bin/bash
# recovery-script.sh

# Restore database
aws s3 cp s3://metis-backups/database/backup-20240101.sql .
kubectl exec -n metis-production deployment/postgres -- psql -U metis -d metis_production < backup-20240101.sql

# Restore Kubernetes resources
aws s3 cp s3://metis-backups/k8s/k8s-backup-20240101.yaml .
kubectl apply -f k8s-backup-20240101.yaml
```

---

## 📋 Deployment Checklist

### **Pre-deployment**
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance testing completed
- [ ] Database migrations tested
- [ ] Backup procedures verified
- [ ] Monitoring alerts configured

### **Deployment**
- [ ] Blue-green deployment strategy
- [ ] Health checks passing
- [ ] Database migrations applied
- [ ] Configuration updated
- [ ] SSL certificates valid
- [ ] Load balancer configured

### **Post-deployment**
- [ ] Application health verified
- [ ] Monitoring dashboards updated
- [ ] Performance metrics baseline
- [ ] Error rates within acceptable limits
- [ ] User acceptance testing
- [ ] Documentation updated

---

*This deployment guide ensures reliable, secure, and scalable deployment of the Metis platform across all environments.*
