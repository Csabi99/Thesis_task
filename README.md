# Federated Learning Infrastructure - Thesis Task

This repository contains the necessary configuration files, aggregation strategies, and deployment scripts to launch a federated learning experiment using Kubernetes (`k3s`) and Docker. The setup is designed to run distributed learning jobs across a cluster of nodes with centralized logging and orchestration.

---

## 🚀 Quick Start

### Step 1: Install `k3s` on Every Node

You must install [k3s](https://k3s.io/) on each node in your cluster.

---

### Step 2: Define the Experiment Configuration

Use or create a configuration directory under:

```
/home/ubuntu/data/strategy_configs/
```

Each config directory should contain:
- Parameters of the federated learning process (`*.yaml`)
- Client job generator script (`generator.sh`)

Example:
```bash
/home/ubuntu/data/strategy_configs/FedAvg/
```

---

### Step 3: Prepare the Environment

Copy the relevant scripts, Dockerfiles, strategies, and config files into the target node:

```bash
cp -r ./dockerfiles ./strategy_configs ./setup_cifar.sh /home/ubuntu/data/
```

Ensure all contents are accessible under `/home/ubuntu/data`.

---

### Step 4: Launch the Server

Run the preparation script to:
- Build Docker images
- Push them to Docker Hub
- Apply PersistentVolumeClaims for logs
- Launch the server job in Kubernetes

```bash
/home/ubuntu/data/setup_cifar.sh /home/ubuntu/data/strategy_configs/FedAvg fedavg
```

You should see:
- Images built and pushed
- `fedavg-server-job.yaml` applied
- Server pod starting in namespace `fedavg`

---

### Step 5: Launch the Clients

Generate the client job YAMLs:

```bash
cd /home/ubuntu/data/strategy_configs/FedAvg
./generator.sh
```

Then apply them to the cluster:

```bash
for i in {1..32}; do
  sudo kubectl apply -f client-job-$i.yaml -n fedavg
done
```

This will spin up 32 client jobs running under the `fedavg` namespace.

---

## 📝 Notes

- Docker must be configured and logged into Docker Hub to push images.
- You can monitor job logs using:

```bash
kubectl logs -n fedcm job/fedcm-server
kubectl logs -n fedcm job/client-job-<id>
```

- Adjust the number of clients (`1..32`) based on your experiment needs.
- Make sure persistent volumes are correctly set up for log storage.

---

