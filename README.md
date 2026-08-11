# LIGHTYEAR: NTK-Guided Personalized Aggregation for Robust Federated Learning

Official implementation of our BMVC 2026 paper:

> **Beyond Parameter Space: NTK-Guided Personalized Aggregation for Robust Federated Learning**  

**LIGHTYEAR** stands for **Local Inference Guided Aggregation for Heterogeneous Training Environments to Yield Enhancement Through Agreement and Regularization**.

LIGHTYEAR is a decentralized peer-to-peer (P2P) federated learning framework for robust and personalized aggregation under heterogeneous data distributions and malfunctioning clients. Instead of deciding which client updates to aggregate using parameter-space distances, LIGHTYEAR evaluates neighboring models in **function space** using a Neural Tangent Kernel (NTK)-based agreement score computed on each client's private validation data.

---

## Overview

Federated learning systems commonly aggregate model updates using information available in parameter space, such as parameter or gradient distances. Under strong client heterogeneity, however, similar parameter values do not necessarily imply similar predictive behavior.

LIGHTYEAR addresses this limitation by allowing each client to:

1. train a local model on its private data;
2. receive model updates from neighboring clients in a P2P topology;
3. evaluate the predictive behavior of incoming models on its local validation data;
4. compute an NTK-based agreement score between its local model and each received model;
5. select only sufficiently aligned updates; and
6. aggregate the selected models using a round-dependent regularized aggregation rule.

The result is a personalized aggregation set for every client, designed to reject harmful or irrelevant updates while retaining updates that are useful for the client's target domain.

---

## Repository Structure

```text
LIGHTYEAR/
├── datasets/          # Dataset loading 
├── networks/          # Model definitions
├── training/          # Local training
├── training/          # Local optimization 
├── aggregation/       # FL aggregation methods
├── executors/         # P2P aggregation methods
├── configs/           # Experiment entry configurations
├── scripts/           # Convenience scripts
├── requirements.txt
└── README.md
```

## Method

### Function-space agreement

For a client model with parameters ($\theta$), LIGHTYEAR computes the Jacobian of the network output with respect to the parameters of the final layer. The resulting empirical final-layer Neural Tangent Kernel captures local predictive sensitivities on the client's validation samples.

The kernel matrix is centered and Frobenius-normalized. Agreement between a local model ($\theta_i$) and a neighboring model ($theta_j$) is then measured using normalized kernel alignment:

$$
A(\theta_i, \theta_j; V_i)
= \langle \tilde{K}_{\theta_i}(V_i),
          \tilde{K}_{\theta_j}(V_i) \rangle_F.
$$

This lets each client compare models based on their behavior on the local target domain rather than only on their parameter values.

### Personalized update selection

A neighboring update is selected when its agreement score exceeds a threshold \(\tau\):

$$
S_i = \{\theta_j \in \mathcal{N}(i) \mid
A(\theta_i, \theta_j; V_i) \geq \tau\}.
$$

The paper uses:

```text
agreement threshold tau = 0.6
```

### Regularized aggregation

After update selection, LIGHTYEAR aggregates the accepted updates using a round-dependent regulation term. This reduces the influence of incoming updates as training progresses and helps mitigate client drift under heterogeneous data distributions.

The paper uses:

```text
regularization parameter gamma = 0.95
```

Setting the regulation parameter to 1 recovers standard averaging over the selected aggregation set.

---

## Datasets

The paper evaluates LIGHTYEAR on five heterogeneous federated learning benchmarks spanning classification and segmentation.

| Dataset | Task | Clients | Input / preprocessing | Model |
|---|---|---:|---|---|
| FEMNIST | Multi-class classification | 8 | 28x28 grayscale images | Two-layer CNN |
| Camelyon17-WILDS | Binary classification | 5 | 98x98 tissue patches; one hospital per client | DenseNet121 |
| ISIC19 | Multi-class classification | 6 | Dermoscopic images resized to 224x224; data from six medical centers | EfficientNet |
| Fetal Abdominal Structures (Ultrasound) | Binary segmentation | 5 | Images resized to 64x64 | TransUNet |
| ChestXray | Segmentation | 5 | 200-sample subset; images resized to 64x64 | TransUNet |

Each client maintains its own training, validation, and test split from its local data.

---

## Client Malfunctions

LIGHTYEAR is evaluated against three types of malfunctioning client updates.

### Additive-Noise Attack (ANA)

A client corrupts the transmitted model by adding Gaussian noise to its parameters:

$$
\tilde{\theta} = \theta + \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, \sigma^2 I).
$$

### Sign-Flipping Attack (SFA)

A client reverses the model update direction by multiplying the transmitted parameters by a negative factor:

$$
\tilde{\theta} = -\alpha\theta, \qquad \alpha > 0.
$$

### Random model update

To simulate technical failures such as broken training or data pipelines, a malfunctioning client can broadcast randomly initialized model weights instead of a locally optimized model.

---

## Baselines

The paper compares LIGHTYEAR with the following federated learning methods:

- FedAvg
- AFA
- ASMR
- CFL
- Ditto
- Krum
- FedProx
- BALANCE
- SCCLIP

BALANCE and SCCLIP are evaluated as P2P approaches; the remaining comparison methods use centralized FL settings in the experiments described in the paper.

---

## Experimental Scenarios

The evaluation covers three main robustness settings.

### 1. Individual malfunction types

Each method is evaluated separately under:

- Additive-Noise Attack (ANA)
- Sign-Flipping Attack (SFA)
- Random model updates

The number of malfunctioning clients is increased from one client up to ($n-1$) clients.

### 2. Increasing numbers of malfunctioning clients

The experiments compare centralized FL, standard P2P FL, and LIGHTYEAR as the proportion of malfunctioning clients increases.

### 3. Dynamic malfunctions

In the dynamic setting, every malfunctioning client randomly chooses one of the three malfunction types in each communication round. This experiment evaluates robustness when client failures change over time.

---

## Evaluation Metrics

The paper reports:

- **Accuracy** for classification datasets: FEMNIST, Camelyon17, and ISIC19.
- **Dice score** for segmentation datasets: Ultrasound and ChestXray.

Scores are evaluated on the clients' local test data and aggregated across clients/runs according to the experiment.

---

## Installation

```bash
# Clone the repository
# git clone https://github.com/MECLabTUDA/LIGHTYEAR.git
# cd LIGHTYEAR
# Create an environment
# conda create -n lightyear python=3.10.20
# conda activate lightyear

# Install dependencies
# pip install -r requirements.txt
```

---

## Running the Experiments

The following commands are intentionally provided as **placeholders**. Replace the script names and arguments with the corresponding commands in this repository.

### LIGHTYEAR (P2P)

```bash
python3 main.py --gpu 1 --workdir LIGHTYEAR/mix_2 --dataset camelyon17 --clients 5 --malf sfa ana random --malclients 1 2 --algo LIGHTYEAR

```

### Centralized FL

```bash
python3 main_fl.py --gpu 1 --workdir fedavg_run1 --dataset camelyon17 --clients 5 --malf sfa ana random --malclients 1 2 --algo FedAvg

```


---

## Evaluation

```bash
# Evaluate a trained model/checkpoint
python3 eval.py --algo LIGHTYEAR --topology p2p --dataset camelyon17 --malfunction mix >> results/LIGHTYEAR_mix.txt
```

---

## Reproducing the Paper Experiments

A complete reproduction should cover the following combinations:

1. all five datasets;
2. LIGHTYEAR and all nine comparison methods;
3. ANA, SFA, and random-update malfunctions;
4. increasing numbers of malfunctioning clients;
5. dynamically changing malfunctions;
6. centralized FL, standard P2P FL, and LIGHTYEAR topology comparisons; and
7. the ($\gamma$) and agreement-score analyses reported in the ablation study.

For LIGHTYEAR, the main paper configuration is:

```yaml
method: LIGHTYEAR
topology: P2P
tau: 0.6
gamma: 0.95
```

---

## Citation

If you use this code in your research, please cite the paper.

> **TODO:** Replace this placeholder with the final BibTeX entry once publication metadata is available.

```bibtex
@inproceedings{lightyear2026,
  title     = {Beyond Parameter Space: NTK-Guided Personalized Aggregation for Robust Federated Learning},
  author    = {<AUTHORS>},
  booktitle = {British Machine Vision Conference (BMVC)},
  year      = {2026}
}
```
