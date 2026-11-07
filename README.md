# Hierarchical Two-Stage Detection of Volumetric and Low-Rate DDoS Attacks in Software-Defined Networks

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Platform: Ubuntu](https://img.shields.io/badge/Platform-Ubuntu%2020.04%20%7C%2022.04-orange)
![Python](https://img.shields.io/badge/Python-3.8--3.10-green)

A lab framework for detecting **DDoS attacks** in a **Software-Defined Network (SDN)**. The network is emulated with **Mininet** and **Open vSwitch**, the switches are controlled by a **Ryu** controller, and attack traffic is classified by a **hierarchical two-stage detector**:

- **Stage 1** catches **volumetric** attacks, interval by interval.
- **Stage 2** catches **low-rate** attacks, using a **temporal sliding window of 10 intervals**.

## Contents

- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Controller Settings](#controller-settings)
- [Evaluation](#evaluation)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Responsible Use](#responsible-use)
- [Citation](#citation)
- [License](#license)

## How It Works

```
 Mininet hosts ──► Open vSwitch ◄──── OpenFlow ────► Ryu controller (app1.py)
 (normal + attack                                      │
  traffic)                                             ▼
                                         flow statistics, polled every INTERVAL s
                                                       │
                                                       ▼
                                   feature extraction: FER  SIGS  FSR  PLV  BLV
                                                       │
                                                       ▼
                          ┌─────────────── STAGE 1 : volumetric ───────────────┐
                          │ classifies the CURRENT interval                    │
                          └───────┬─────────────────────────────────┬──────────┘
                          volumetric                          not volumetric
                                  │                                 │
                      alert + clear the window                      ▼
                                  │             ┌───────── STAGE 2 : low-rate ─────────┐
                                  │             │ sliding window of the last 10        │
                                  │             │ non-volumetric intervals (stride 1)  │
                                  │             └──────────┬──────────────────┬────────┘
                                  │                    low-rate             normal
                                  ▼                        ▼                  ▼
                            VOLUMETRIC attack       LOW-RATE attack       NORMAL traffic
                                  └──────── optional mitigation (PREVENTION = 1) ────────┘
```

1. **Topology** (`topo5.py`): Mininet builds the virtual network and connects its switch(es) to the remote Ryu controller.
2. **Controller** (`app1.py`): the Ryu application manages the switches, polls their flow tables every `INTERVAL` seconds (default 3 s), extracts the features and runs the two-stage detector.
3. **Detector** (`hierarchical_ml.py`): holds the Stage 1 and Stage 2 models and the sliding window. The same code is used online by the controller and offline by the trainer, so the evaluation exercises the real decision logic.
4. **Training** (`train_hierarchical.py`): builds both models from the CSV files that the controller writes in data-collection mode.

### Features (per switch, per interval)

| Feature | Meaning |
|---------|---------|
| **FER** | Flow Entry Rate: change in the number of flow entries since the previous interval |
| **SIGS** | Source-IP Generation Speed: change in the number of distinct source IPs |
| **FSR** | Flow Success Ratio: share of flows that have a matching reverse flow (two-way traffic) |
| **PLV** | Packet Load Variance: standard deviation of the per-flow packet counts in the interval |
| **BLV** | Byte Load Variance: standard deviation of the per-flow byte counts in the interval |

The first interval after a switch connects only sets the baseline for the "change" features, so no decision is made for it.

### Stage 1: volumetric detection

Stage 1 sees the five features of **one** interval and answers *volumetric / not volumetric*. It is trained as *volumetric versus everything else* (normal and low-rate), so a low-rate attack passes through it. Default classifier: Random Forest.

### Stage 2: low-rate detection (sliding window of 10)

Low-rate attacks stay close to normal in any single interval, so Stage 2 looks at **how the traffic evolves over time**:

- The window holds the last **10** intervals (30 s with `INTERVAL = 3`) and advances **one interval at a time**, so from the 10th interval onward there is a decision at every interval.
- The window is summarised into **25 values**: mean, standard deviation, minimum, maximum and slope (trend per interval) of each of the 5 features.
- Only intervals that **Stage 1 passed as non-volumetric** enter the window. A volumetric detection **clears** the window, so Stage 2 only sees the kind of data it was trained on. After a volumetric alert, Stage 2 needs 10 new intervals before it can decide again.

Default classifier: K-Nearest Neighbours (scaled features, distance-weighted, `k = 5`).

### Mitigation (optional)

With `PREVENTION = 1`, while an alert from **either** stage is active, packets whose source IP was never validated by ARP on their ingress port are dropped and that port is blocked for 300 s (an `OFPFC_DELETE` of the port's flows followed by a drop rule). An alert is cleared after 3 consecutive normal intervals.

> This protects against spoofed sources. It does not stop a low-rate attacker that uses a valid, ARP-validated source address.

## Repository Structure

```
.
├── app1.py                    # Ryu controller: feature extraction, two-stage detection, mitigation
├── hierarchical_ml.py         # Stage 1 / Stage 2 models, sliding window, HierarchicalDetector
├── train_hierarchical.py      # trains both stages from the collected CSV files
├── topo5.py                   # Mininet topology
├── models/                    # created by train_hierarchical.py
│   ├── stage1_volumetric.joblib
│   ├── stage2_lowrate.joblib
│   ├── metadata.json          # window size, feature names, model choices
│   └── metrics.json           # per-stage and end-to-end evaluation
├── accuracy_score_Hybrid.py   # evaluation scripts of the earlier single-stage version (see Evaluation)
├── confusion_matrix.py
├── detection_rate.py
├── requirements.txt           # Python dependencies
├── LICENSE                    # GPL-3.0 license text
└── README.md
```

> Adjust this tree to match your repository exactly (add your dataset folder and any other files).

## Requirements

| Component | Version / Notes |
|-----------|-----------------|
| OS | Ubuntu 20.04 or 22.04 (recommended) |
| Python | 3.8 – 3.10 (Ryu may not run on Python 3.11 or newer) |
| Mininet | from the Ubuntu repositories |
| Open vSwitch | from the Ubuntu repositories |
| Ryu | installed with `pip` (see below) |
| hping3 | used only to generate attack traffic |

Python libraries: `numpy` and `scikit-learn` (which brings `joblib`) are needed by the **controller and the trainer**, so they must be installed in the **same virtual environment as Ryu**. `pandas` and `matplotlib` are only used by the earlier evaluation scripts.

> Train and run in the **same scikit-learn version**: the saved model files are version-sensitive.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install Mininet, Open vSwitch and hping3

```bash
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch hping3
sudo systemctl enable --now openvswitch-switch
```

### 3. Create a Python virtual environment

Using a virtual environment avoids conflicts with system packages (newer Ubuntu releases refuse system-wide `pip install`).

```bash
sudo apt-get install -y python3-venv python3-pip
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Ryu and the Python libraries

```bash
pip install --upgrade pip
pip install ryu eventlet==0.30.2
pip install -r requirements.txt
```

`eventlet==0.30.2` is pinned on purpose: newer versions break Ryu (see [Troubleshooting](#troubleshooting)).

Example `requirements.txt` (add anything else your scripts import):

```
numpy
scikit-learn
joblib
pandas
matplotlib
```

## Usage

The workflow has three phases: **collect** labelled data, **train** the two stages, **detect**. Use two terminals in the repository folder (a third is handy for watching the CSV files). In every phase start the Ryu controller **first**, then Mininet.

All CSV files are written to the folder in which `ryu-manager` is started.

### Phase 1: Collect labelled data

The controller needs no models in this phase. Open `app1.py` and set, at the top:

```python
APP_TYPE = 0      # data collection
TEST_TYPE = 0     # label of THIS run: 0 normal, 1 volumetric, 2 low-rate
```

Repeat the steps below once per scenario, changing `TEST_TYPE` each time (`0`, `1`, `2`), and restart both the controller and the topology for every run.

**Terminal 1: controller**

```bash
source venv/bin/activate
ryu-manager app1.py
```

Leave it running. By default Ryu listens for switches on port `6653`.

**Terminal 2: topology**

```bash
sudo python3 topo5.py
```

You will get the `mininet>` prompt. Check that the network works:

```
mininet> pingall
```

**Generate the traffic for this run (inside the `mininet>` prompt).** Traffic must be generated **from a Mininet host**, not from your own machine, so it travels through the emulated switches and the controller can see it. Replace `h1`, `h2` and `h3` with the host names defined in `topo5.py`.

*`TEST_TYPE = 0`: normal traffic*

```
mininet> h2 python3 -m http.server 80 &
mininet> h3 ping -c 20 h2
```

*`TEST_TYPE = 1`: volumetric attack (SYN flood from `h1` to `h2`)*

```
mininet> h1 hping3 -S --flood -p 80 h2
```

*`TEST_TYPE = 2`: low-rate attack (interval between packets, here 20 ms)*

```
mininet> h1 hping3 -S -p 80 -i u20000 h2
```

Stop an attack with `Ctrl+C`. Let every run last long enough: Stage 2 needs **at least 10 consecutive intervals** (30 s at `INTERVAL = 3`) of a run to form even one window, and a useful model needs many more, so run each scenario for several minutes and repeat it. Run normal traffic and low-rate traffic for equally long periods.

Every interval of the run is appended to `switch_<dpid>_data.csv` (columns `time, FER, SIGS, FSR, PLV, BLV, type`) and to `result.csv` (the same without the time). Clean up before the next run:

```
mininet> exit
```

```bash
sudo mn -c
```

`sudo mn -c` removes leftover Mininet interfaces and processes. Run it whenever a previous session did not close cleanly.

### Phase 2: Train the two stages

```bash
source venv/bin/activate
python3 train_hierarchical.py switch_1_data.csv
```

This reads the labelled intervals, trains Stage 1 and Stage 2, prints the evaluation and writes everything to `models/`. Useful options:

| Option | Meaning | Default |
|--------|---------|---------|
| `--test-files A.csv B.csv` | held-out runs used **only** for testing | none (last 30 % of each run is held out) |
| `--train-frac` | share of each run used for training when there are no test files | `0.7` |
| `--stage1`, `--stage2` | classifier: `rf`, `dt`, `knn`, `svm` | `rf` / `knn` |
| `--k`, `--weights` | KNN neighbours and weighting (`uniform` or `distance`), e.g. values from a hyper-parameter search | `5` / `distance` |
| `--window` | Stage 2 window length in intervals | `10` |
| `--interval` | monitoring interval used when the data was collected (s) | `3` |
| `--skip-first` | warm-up rows dropped at the start of every run | `1` |
| `--model-dir` | output folder | `models` |

A file argument can also carry a label rule after a colon, which is how files with an older labelling are used:

| Argument | Effect |
|----------|--------|
| `data.csv` | labels are used as written (0 normal, 1 volumetric, 2 low-rate) |
| `attack_only.csv:1` | **every** row of the file gets label 1 (a file that holds a single class) |
| `stage2.csv:1=2` | label 1 becomes 2 and the other labels stay (a mixed file with 0 = normal and 1 = attack whose attack is low-rate) |

Headerless files with six columns (`FER,SIGS,FSR,PLV,BLV,label`) are read as they are. For example, a Stage 1 file labelled 0 = normal / 1 = volumetric and a Stage 2 file labelled 0 = normal / 1 = low-rate attack are trained together with:

```bash
python3 train_hierarchical.py stage1.csv stage2.csv:1=2
```

Runs that are appended to the same CSV are separated automatically (at a header line, a label change, or a time gap of more than 2.5 intervals), so a window never spans two different runs.

### Phase 3: Detect

Set the controller to detection mode in `app1.py`:

```python
APP_TYPE = 1      # hierarchical detection
PREVENTION = 0    # 1 = also block attacker ports
```

Start the controller, then the topology, and generate traffic as in Phase 1:

```bash
source venv/bin/activate
ryu-manager app1.py
```

The controller log shows what each stage decides:

```
Application Started with Hierarchical DDoS Detection (Stage 1 volumetric | Stage 2 low-rate, window = 10 intervals = 30 s)
Switch 1: baseline interval collected, no decision yet
 Normal Traffic (Stage 1) - Stage 2 window 4/10        <- Stage 2 still filling its window
 Normal Traffic
 ____________ Volumetric attack detected in Switch 1 (Stage 1) ____________
Switch 1: volumetric alert cleared after 3 normal intervals
 ____________ Low-rate attack detected in Switch 1 (Stage 2, window of 10 intervals = 30 s) ____________
```

Every decision is also appended to `detections_switch_<dpid>.csv` (columns `time, FER, SIGS, FSR, PLV, BLV, stage1, stage2, decision, window_fill`), which is the file to use for detection-delay and timeline plots.

A low-rate decision needs a full window, so it can come at the earliest at the 10th interval after the controller starts or after the last volumetric alert.

Stop and clean up as in Phase 1 (`exit`, then `sudo mn -c`).

## Controller Settings

Set at the top of `app1.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `APP_TYPE` | `1` | `0` = data collection, `1` = hierarchical detection |
| `PREVENTION` | `0` | `1` = block attacker ports while an alert is active |
| `TEST_TYPE` | `0` | label written in data collection: `0` normal, `1` volumetric, `2` low-rate |
| `INTERVAL` | `3` | flow-statistics polling interval in seconds (the window covers `10 × INTERVAL` s) |
| `MODEL_DIR` | `models/` next to `app1.py` | where the trained models are loaded from |
| `CLEAR_AFTER` | `3` | consecutive normal intervals that clear an alert |
| `MITIGATE` | both attack types `True` | which alert types trigger port blocking |
| `BLOCK_SECONDS` | `300` | how long a blocked port stays blocked |

The window length is stored in `models/metadata.json` at training time (`--window`) and read by the controller, so training and detection always agree. Use the **same `INTERVAL`** for data collection and detection.

## Evaluation

`train_hierarchical.py` evaluates the model right after training and saves the numbers to `models/metrics.json`:

| Report | What it measures |
|--------|------------------|
| **Stage 1** | per-interval *volumetric vs. rest* (precision, recall, F1, confusion matrix) |
| **Stage 2** | *low-rate vs. normal* on windows of 10 intervals |
| **End to end** | the full hierarchy on every interval, three classes (normal / volumetric / low-rate), **including** the Stage 2 warm-up |
| **End to end, steady state** | the same without the intervals in which the window was not yet full |

The split is leak-free: by default the **last 30 %** of every run is held out and no window crosses the train/test boundary (consecutive windows overlap heavily, so a random split would leak). For a stricter test use separate runs:

```bash
python3 train_hierarchical.py run1/*.csv --test-files run2/*.csv
```

The earlier scripts `accuracy_score_Hybrid.py`, `confusion_matrix.py` and `detection_rate.py` were written for the single-stage version of this project and have **not** been updated for the hierarchical pipeline. Use `models/metrics.json` for the results of the two-stage detector.

## Results

Fill in this table with the output of `train_hierarchical.py` (`models/metrics.json`).

| Metric | Stage 1 (volumetric) | Stage 2 (low-rate) | End to end |
|--------|----------------------|--------------------|------------|
| Accuracy | – | – | – |
| Precision | – | – | – |
| Detection rate (recall) | – | – | – |
| F1-score | – | – | – |
| Confusion matrix | see `metrics.json` | see `metrics.json` | see `metrics.json` |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ImportError: cannot import name 'ALREADY_HANDLED' from 'eventlet.wsgi'` | `pip install eventlet==0.30.2` inside the virtual environment |
| `ryu-manager: command not found` | Activate the environment: `source venv/bin/activate` |
| Switch never connects to the controller | Make sure Ryu is running **before** Mininet, and check with `sudo ovs-vsctl show` |
| `Exception: Error creating interface pair` or stale links | Run `sudo mn -c` and start again |
| `ModuleNotFoundError: No module named 'sklearn'` | Activate the environment and run `pip install -r requirements.txt` |
| `hping3: command not found` | `sudo apt-get install hping3` |
| `python: command not found` | Use `python3`, as in the commands above |
| Controller stops at start with `model file not found ... run train_hierarchical.py first` | Train first (Phase 2), or point `MODEL_DIR` at the folder that holds the models |
| Trainer: `Stage 1 needs volumetric rows AND non-volumetric ...` | Collect at least one volumetric run (`TEST_TYPE = 1`) and one normal or low-rate run |
| Trainer: `Stage 2 needs normal AND low-rate runs ...` | Collect both a normal run and a low-rate run (`TEST_TYPE = 2`), each at least 10 consecutive intervals long |
| `InconsistentVersionWarning` or an error while loading the models | Retrain with the scikit-learn version that the controller uses |
| Log keeps saying `Stage 2 window k/10` | Expected during the first 10 intervals, and again for 10 intervals after every volumetric alert |
| Low-rate attack is not detected | The attack must last longer than one window (30 s); check `detections_switch_<dpid>.csv`, and make sure the training data contains enough low-rate and normal runs collected with the same `INTERVAL` |
| Ports are not blocked | Blocking needs `PREVENTION = 1`, an active alert, and a source IP that was never seen in an ARP packet on that port |

## Responsible Use

The attack commands in this repository are meant **only** for the emulated Mininet network on your own machine. Never run them against hosts, networks or services that you do not own or have explicit permission to test.

## Citation

If you use this work, please cite it (fill in the details):

```bibtex
@misc{<key>,
  author       = {<author>},
  title        = {<title>},
  year         = {<year>},
  howpublished = {\url{https://github.com/<your-username>/<your-repo>}}
}
```

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html). See the [LICENSE](LICENSE) file for details.
