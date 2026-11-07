# Robust DDoS Detection in Software-Defined Networks using Ensemble Voting and Feature Extraction

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)
![Platform: Ubuntu](https://img.shields.io/badge/Platform-Ubuntu%2020.04%20%7C%2022.04-orange)
![Python](https://img.shields.io/badge/Python-3.8--3.10-green)

A lab framework for detecting **DDoS attacks** in a **Software-Defined Network (SDN)**. The network is emulated with **Mininet** and **Open vSwitch**, the switches are controlled by a **Ryu** controller, and attack traffic is classified by a **hybrid ensemble of machine-learning classifiers combined by voting**.

## Contents

- [How It Works](#how-it-works)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
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
                                              flow statistics collected
                                                       │
                                                       ▼
                                              feature extraction
                                                       │
                                                       ▼
                                   ensemble voting classifier (hybrid)
                                                       │
                                                       ▼
                                           normal  /  DDoS attack
```

1. **Topology** (`topo5.py`): Mininet builds the virtual network and connects its switch(es) to the remote Ryu controller.
2. **Controller** (`app1.py`): the Ryu application manages the switches and collects flow statistics from them.
3. **Features**: traffic features are extracted from the collected flow statistics.
4. **Detection**: several classifiers are combined by **ensemble voting** to label traffic as normal or attack.
5. **Evaluation**: separate scripts compute the accuracy, confusion matrix and detection rate.

## Repository Structure

```
.
├── app1.py                    # Ryu controller application
├── topo5.py                   # Mininet topology
├── accuracy_score_Hybrid.py   # Accuracy of the ensemble (hybrid) model
├── confusion_matrix.py        # Confusion matrix
├── detection_rate.py          # Detection rate
├── requirements.txt           # Python dependencies for the evaluation scripts
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

Python libraries for the evaluation scripts: `numpy`, `pandas`, `scikit-learn`, `matplotlib`.

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
pandas
scikit-learn
matplotlib
```

## Usage

Open **three terminals** in the repository folder. Start them in this order.

### Step 1: Start the Ryu controller (Terminal 1)

```bash
source venv/bin/activate
ryu-manager app1.py
```

Leave it running. By default Ryu listens for switches on port `6653`.

### Step 2: Start the Mininet topology (Terminal 2)

```bash
sudo python3 topo5.py
```

This creates the SDN topology defined in `topo5.py` and connects it to the controller. You will get the `mininet>` prompt. Check that the network works:

```
mininet> pingall
```

### Step 3: Generate traffic (Terminal 2, inside the `mininet>` prompt)

Attack traffic must be generated **from a Mininet host**, not from your own machine, so it travels through the emulated switches and the controller can see it.

Optional: start a web server on the target host and generate normal traffic first.

```
mininet> h2 python3 -m http.server 80 &
mininet> h3 ping -c 20 h2
```

Simulate a **SYN flood** from host `h1` to host `h2`:

```
mininet> h1 hping3 -S --flood -p 80 h2
```

Stop the attack with `Ctrl+C`. A slower, lower-rate attack can be simulated with an interval between packets (here 20 ms):

```
mininet> h1 hping3 -S -p 80 -i u20000 h2
```

Replace `h1`, `h2` and `h3` with the host names defined in `topo5.py`.

### Step 4: Stop and clean up

```
mininet> exit
```

```bash
sudo mn -c
```

`sudo mn -c` removes leftover Mininet interfaces and processes. Run it whenever a previous session did not close cleanly.

## Evaluation

After traffic has been collected, run the evaluation scripts in the virtual environment (no `sudo` needed):

```bash
source venv/bin/activate

# Accuracy of the ensemble (hybrid) model
python3 accuracy_score_Hybrid.py

# Confusion matrix
python3 confusion_matrix.py

# Detection rate
python3 detection_rate.py
```

Each script reads the traffic dataset / predictions produced in the previous steps and prints or plots its metric.

## Results

Fill in this table with the output of your scripts.

| Metric | Value |
|--------|-------|
| Accuracy | – |
| Detection rate | – |
| Confusion matrix | see `confusion_matrix.py` output |

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

## Responsible Use

The attack commands in this repository are for **educational and research purposes in an isolated, emulated network only**. Never run flood or scanning tools against networks or machines you do not own or do not have explicit permission to test.

## Citation

If you use this work in research, please cite it:

```bibtex
@misc{<citation-key>,
  author = {<Your Name>},
  title  = {Robust DDoS Detection in Software-Defined Networks using Ensemble Voting and Feature Extraction},
  year   = {<year>},
  url    = {https://github.com/<your-username>/<your-repo>}
}
```

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html). See the [LICENSE](LICENSE) file for details.
