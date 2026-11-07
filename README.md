# Robust DDoS Detection in Software Defined Networks using Ensemble Voting and Feature Extraction

## Project Description
This project provides a basic framework for **DDoS detection** using **Ensemble Voting** in a **Software Defined Network (SDN)**. The network is implemented using **Mininet** and controlled by **RYU**. The detection system uses a hybrid of classification algorithms for DDoS attack detection.

## Prerequisites
Before you begin, ensure you have the following dependencies installed:

### Install Ubuntu OS
Ensure you are running Ubuntu as your operating system. This guide assumes you are using Ubuntu 20.04 or later.

### Install Mininet, OpenVSwitch, and Ryu
To run the SDN network with Ryu controller and Mininet, follow these installation steps.

1. **Install Mininet**
    ```bash
    sudo apt-get install mininet
    ```

2. **Install OpenVSwitch**
    ```bash
    sudo apt-get install openvswitch-switch
    ```

3. **Install Ryu SDN Controller**
    ```bash
    sudo apt-get install python3-ryu
    ```

4. **Install Python and Dependencies**
    Ensure Python 3.x and pip are installed, then install the required Python libraries:
    ```bash
    sudo apt-get install python3-pip
    pip install numpy scikit-learn pandas matplotlib
    ```

## Running the SDN Environment

### Step 1: Run the Ryu Controller
Navigate to the directory where your Ryu controller application (`app1.py`) is located, and run it using the following command:

```bash
cd /path/to/your/ryu/app/
ryu-manager app1.py
```

### Step 2: Run the Network Topology
In another terminal, navigate to the directory with the topo5.py file and run the network topology in Mininet:

```bash
cd /path/to/your/mininet/topo/
sudo python3 topo5.py
```
This will create the SDN topology as defined in topo5.py.

### Step 3: Test the Environment (DDoS Attack Simulation)
You can simulate a DDoS attack using Hping3. First, ensure you have Hping3 installed:
```bash
sudo apt-get install hping3
```
Then, run the DDoS attack with the following command:
```bash
sudo hping3 --flood -p 80 <target-ip>
```
Replace <target-ip> with the IP address of the target machine in your network.

### Step 4: Running the Detection Algorithms
After running the network and simulating the DDoS attack, you can run the detection algorithms to evaluate the DDoS detection performance.

Execute the following Python scripts:
Run the Ensemble algorithm accuracy score:

```bash
python accuracy_score_Hybrid.py
```

Generate the confusion matrix:


```bash
python confusion_matrix.py
```

Evaluate the detection rate:

```bash
python detection_rate.py
```
These scripts will analyze the attack and evaluate the detection system's performance.

## License
This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).
