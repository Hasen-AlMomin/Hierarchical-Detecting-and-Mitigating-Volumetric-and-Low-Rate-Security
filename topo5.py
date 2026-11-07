#!/usr/bin/python
from mininet.topo import Topo
from mininet.net import Mininet, Host
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import OVSSwitch, Controller, RemoteController
from mininet.link import TCLink
from time import sleep
import random

'''
                      s1

	s2	S3	S4	S5

      h1      h2      h3     h4     h5  

'''

TEST_TIME = 600 #seconds
TEST_TYPE = "cli"
#normal5 attack5 cli attack_loW

class SingleSwitchTopo(Topo):
    "Single switch connected to 5 hosts."
    def build(self):
        s1 = self.addSwitch('s1')#,dpid='0000000000002203')
        s2 = self.addSwitch('s2')#,dpid='0000000000002203')
        s3 = self.addSwitch('s3')#,dpid='0000000000002203')
        s4 = self.addSwitch('s4')#,dpid='0000000000002203')
        s5 = self.addSwitch('s5')#,dpid='0000000000002203')
	
        h1 = self.addHost('h1', ip='10.1.1.1/24', mac="00:00:00:00:00:01", defaultRoute="via 10.1.1.100")
        h2 = self.addHost('h2', ip='10.1.1.2/24', mac="00:00:00:00:00:02", defaultRoute="via 10.1.1.100")
        h3 = self.addHost('h3', ip='10.1.1.3/24', mac="00:00:00:00:00:03", defaultRoute="via 10.1.1.100")
        h4 = self.addHost('h4', ip='10.1.1.4/24', mac="00:00:00:00:00:04", defaultRoute="via 10.1.1.100")
        h5 = self.addHost('h5', ip='10.1.1.5/24', mac="00:00:00:00:00:05", defaultRoute="via 10.1.1.100")
        h6 = self.addHost('h6', ip='10.1.1.6/24', mac="00:00:00:00:00:06", defaultRoute="via 10.1.1.100")
        h7 = self.addHost('h7', ip='10.1.1.7/24', mac="00:00:00:00:00:07", defaultRoute="via 10.1.1.100")
        h8 = self.addHost('h8', ip='10.1.1.8/24', mac="00:00:00:00:00:08", defaultRoute="via 10.1.1.100")
        h9 = self.addHost('h9', ip='10.1.1.9/24', mac="00:00:00:00:00:09", defaultRoute="via 10.1.1.100")
        h10 = self.addHost('h10', ip='10.1.1.10/24', mac="00:00:00:00:00:10", defaultRoute="via 10.1.1.100")
        h11 = self.addHost('h11', ip='10.1.1.11/24', mac="00:00:00:00:00:11", defaultRoute="via 10.1.1.100")
        h12 = self.addHost('h12', ip='10.1.1.12/24', mac="00:00:00:00:00:12", defaultRoute="via 10.1.1.100")
        h13 = self.addHost('h13', ip='10.1.1.13/24', mac="00:00:00:00:00:13", defaultRoute="via 10.1.1.100")
        h14 = self.addHost('h14', ip='10.1.1.14/24', mac="00:00:00:00:00:14", defaultRoute="via 10.1.1.100")
        h15 = self.addHost('h15', ip='10.1.1.15/24', mac="00:00:00:00:00:15", defaultRoute="via 10.1.1.100")
        h16 = self.addHost('h16', ip='10.1.1.16/24', mac="00:00:00:00:00:16", defaultRoute="via 10.1.1.100")

        self.addLink(s2, s1, cls=TCLink, bw=10)
        self.addLink(s3, s1, cls=TCLink, bw=10)
        self.addLink(s4, s1, cls=TCLink, bw=10)
        self.addLink(s5, s1, cls=TCLink, bw=10)

        self.addLink(h1, s2, cls=TCLink, bw=10)
        self.addLink(h2, s2, cls=TCLink, bw=10)
        self.addLink(h3, s2, cls=TCLink, bw=10)
        self.addLink(h4, s2, cls=TCLink, bw=10)

        self.addLink(h5, s3, cls=TCLink, bw=10)
        self.addLink(h6, s3, cls=TCLink, bw=10)
        self.addLink(h7, s3, cls=TCLink, bw=10)
        self.addLink(h8, s3, cls=TCLink, bw=10)

        self.addLink(h9, s4, cls=TCLink, bw=10)
        self.addLink(h10, s4, cls=TCLink, bw=10)
        self.addLink(h11, s4, cls=TCLink, bw=10)
        self.addLink(h12, s4, cls=TCLink, bw=10)

        self.addLink(h13, s5, cls=TCLink, bw=10)
        self.addLink(h14, s5, cls=TCLink, bw=10)
        self.addLink(h15, s5, cls=TCLink, bw=10)
        self.addLink(h16, s5, cls=TCLink, bw=10)
       

if __name__ == '__main__':
    setLogLevel('info')
    topo = SingleSwitchTopo()
    c1 = RemoteController('c1', ip='127.0.0.1')
    net = Mininet(topo=topo, controller=c1)
    net.start()

    if TEST_TYPE == "normal5":
        #print "Generating NORMAL Traffic......."
        #start server in h5 h16
        h5 = net.get('h5')
        h16 = net.get('h16')
        cmd1 = "iperf -s &"
        cmd2 = "iperf -u -s &"
        h5.cmd(cmd1)
        h5.cmd(cmd2)
        h16.cmd(cmd1)
        h16.cmd(cmd2)
        sleep(5)
        #start clients in h3 and h4
        h1 = net.get('h1')
        h2 = net.get('h2')
        h3 = net.get('h3')
        h4 = net.get('h4')
        #h5 = net.get('h5')
        h6 = net.get('h6')
        h7 = net.get('h7')
        h8 = net.get('h8')
        h9 = net.get('h9')
        h10 = net.get('h10')
        h11 = net.get('h11')
        h12 = net.get('h12')
        h13 = net.get('h13')
        h14 = net.get('h14')
        h15 = net.get('h15')
        #h16 = net.get('h16')
        cmd1 = "bash normal5.sh &"
        h1.cmd(cmd1)
        h2.cmd(cmd1)
        h3.cmd(cmd1)
        h4.cmd(cmd1)
        #h5.cmd(cmd1)
        h6.cmd(cmd1)
        h7.cmd(cmd1)
        h8.cmd(cmd1)
        h9.cmd(cmd1)
        h10.cmd(cmd1)
        h11.cmd(cmd1)
        h12.cmd(cmd1)
        h13.cmd(cmd1)
        h14.cmd(cmd1)
        h15.cmd(cmd1)
        #h16.cmd(cmd1)
        cmd1 = "iperf -u -b 2m -c 10.1.1.5  -t " + str(TEST_TIME) + "  & "
        h3.cmd(cmd1)
        h4.cmd(cmd1)
        h7.cmd(cmd1)
        h8.cmd(cmd1)
        h9.cmd(cmd1)
        h10.cmd(cmd1)
        h11.cmd(cmd1)
        h12.cmd(cmd1)
        h13.cmd(cmd1)
        h14.cmd(cmd1)
        h15.cmd(cmd1)

        cmd1 = "iperf -c 10.1.1.5  -t " + str(TEST_TIME) + "  & "
        h3.cmd(cmd1)
        h4.cmd(cmd1)
        h8.cmd(cmd1)
        h9.cmd(cmd1)
        h10.cmd(cmd1)
        h11.cmd(cmd1)
        sleep(TEST_TIME)
        net.stop()

    elif TEST_TYPE == "attack5":
        print("Generating ATTACK Traffic.......")
        h1 = net.get('h1')
        h2 = net.get('h2')
        cmd1 = "bash attack5.sh &"
        h1.cmd(cmd1)
        h2.cmd(cmd1)

        sleep(TEST_TIME)
        net.stop()

    elif TEST_TYPE == "attack_loW":
        print("Generating low Rate ATTACK Traffic.......")
        h1 = net.get('h1')
        h2 = net.get('h2')
        cmd1 = "bash attack5.sh &"
        h1.cmd(cmd1)
        h2.cmd(cmd1)

        sleep(TEST_TIME)
        net.stop()

    else:
        #print "Generating NORMAL Traffic......."
        #start server in h5
        h5 = net.get('h5')
        cmd1 = "iperf -s &"
        cmd2 = "iperf -u -s &"
        h5.cmd(cmd1)
        h5.cmd(cmd2)
        sleep(5)
        #start clients in h3 and h4
        h1 = net.get('h1')
        h2 = net.get('h2')
        h3 = net.get('h3')
        h4 = net.get('h4')
        cmd1 = "bash normal.sh &"
        h1.cmd(cmd1)
        h2.cmd(cmd1)
        h3.cmd(cmd1)
        h4.cmd(cmd1)
        cmd1 = "iperf -u -b 2m -c 10.1.1.5  -t  " + str(TEST_TIME) + "  & "
        h1.cmd(cmd1)
        h2.cmd(cmd1)

        cmd1 = "iperf -c 10.1.1.5  -t " + str(TEST_TIME) + "  & "
        h3.cmd(cmd1)
        h4.cmd(cmd1)
        CLI(net)
        net.stop()
