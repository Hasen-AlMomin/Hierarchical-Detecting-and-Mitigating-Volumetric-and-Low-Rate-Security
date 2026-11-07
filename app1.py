from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types

from ryu.lib.packet import in_proto
from ryu.lib.packet import ipv4
from ryu.lib.packet import icmp
from ryu.lib.packet import tcp
from ryu.lib.packet import udp
from ryu.lib.packet import arp
from sklearn.ensemble import VotingClassifier  # Added this import
from ryu.lib import hub
import csv
import time
import math 
import statistics 
from datetime import datetime

from ml import MachineLearningAlgo
#from ml2 import MachineLearningAlgo2


#Defintions SIGS (source IP Generation Speed), FER =  FLOW ENtry Rate,  FSR = FLOW Success Ratio,  PLV= Packet Load Variance, BLV=Byte Load Variance

#-------------------------------------------------------#
APP_TYPE = 1
#0 datacollection, 1 ddos detection
PREVENTION = 0
# ddos prevention


#TEST_TYPE is applicable only for datacollection
#0  normal, 1 attack
TEST_TYPE = 0


#data collection interval
INTERVAL = 3
#-------------------------------------------------------#



#gflows = []


#source_ip_Generation_speed_SIGS = 0
#flow_Entry_Rate_FER = 0





#FLOW_SERIAL_NO = 0
#iteration = 0

#FLOW_SERIAL_NO = {}




#BLOCKED_PORTS = []
BLOCKED_PORTS = {}

#---------------------------------------------------

gflows = {}
iteration = {}


def get_iteration(dpid):
    global iteration
    iteration.setdefault(dpid, 0)
    return iteration[dpid]

def set_iteration(dpid, count):
    global iteration
    iteration.setdefault(dpid, 0)
    iteration[dpid] = count


#---------------------------------------------------

old_source_ip_Generation_speed_SIGS_len = {}

def get_old_source_ip_Generation_speed_SIGS_len(dpid):
    global old_source_ip_Generation_speed_SIGS_len
    old_source_ip_Generation_speed_SIGS_len.setdefault(dpid, 0)
    return old_source_ip_Generation_speed_SIGS_len[dpid]

def set_old_source_ip_Generation_speed_SIGS_len(dpid, count):
    global old_source_ip_Generation_speed_SIGS_len
    old_source_ip_Generation_speed_SIGS_len.setdefault(dpid, 0)
    old_source_ip_Generation_speed_SIGS_len[dpid] = count


#---------------------------------------------------
# TO count FER

prev_flow_count = {}
def get_prev_flow_count(dpid):
    global prev_flow_count
    prev_flow_count.setdefault(dpid, 0)
    return prev_flow_count[dpid]

def set_prev_flow_count(dpid, count):
    global prev_flow_count
    prev_flow_count.setdefault(dpid, 0)
    prev_flow_count[dpid] = count

#---------------------------------------------------


flow_cookie = {}

#flow serial number
def get_flow_number(dpid):
    global flow_cookie
    flow_cookie.setdefault(dpid, 0)
    flow_cookie[dpid] = flow_cookie[dpid] + 1
    return  flow_cookie[dpid]

def get_time():
    #return time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
    return datetime.now()

#------------------------------
keystore = {}


def calculate_value(key, val):
    '''
    This function we are calculating packets difference.
    - store the previous key and value  in keystore.
    - when we get the key with the new value, 
       1. check the key exists in keystore
       2. Then takes the old value
       3. calculate the paketes  =  (new value - oldvalue) 
       4. assing the new value in the keystone.
    '''
    key = str(key).replace(".", "_")
    if key in keystore:
        oldval = keystore[key]
        cval = (val - oldval) 
        # storing the val
        keystore[key] = val
        return cval
    else:
        keystore[key] = val
        return 0


#---------------------------------------------------


def init_portcsv(dpid):
    fname = "switch_" + str(dpid) + "_data.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    header = ["time", "FER","SIGS","FSR", "PLV", "BLV","type"]
    writ.writerow(header)


def init_flowcountcsv(dpid):
    fname = "switch_" + str(dpid) + "_flowcount.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    header = ["time", "flowcount"]
    writ.writerow(header)



def update_flowcountcsv(dpid, row):
    fname = "switch_" + str(dpid) + "_flowcount.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    writ.writerow(row)


def update_portcsv(dpid, row):
    fname = "switch_" + str(dpid) + "_data.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    row.append(str(TEST_TYPE))
    writ.writerow(row)


def update_resultcsv(row):
    fname = "result.csv"
    writ = csv.writer(open(fname, 'a', buffering=1), delimiter=',')
    row.append(str(TEST_TYPE))
    writ.writerow(row)

#---------------------------------------------------


class DDoSML(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DDoSML, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.flow_thread = hub.spawn(self._flow_monitor)
        self.datapaths = {}
        self.mitigation = 0
        self.mlobj = None
        #self.mlobj2 = None        
        self.arp_ip_to_port = {}

        if APP_TYPE == 1:
            self.mlobj = MachineLearningAlgo()
            #self.mlobj2 = MachineLearningAlgo2()
            self.logger.info("Application Started with DDoS Detection (ML) Mode")
        else:
            self.logger.info("Application Started with Data Collection Mode")


    def _flow_monitor(self):
        #inital delay
        hub.sleep(INTERVAL*2)
        while True:
            #self.logger.info("Starts Flow monitoring")
            for dp in self.datapaths.values():
                self.request_flow_metrics(dp)
            hub.sleep(INTERVAL)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath

        flow_serial_no = get_flow_number(datapath.id)

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions, flow_serial_no)

        init_portcsv(datapath.id)
        init_flowcountcsv(datapath.id)



    def request_flow_metrics(self, datapath):
        ofp = datapath.ofproto
        ofp_parser = datapath.ofproto_parser
        req = ofp_parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

#FER calculation 
    def _flow_Entry_Rate(self, dpid, flows):
        global prev_flow_count
        curr_flow_count = 0
        #collect the packet_count from all the flows
        for flow in flows:
            curr_flow_count += 1

        #print "flow_Entry_Rate_FER ", flow_count
        FER = curr_flow_count - get_prev_flow_count(dpid)
        set_prev_flow_count(dpid, curr_flow_count)
        #flow_Entry_Rate_FER = curr_flow_count
        return FER

#SIGS Calculation 
    def _source_ip_Generation_speed(self, dpid, flows):
        #global old_source_ip_Generation_speed_SIGS_len
        SIGS = [] 
        #print "length of flow table " ,len(flows)
        for flow in flows:
            m = {}
            for i in flow.match.items():
                key = list(i)[0]  # match key 
                val = list(i)[1]  # match value 
                if key == "ipv4_src":
                    #print key,val
                    if val not in SIGS:
                        SIGS.append(val)
        #print "source_ips ", SIGS
        cur_SIGS_len = len(SIGS)
        SIGS_result = cur_SIGS_len - get_old_source_ip_Generation_speed_SIGS_len(dpid)

        set_old_source_ip_Generation_speed_SIGS_len(dpid, cur_SIGS_len)
        #source_ip_Generation_speed_SIGS = cur_SIGS_len
        #print "SIGS ", SIGS
        return SIGS_result

#Calculate FSR
    def _Flow_Success_Ratio(self, dpid, flows):
        #find total number of flows
        # find interactive flows (ideal case - all - 1 )
        flow_count = 0
        for flow in flows:
            flow_count += 1
        #print "total number of flows ", flow_count
        #excluding the table miss entry from flow count
        flow_count -= 1

        interactive_flows = {}
        for flow in flows:
            m = {}
            srcip = dstip = None
            for i in flow.match.items():
                key = list(i)[0]  # match key 
                val = list(i)[1]  # match value 
                if key == "ipv4_src":
                    srcip = val
                    #print key,val
                if key == "ipv4_dst":
                    dstip = val
            if srcip and dstip:
                fwdflowhash = srcip + "_" + dstip
                revflowhash = dstip + "_" + srcip
                #check flowhash is already exist
                if not fwdflowhash in interactive_flows:
                    #check you have reverse flowhash exists?
                    if not revflowhash in interactive_flows:
                        interactive_flows[fwdflowhash] = {}
                    else:
                        interactive_flows[revflowhash][fwdflowhash] = 1
        #identify number of interactive flows
        onesideflow = iflow = 0
        for key in interactive_flows:
            if interactive_flows[key] == {}:
                onesideflow += 1
            else:
                iflow +=2
        #print "interactive_flows", interactive_flows
        #print "oneside flow", onesideflow
        #print "interactive flow ", iflow
        if flow_count != 0 :
            FSR = float(iflow) / flow_count
            #print "FSR ", FSR
            return FSR
        return 1.0


    def _stddev_packets(self, dpid, flows):
        '''
        This is the standard deviation of the number of packets in the T period
        This is the standard deviation of the number of bytes in the T period
        '''
        packet_counts = []
        byte_counts = []
        hdr = "switch_" + str(dpid)
        for flow in flows:

            m = {}
            srcip = None
            dstip = None
            for i in flow.match.items():
                key = list(i)[0]  # match key 
                val = list(i)[1]  # match value 
                if key == "ipv4_src":
                    srcip = val
                    #print key,val
                if key == "ipv4_dst":
                    dstip = val

            if srcip == None and dstip == None:
                continue

            bytehdr = hdr + "_" + str(srcip) + "_" + str(dstip) + ".bytes_count"
            packethdr = hdr + "_" + str(srcip) + "_" + str(dstip) + ".packets_count"

            bytescnt = calculate_value(bytehdr, int(flow.byte_count))
            #bytescnt = flow.byte_count
            byte_counts.append(bytescnt)
            pktscnt = calculate_value(packethdr, int(flow.packet_count))
            #pktscnt = flow.packet_count
            packet_counts.append(pktscnt)

        #self.logger.info(packet_counts)
        Packet_load_Variance = statistics.stdev(packet_counts)
        Byte_load_Variance = statistics.stdev(byte_counts)
        return Packet_load_Variance, Byte_load_Variance

    @set_ev_cls([ofp_event.EventOFPFlowStatsReply], MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        global gflows
        t_flows = ev.msg.body
        flags = ev.msg.flags
        dpid = ev.msg.datapath.id
        gflows.setdefault(dpid, [])

        gflows[dpid].extend(t_flows)

        if flags == 0:
            FER  = self._flow_Entry_Rate(dpid, gflows[dpid])
            SIGS = self._source_ip_Generation_speed(dpid, gflows[dpid])
            FSR = self._Flow_Success_Ratio(dpid, gflows[dpid])
            PLV, BLV = self._stddev_packets(dpid, gflows[dpid])

            if APP_TYPE == 1 and get_iteration(dpid) == 1:
                self.logger.info("FER %s SIGS %s FSR %s PLV %s BLV %s", FER,SIGS,FSR,PLV,BLV)
                result = self.mlobj.classify([FER,SIGS,FSR,PLV,BLV])
            

                #print "Attack result ", result
                #svmobj.classify([7.6721512473,1657.02046716,0,0,0])
                # Convert the value in 'result' to an integer
                result = int(result)
                if result == 1:
                #if  1 in result:
                    # to make the result haybird with 2 machine learning .... 
                    #result2 = self.mlobj2.classify([FER, SIGS, FSR, PLV, BLV])
                    #if 1 in result2:
                    
                    #self.logger.info(" %s Attack detected in Switch %d", get_time(), dpid)
                    self.logger.info(" _________________________________________________________________________________________________Attack detected in Switch %d______________________________________________", dpid)
                    self.mitigation = 1
                    if PREVENTION == 1 :
                        self.logger.info("Prevention Started")

                if result == 0:
                #if 0 in result:
                    #self.logger.info(" %s Normal Traffic %d ", get_time(), dpid)
                    self.logger.info(" Normal Traffic ")
            else:
                t = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
                row = [t, str(FER), str(SIGS), str(FSR), str(PLV), str(BLV)]
                #self.logger.info(row)
                update_portcsv(dpid, row)
                update_resultcsv([str(FER), str(SIGS), str(FSR), str(PLV), str(BLV)])

            gflows[dpid] = []
            #iteration = 1
            set_iteration(dpid, 1)

            #update the flowcount csv vile
            t = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
            update_flowcountcsv(dpid, [t, str(prev_flow_count)])




    def add_flow(self, datapath, priority, match, actions,serial_no, buffer_id=None, idletime=0, hardtime=0):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, cookie=serial_no, buffer_id=buffer_id,
                                    idle_timeout=idletime, hard_timeout=hardtime,
                                    priority=priority, match=match,
                                    instructions=inst)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, cookie=serial_no, priority=priority,
                                    idle_timeout=idletime, hard_timeout=hardtime,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)


    def block_port(self, datapath, portnumber):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(in_port=portnumber)
        actions = []
        flow_serial_no = get_flow_number(datapath.id)
        self.add_flow(datapath, 100, match, actions, flow_serial_no, hardtime=300)        


    def remove_attack_flows(self, datapath, portnumber):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(in_port=portnumber)
        instructions = []
        mod = parser.OFPFlowMod(datapath=datapath,
                                command=ofproto.OFPFC_DELETE,
                                out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
                                match=match, instructions=[])
        datapath.send_msg(mod)
        #self.logger.info(" deleted flow %s", mod)



    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        # If you hit this you might want to increase
        # the "miss_send_length" of your switch
        if ev.msg.msg_len < ev.msg.total_len:
            self.logger.debug("packet truncated: only %s of %s bytes",
                              ev.msg.msg_len, ev.msg.total_len)
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return
        dst = eth.dst
        src = eth.src

        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})
        self.arp_ip_to_port.setdefault(dpid, {})
        self.arp_ip_to_port[dpid].setdefault(in_port, [])
        #self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)

        BLOCKED_PORTS.setdefault(dpid, [])

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        #if ARP Request packet , log the IP and MAC Address from that port
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            #self.logger.info("Received ARP Packet %s %s %s ", dpid, src, dst)
            a = pkt.get_protocol(arp.arp)
            #print "arp packet ", a
            if a.opcode == arp.ARP_REQUEST or a.opcode == arp.ARP_REPLY:
                if not a.src_ip in self.arp_ip_to_port[dpid][in_port]:
                    self.arp_ip_to_port[dpid][in_port].append(a.src_ip)
                    #print "arp_ip_to_port " ,self.arp_ip_to_port

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:

            # check IP Protocol and create a match for IP
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip = pkt.get_protocol(ipv4.ipv4)
                srcip = ip.src
                dstip = ip.dst
                protocol = ip.proto


                if self.mitigation and PREVENTION:
                    if not (srcip in self.arp_ip_to_port[dpid][in_port]):
                        if not in_port in BLOCKED_PORTS[dpid]:
                            self.logger.info(" %s  : attack detected in switch %d from port %d", get_time(), dpid, in_port)
                            self.block_port(datapath, in_port)
                            self.logger.info(" %s:  Switch %d Blocked the port %d",get_time(), dpid, in_port)
                            self.remove_attack_flows(datapath, in_port)
                            self.logger.info(" %s:  Switch %d Removed the attacker flows ",get_time(), dpid )
                            BLOCKED_PORTS[dpid].append(in_port)
                            self.block_port(datapath, in_port)
                        #print ip
                        return

                match = parser.OFPMatch(in_port=in_port, eth_type=ether_types.ETH_TYPE_IP, ipv4_src=srcip, ipv4_dst=dstip)

                # verify if we have a valid buffer_id, if yes avoid to send both
                # flow_mod & packet_out  
                flow_serial_no = get_flow_number(datapath.id)                
                if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                    self.add_flow(datapath, 1, match, actions, flow_serial_no,  buffer_id=msg.buffer_id)
                    return
                else:
                    self.add_flow(datapath, 1, match, actions, flow_serial_no)
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
