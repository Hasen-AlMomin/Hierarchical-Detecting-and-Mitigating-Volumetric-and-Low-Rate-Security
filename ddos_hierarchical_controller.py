"""
Ryu controller: hierarchical two-stage detection of volumetric and low-rate DDoS
attacks in an OpenFlow 1.3 SDN.

Every INTERVAL seconds the controller polls the flow table of each switch and
computes five features:

    FER  = Flow Entry Rate            (change in number of flow entries)
    SIGS = Source-IP Generation Speed (change in number of distinct source IPs)
    FSR  = Flow Success Ratio         (share of flows that have a reverse flow)
    PLV  = Packet Load Variance       (std-dev of per-flow packet deltas)
    BLV  = Byte Load Variance         (std-dev of per-flow byte deltas)

and feeds them to the hierarchical detector in hierarchical_ml.py:

    Stage 1  volumetric : classifies the current interval on its own.
    Stage 2  low-rate   : classifies a sliding window of the last 10 intervals
                          (window advances one interval at a time) that Stage 1
                          passed as non-volumetric.

Modes (see the switches below)
    APP_TYPE = 0   data collection: every interval is appended to
                   switch_<dpid>_data.csv with the label TEST_TYPE
                   (0 normal, 1 volumetric, 2 low-rate)
    APP_TYPE = 1   detection: both stages run; decisions are logged to
                   detections_switch_<dpid>.csv
    PREVENTION = 1 additionally blocks the ports that send packets whose source
                   IP was never validated by ARP, while an alert is active.

Train the models first:  python train_hierarchical.py <csv files>
"""
import csv
import os
import statistics
import time
from datetime import datetime

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.lib.packet import ipv4
from ryu.lib.packet import arp
from ryu.lib import hub

from hierarchical_ml import (HierarchicalDetector, NORMAL, VOLUMETRIC, LOW_RATE,
                             LABEL_NAMES)

# -------------------------------------------------------#
APP_TYPE = 1
# 0 data collection, 1 ddos detection (hierarchical)

PREVENTION = 0
# 1 = block attacker ports while an alert is active

# TEST_TYPE is applicable only for data collection (APP_TYPE = 0)
# 0 normal, 1 volumetric attack, 2 low-rate attack
TEST_TYPE = 0

# data collection / monitoring interval (seconds)
INTERVAL = 3

# directory holding stage1_volumetric.joblib, stage2_lowrate.joblib, metadata.json
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# an alert is cleared after this many consecutive normal intervals
CLEAR_AFTER = 3

# which attack types trigger port blocking when PREVENTION = 1
MITIGATE = {VOLUMETRIC: True, LOW_RATE: True}

# how long a blocked port stays blocked (seconds)
BLOCK_SECONDS = 300
# -------------------------------------------------------#

# BLOCKED_PORTS[dpid][port] = time at which the block expires
BLOCKED_PORTS = {}


def get_time():
    return datetime.now()


# ---------------------------------------------------
# flow serial numbers (OpenFlow cookies)
flow_cookie = {}


def get_flow_number(dpid):
    flow_cookie.setdefault(dpid, 0)
    flow_cookie[dpid] += 1
    return flow_cookie[dpid]


# ---------------------------------------------------
def calculate_value(store, key, val):
    '''
    Difference between the current and the previous counter value of a flow.
    - the first time a key is seen the difference is 0
    - if the counter went down the flow was deleted and re-created, so the
      new value itself is the difference since the counter restarted
    '''
    key = str(key).replace(".", "_")
    old = store.get(key)
    store[key] = val
    if old is None:
        return 0
    delta = val - old
    return delta if delta >= 0 else val


# ---------------------------------------------------
# CSV helpers (files are closed after every write)
def _append_row(fname, row):
    with open(fname, "a", newline="") as fh:
        csv.writer(fh).writerow(row)


def _ensure_header(fname, header):
    if not os.path.exists(fname) or os.path.getsize(fname) == 0:
        _append_row(fname, header)


def init_portcsv(dpid):
    _ensure_header("switch_%s_data.csv" % dpid,
                   ["time", "FER", "SIGS", "FSR", "PLV", "BLV", "type"])


def init_flowcountcsv(dpid):
    _ensure_header("switch_%s_flowcount.csv" % dpid, ["time", "flowcount"])


def init_detectioncsv(dpid):
    _ensure_header("detections_switch_%s.csv" % dpid,
                   ["time", "FER", "SIGS", "FSR", "PLV", "BLV",
                    "stage1", "stage2", "decision", "window_fill"])


def update_flowcountcsv(dpid, row):
    _append_row("switch_%s_flowcount.csv" % dpid, row)


def update_portcsv(dpid, row):
    _append_row("switch_%s_data.csv" % dpid, list(row) + [str(TEST_TYPE)])


def update_resultcsv(row):
    _append_row("result.csv", list(row) + [str(TEST_TYPE)])


def update_detectioncsv(dpid, row):
    _append_row("detections_switch_%s.csv" % dpid, row)


# ---------------------------------------------------
class SwitchState(object):
    """Everything the controller remembers about one switch between intervals."""

    def __init__(self):
        self.intervals = 0          # monitoring intervals seen so far
        self.prev_flow_count = 0    # for FER
        self.prev_sigs_len = 0      # for SIGS
        self.counters = {}          # per-flow previous counters for PLV / BLV
        self.alert = NORMAL         # attack type that currently drives mitigation
        self.clean_run = 0          # consecutive normal intervals since the alert


class DDoSML(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DDoSML, self).__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.arp_ip_to_port = {}
        self.datapaths = {}
        self.state = {}             # dpid -> SwitchState
        self.gflows = {}            # dpid -> flow stats of the reply being assembled
        self.detector = None

        if APP_TYPE == 1:
            self.detector = HierarchicalDetector.load(MODEL_DIR)
            self.logger.info(
                "Application Started with Hierarchical DDoS Detection "
                "(Stage 1 volumetric | Stage 2 low-rate, window = %d intervals = %d s)",
                self.detector.window_size, self.detector.window_size * INTERVAL)
        else:
            self.logger.info("Application Started with Data Collection Mode "
                             "(label TEST_TYPE = %d)", TEST_TYPE)

        self.flow_thread = hub.spawn(self._flow_monitor)

    def _state(self, dpid):
        st = self.state.get(dpid)
        if st is None:
            st = self.state[dpid] = SwitchState()
        return st

    def _flow_monitor(self):
        # initial delay
        hub.sleep(INTERVAL * 2)
        while True:
            for dp in list(self.datapaths.values()):
                self.request_flow_metrics(dp)
            hub.sleep(INTERVAL)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.datapaths[datapath.id] = datapath
        # a (re)connecting switch starts with an empty flow table, so start fresh
        self.state[datapath.id] = SwitchState()
        self.gflows[datapath.id] = []
        if self.detector is not None:
            self.detector.reset(datapath.id)

        flow_serial_no = get_flow_number(datapath.id)

        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions, flow_serial_no)

        init_portcsv(datapath.id)
        init_flowcountcsv(datapath.id)
        init_detectioncsv(datapath.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == DEAD_DISPATCHER and dp is not None:
            self.datapaths.pop(dp.id, None)
            self.gflows.pop(dp.id, None)

    def request_flow_metrics(self, datapath):
        ofp_parser = datapath.ofproto_parser
        req = ofp_parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    # ------------------------------------------------------------------
    # Feature extraction (definitions unchanged from the original code)
    # ------------------------------------------------------------------
    # FER calculation
    def _flow_Entry_Rate(self, st, flows):
        curr_flow_count = len(flows)
        FER = curr_flow_count - st.prev_flow_count
        st.prev_flow_count = curr_flow_count
        return FER

    # SIGS calculation
    def _source_ip_Generation_speed(self, st, flows):
        sources = set()
        for flow in flows:
            for key, val in flow.match.items():
                if key == "ipv4_src":
                    sources.add(val)
        cur_SIGS_len = len(sources)
        SIGS_result = cur_SIGS_len - st.prev_sigs_len
        st.prev_sigs_len = cur_SIGS_len
        return SIGS_result

    # FSR calculation
    def _Flow_Success_Ratio(self, flows):
        # total number of flows, excluding the table-miss entry
        flow_count = len(flows) - 1

        interactive_flows = {}
        for flow in flows:
            srcip = dstip = None
            for key, val in flow.match.items():
                if key == "ipv4_src":
                    srcip = val
                if key == "ipv4_dst":
                    dstip = val
            if srcip and dstip:
                fwdflowhash = srcip + "_" + dstip
                revflowhash = dstip + "_" + srcip
                # is this direction already known?
                if fwdflowhash not in interactive_flows:
                    # does the reverse direction exist?
                    if revflowhash not in interactive_flows:
                        interactive_flows[fwdflowhash] = {}
                    else:
                        interactive_flows[revflowhash][fwdflowhash] = 1
        # number of flows that belong to an interactive (two-way) pair
        iflow = 0
        for key in interactive_flows:
            if interactive_flows[key] != {}:
                iflow += 2
        if flow_count > 0:
            return float(iflow) / flow_count
        return 1.0

    # PLV / BLV calculation
    def _stddev_packets(self, st, dpid, flows):
        '''
        Standard deviation, over all flows, of the number of packets (PLV)
        and of bytes (BLV) seen during the last interval.
        '''
        packet_counts = []
        byte_counts = []
        hdr = "switch_" + str(dpid)
        for flow in flows:
            srcip = None
            dstip = None
            for key, val in flow.match.items():
                if key == "ipv4_src":
                    srcip = val
                if key == "ipv4_dst":
                    dstip = val

            if srcip is None and dstip is None:
                continue

            bytehdr = hdr + "_" + str(srcip) + "_" + str(dstip) + ".bytes_count"
            packethdr = hdr + "_" + str(srcip) + "_" + str(dstip) + ".packets_count"

            byte_counts.append(calculate_value(st.counters, bytehdr, int(flow.byte_count)))
            packet_counts.append(calculate_value(st.counters, packethdr, int(flow.packet_count)))

        # statistics.stdev needs at least two samples
        if len(packet_counts) < 2:
            return 0.0, 0.0
        return statistics.stdev(packet_counts), statistics.stdev(byte_counts)

    # ------------------------------------------------------------------
    # Flow statistics -> features -> hierarchical decision
    # ------------------------------------------------------------------
    @set_ev_cls([ofp_event.EventOFPFlowStatsReply], MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        buf = self.gflows.setdefault(dpid, [])
        buf.extend(ev.msg.body)

        if ev.msg.flags != 0:
            return                      # more reply parts are coming
        flows = buf
        self.gflows[dpid] = []          # take the complete reply and reset the buffer

        st = self._state(dpid)
        FER = self._flow_Entry_Rate(st, flows)
        SIGS = self._source_ip_Generation_speed(st, flows)
        FSR = self._Flow_Success_Ratio(flows)
        PLV, BLV = self._stddev_packets(st, dpid, flows)
        features = [FER, SIGS, FSR, PLV, BLV]
        st.intervals += 1

        if APP_TYPE == 0:
            # data collection: log every interval with the TEST_TYPE label
            t = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
            row = [t] + [str(v) for v in features]
            update_portcsv(dpid, row)
            update_resultcsv([str(v) for v in features])
        elif st.intervals == 1:
            # The first interval only establishes the baseline for the
            # "change" features (FER, SIGS, PLV, BLV): no decision yet.
            self.logger.info("Switch %d: baseline interval collected, no decision yet", dpid)
        else:
            self._detect(dpid, st, features)

        t = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
        update_flowcountcsv(dpid, [t, str(st.prev_flow_count)])

    def _detect(self, dpid, st, features):
        FER, SIGS, FSR, PLV, BLV = features
        self.logger.info("FER %s SIGS %s FSR %s PLV %s BLV %s", FER, SIGS, FSR, PLV, BLV)

        v = self.detector.step(dpid, features)

        t = time.strftime("%m/%d/%Y, %H:%M:%S", time.localtime())
        update_detectioncsv(dpid, [t] + [str(x) for x in features] +
                            [v.stage1, "" if v.stage2 is None else v.stage2,
                             LABEL_NAMES[v.label], v.window_fill])

        if v.label == VOLUMETRIC:
            self.logger.info(" ____________ Volumetric attack detected in Switch %d "
                             "(Stage 1) ____________", dpid)
            self._raise_alert(dpid, st, VOLUMETRIC)
        elif v.label == LOW_RATE:
            self.logger.info(" ____________ Low-rate attack detected in Switch %d "
                             "(Stage 2, window of %d intervals = %d s) ____________",
                             dpid, self.detector.window_size,
                             self.detector.window_size * INTERVAL)
            self._raise_alert(dpid, st, LOW_RATE)
        else:
            if v.warming_up:
                self.logger.info(" Normal Traffic (Stage 1) - Stage 2 window %d/%d",
                                 v.window_fill, self.detector.window_size)
            else:
                self.logger.info(" Normal Traffic ")
            self._note_clean_interval(dpid, st)

    def _raise_alert(self, dpid, st, attack):
        st.alert = attack
        st.clean_run = 0
        if PREVENTION == 1 and MITIGATE.get(attack, False):
            self.logger.info("Prevention Started (%s attack, switch %d)",
                             LABEL_NAMES[attack], dpid)

    def _note_clean_interval(self, dpid, st):
        if st.alert == NORMAL:
            return
        st.clean_run += 1
        if st.clean_run >= CLEAR_AFTER:
            self.logger.info("Switch %d: %s alert cleared after %d normal intervals",
                             dpid, LABEL_NAMES[st.alert], st.clean_run)
            st.alert = NORMAL
            st.clean_run = 0

    def _mitigation_active(self, dpid):
        st = self.state.get(dpid)
        return (st is not None and st.alert != NORMAL
                and MITIGATE.get(st.alert, False))

    # ------------------------------------------------------------------
    # Flow programming / mitigation
    # ------------------------------------------------------------------
    def add_flow(self, datapath, priority, match, actions, serial_no,
                 buffer_id=None, idletime=0, hardtime=0):
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
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(in_port=portnumber)
        actions = []                    # no actions = drop
        flow_serial_no = get_flow_number(datapath.id)
        self.add_flow(datapath, 100, match, actions, flow_serial_no, hardtime=BLOCK_SECONDS)

    def remove_attack_flows(self, datapath, portnumber):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(in_port=portnumber)
        mod = parser.OFPFlowMod(datapath=datapath,
                                command=ofproto.OFPFC_DELETE,
                                out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
                                match=match, instructions=[])
        datapath.send_msg(mod)

    def _block_attacker_port(self, datapath, in_port, attack):
        dpid = datapath.id
        if time.time() < BLOCKED_PORTS[dpid].get(in_port, 0):
            return                      # already blocked and the block is still active
        self.logger.info(" %s  : %s attack detected in switch %d from port %d",
                         get_time(), LABEL_NAMES[attack], dpid, in_port)
        # Delete first, block second: the delete matches every flow of the port
        # (a non-strict delete ignores priority) and would also remove the drop
        # rule if it were installed before.
        self.remove_attack_flows(datapath, in_port)
        self.logger.info(" %s:  Switch %d Removed the attacker flows ", get_time(), dpid)
        self.block_port(datapath, in_port)
        self.logger.info(" %s:  Switch %d Blocked the port %d", get_time(), dpid, in_port)
        BLOCKED_PORTS[dpid][in_port] = time.time() + BLOCK_SECONDS

    # ------------------------------------------------------------------
    # Learning switch
    # ------------------------------------------------------------------
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

        BLOCKED_PORTS.setdefault(dpid, {})

        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # if ARP packet, remember which IP address was seen on that port
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            a = pkt.get_protocol(arp.arp)
            if a.opcode == arp.ARP_REQUEST or a.opcode == arp.ARP_REPLY:
                if a.src_ip not in self.arp_ip_to_port[dpid][in_port]:
                    self.arp_ip_to_port[dpid][in_port].append(a.src_ip)

        # install a flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:

            # check IP Protocol and create a match for IP
            if eth.ethertype == ether_types.ETH_TYPE_IP:
                ip = pkt.get_protocol(ipv4.ipv4)
                srcip = ip.src
                dstip = ip.dst

                # While an alert (either stage) is active, traffic whose source
                # IP was never validated by ARP on this port is dropped and the
                # port is blocked.
                if PREVENTION and self._mitigation_active(dpid):
                    if srcip not in self.arp_ip_to_port[dpid][in_port]:
                        self._block_attacker_port(datapath, in_port, self.state[dpid].alert)
                        return

                match = parser.OFPMatch(in_port=in_port, eth_type=ether_types.ETH_TYPE_IP,
                                        ipv4_src=srcip, ipv4_dst=dstip)

                # verify if we have a valid buffer_id, if yes avoid to send both
                # flow_mod & packet_out
                flow_serial_no = get_flow_number(datapath.id)
                if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                    self.add_flow(datapath, 1, match, actions, flow_serial_no, buffer_id=msg.buffer_id)
                    return
                else:
                    self.add_flow(datapath, 1, match, actions, flow_serial_no)
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
