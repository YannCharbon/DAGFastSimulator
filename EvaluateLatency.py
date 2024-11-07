from scapy.all import *
import argparse

SUBNET = 'fd12:3456::302:304:506:'

if __name__ == '__main__':
    parseargs = argparse.ArgumentParser(prog="Evaluate Latency program")
    parseargs.add_argument('-n', '--nodes', action='store', type=int, default=15)
    parseargs.add_argument('-p', '--packets', action='store', type=int, default=100)
    parseargs.add_argument('-i', '--interval', action='store', type=float, default=0.5)
    parseargs.add_argument('-t', '--timeout', action='store', type=int, default=None)
    args = parseargs.parse_args()
    targets = [f'{SUBNET}{id:x}' for id in range(args.nodes)]

    conf.raw_layer = IPv6 # Required in order to receive reply for ICMPv6
    print("Transmission in progress...")
    ans, unans = srloop(IPv6(dst=targets)/ICMPv6EchoRequest(), inter=args.interval, count=args.packets, timeout=args.timeout, iface='tun0', verbose=False)
    nodes_tracking = {host: {'latencies': [], 'icmp_success': 0, 'icmp_fail': 0} for host in targets}
    for entry in ans:
        icmp_req = entry[0]
        icmp_resp = entry[1]
        nodes_tracking[icmp_req.dst]['latencies'].append(icmp_resp.time - icmp_req.sent_time)
        nodes_tracking[icmp_req.dst]['icmp_success'] += 1
    
    for entry in unans:
        nodes_tracking[entry.dst]['icmp_fail'] += 1
    print("Average icmp latency per node:")
    for node_ip, node_data in nodes_tracking.items():
        print(f"Node ID: {int(node_ip.split(':')[-1], 16)}")
        print(f"    IPv6 : {node_ip}")
        print(f"    Average latency : {sum(node_data['latencies']) / len(node_data['latencies']) if len(node_data['latencies']) > 0 else 0.0}")
        print(f"    ICMP success : {node_data['icmp_success']}")
        print(f"    ICMP fail : {node_data['icmp_fail']}")
