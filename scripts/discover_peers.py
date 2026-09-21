"""Read-only discovery of LAN peers; no credentials, no network changes.

Addresses are arguments, never literals in the source: this file is published and a
hardcoded address describes somebody's actual network.

    python scripts/discover_peers.py --from <local-ip> <peer-ip> [<peer-ip> ...]
"""

import argparse
import concurrent.futures
import json
import socket

PORTS = (22, 445, 3389, 5985)


def inspect(ip, source):
    result = {"ip": ip, "open_ports": []}
    for port in PORTS:
        with socket.socket() as connection:
            connection.settimeout(0.6)
            connection.bind((source, 0))
            if connection.connect_ex((ip, port)) == 0:
                result["open_ports"].append(port)
                if port == 22:
                    result["ssh_banner"] = connection.recv(256).decode(errors="replace").strip()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("peers", nargs="+", help="Peer addresses to probe.")
    parser.add_argument("--from", dest="source", required=True, help="Local address to bind.")
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        found = list(pool.map(lambda ip: inspect(ip, args.source), args.peers))
    print(json.dumps(found, indent=2))
