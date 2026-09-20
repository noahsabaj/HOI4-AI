"""Read-only discovery of already observed LAN peers; no credentials or network changes."""

import concurrent.futures
import json
import socket

PEERS = ["<lan-address>", "<peer-address>", "<lan-address>", "<lan-address>"]


def inspect(ip):
    result = {"ip": ip, "open_ports": []}
    for port in [22, 445, 3389, 5985]:
        with socket.socket() as connection:
            connection.settimeout(0.6)
            connection.bind(("<coordinator-address>", 0))
            if connection.connect_ex((ip, port)) == 0:
                result["open_ports"].append(port)
                if port == 22:
                    result["ssh_banner"] = connection.recv(256).decode(errors="replace").strip()
    return result


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        print(json.dumps(list(pool.map(inspect, PEERS)), indent=2))
