"""
Author: Eric Tran <eric.tran@heig-vd.ch>

Format converter to generate simulation file that can be used with https://github.com/mahboobkarimian/wisun-mbed-simulator
"""

import pandas as pd
from jinja2 import FileSystemLoader, Environment
import argparse
from pathlib import Path

DAG_TYPE = 'dag'
TPG_TYPE = 'tpg'

class EmulatorConvertor:
    def __init__(self):
        pass
    @staticmethod
    def convert_topology():
        df = pd.read_csv(args.file, index_col='node_id')
        templateLoader = FileSystemLoader(searchpath='./')
        templateEnv = Environment(loader=templateLoader)
        # Format : parent,child:rssi
        template = templateEnv.get_template('run.sh.j2')
        nodes = len(df) - 1 # BR does not count as a node
        tpg = ''
        link_tracking = list()
        for node_id, item in df.iterrows():
            neighbors = item['neighbors']
            for neighbor in neighbors.split(';'):
                ngh_id, rssi = neighbor.split(':')
                ngh_id = int(ngh_id)
                edge = (node_id, ngh_id)
                if node_id > ngh_id: # For consistency and tracking, we always use the lower int on the first position of the edge
                    edge = (ngh_id, node_id)
                if edge in link_tracking: # We skip link already added in the topology
                    continue
                link_tracking.append(edge)
                tpg += f"-g {edge[0]},{edge[1]}:{rssi} "
        with open(args.emulatorpath / Path('run_from_dfs.sh'), 'w') as f:
            f.write(template.render({'nodes':nodes, 'tpg': tpg}))
        print(f"Write to path {args.emulatorpath}")

    @staticmethod
    def convert_dag(file_path:str, emulator_path:str):
        df = pd.read_csv(file_path, index_col='node_id')
        templateLoader = FileSystemLoader(searchpath='./')
        templateEnv = Environment(loader=templateLoader)
        # Format : parent,child:rssi
        template = templateEnv.get_template('run.sh.j2')
        nodes = len(df) - 1 # BR does not count as a node
        tpg = ''
        link_tracking = list()
        for node_id, item in df.iterrows():
            parent = int(item['parent'])
            if parent == -1:
                continue
            rssi = item['rssi']
            edge = (node_id, parent)
            if node_id > parent: # For consistency and tracking, we always use the lower int on the first position of the edge
                edge = (parent, node_id)
            if edge in link_tracking: # We skip link already added in the topology
                continue
            link_tracking.append(edge)
            tpg += f"-g {edge[0]},{edge[1]}:{rssi} "
        with open(emulator_path / Path('run_from_dag.sh'), 'w') as f:
            f.write(template.render({'nodes':nodes, 'tpg': tpg}))
        print(f"Write to path {emulator_path}")

if __name__ == '__main__':
    parserarg = argparse.ArgumentParser()
    parserarg.add_argument('-f','--file', action='store', type=Path)
    parserarg.add_argument('-e','--emulatorpath', action='store', type=Path, default='../wisun-mbed-simulator')
    parserarg.add_argument('-t', '--type', action='store', type=str, choices=[DAG_TYPE, TPG_TYPE], default=TPG_TYPE)
    args = parserarg.parse_args()

    # Convert dags
    if args.type == DAG_TYPE:
        EmulatorConvertor.convert_dag(args.file, args.emulatorpath)
    elif args.type == TPG_TYPE:
        EmulatorConvertor.convert_topology(args.file, args.emulatorpath)