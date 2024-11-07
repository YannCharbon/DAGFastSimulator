import pandas as pd
from jinja2 import FileSystemLoader, Environment
import argparse
from pathlib import Path

if __name__ == '__main__':
    parserarg = argparse.ArgumentParser()
    parserarg.add_argument('-f','--file', action='store', type=Path)
    parserarg.add_argument('-e','--emulatorpath', action='store', type=Path, default='../wisun-mbed-simulator')
    args = parserarg.parse_args()
    df = pd.read_csv(args.file, index_col='node_id')
    templateLoader = FileSystemLoader(searchpath='./')
    templateEnv = Environment(loader=templateLoader)
    # Format : parent,child:rssi
    template = templateEnv.get_template('run.sh.j2')
    nodes = len(df)
    tpg = ''
    for node_id, item in df.iterrows():
        if item['parent'] == -1:
            continue
        tpg += f"-g {item['parent']},{node_id}:{item['rssi']} "
    with open(args.emulatorpath / Path('run_from_dfs.sh'), 'w') as f:
        f.write(template.render({'nodes':nodes, 'tpg': tpg}))
    print(f"Write to path {args.emulatorpath}")