import subprocess
import argparse
import json
import sys

experiments=['baseline','domain_disentangle','clip_disentangle']
targets=[ 'cartoon','sketch','photo']

for target in targets:
    for experiment in experiments:
        args1 = ['python', 'main.py', '--experiment', experiment, '--target_domain', target]
        subprocess.run(args1, check=True)

#DOMGEN
for target in targets:
    for experiment in experiments:
        args1 = ['python', 'main.py', '--experiment', experiment, '--target_domain', target,'--dom_gen']
        subprocess.run(args1, check=True)
