import subprocess
import argparse
import json
import sys

experiments=[, 'clip_disentangle','domain_disentangle']
targets=[ 'sketch','cartoon','photo']

parser = argparse.ArgumentParser()
parser.add_argument('--w', type=str, help='Vector argument')
args = parser.parse_args()

if not args.w:
    print("Please provide the vector argument '--w=[0.1, 0.2, 0.3]'.")
    sys.exit(-1)
else:
    weights = args.w

for target in targets:
    for experiment in experiments:
        args1 = ['python', 'main.py', '--experiment', experiment, '--target_domain', target,'--weights',weights]
        subprocess.run(args1, check=True)

#DOMGEN
for target in targets:
    for experiment in experiments:
        args1 = ['python', 'main.py', '--experiment', experiment, '--target_domain', target,'--dom_gen','--weights',weights]
        subprocess.run(args1, check=True)


'''

# Run the second Python program
args2 = ['python', 'main.py', '--e', '1', '--b', '2']
subprocess.run(args2, check=True)

'''
