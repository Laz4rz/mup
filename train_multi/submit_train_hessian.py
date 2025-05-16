import uuid
import itertools
import subprocess
import numpy as np
import time
import os
import sys
import logging
import datetime

PATH = "/home/Mikolaj/mup-repo/"

logger = logging.getLogger()
file_handler = logging.FileHandler(PATH+f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log", mode='w')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)


training_script = PATH+"train_multi/train_hessian_multiplier.py"
# models = ["mup", "ntp"]
models = ["mup"]
# models = ["sp"]
device = 0
num_epochs = 5000
# widths = [64, 128, 256, 512, 1024, 2048, 4096]
# widths = [64, 1024, 4096]
widths = [4096]
# lrs = [0.008, 0.01, 0.015, 0.03]
# lrs = [0.001, 0.0025, 0.005, 0.0075, 0.01]
lrs = [0.0025, 0.005, 0.0075]
# lrs = [0.0075]
# lrs = np.linspace(0.00001, 0.02, 25)
# lrs = [0.01]
subsets = [0.05]
toy_cifar = True
sam = True

# # lr=.005 (0.0075 is about optimal): 
# rho=0.01 exploding for some, =0.005 exploding for all (converiging to loss=1)
# [0.001, 0.0066] not enough spacing between sharpness curves for nice plots
# for lr=0.0075 rho=0.0066 started going to loss=1 already

# for rho dependence plot:
# sam_rhos = [0.0001, 0.00066, 0.0033]
# sam_rhos = [0.0001, 0.0015]
# sam_rhos = [0.0001, 0.0015, 0.003]

sam_rhos = [0.001]
lrs = [0.0025, 0.005, 0.0075]
# sam_rhos = [0.0, 0.0001, 0.0015]
# lrs = [0.005]

# lrs = np.linspace(0.001, 0.01, 12)
# sam_rhos = np.linspace(0.001, 0.01, 12)
multiplier = True
hessian_iter = 10
validation = False

seed = 2

if __name__ == '__main__':
    start_time = time.time()
    processes = []
    for i, run in enumerate(itertools.product(*[models, lrs, widths, subsets, sam_rhos])):
        uid = uuid.uuid4().hex[:10]
        model, lr, width, subset, sam_rho = run
        logging.info(f'{i}: model={model}, lr={lr:.5f}, width={width}, subset={subset:.2f}, uid={uid}')

        cmd = [
            "bash", "-c",
            f"/opt/common/envs/mup-abc/bin/python {training_script} "
            f"--model {model} --width {width} --lr {lr} --device {device} "
            f"--epochs {num_epochs} --subset {subset} --toy_cifar {toy_cifar} --seed {seed} "
            f"--sam {sam} --sam_rho {sam_rho} --multiplier {multiplier} --hessian_iter {hessian_iter} --validation {validation}"
        ]

        try:
            p = subprocess.Popen(cmd)
            processes.append(p)
        except Exception as e:
            logging.info(f"Failed to launch run {i}: {e}")

        time.sleep(0.3)

    while any([p.poll() is None for p in processes]):
        finished_processes = [p for p in processes if p.poll() is not None]
        code_0 = [p for p in finished_processes if p.returncode == 0]
        code_1 = [p for p in finished_processes if p.returncode == 1]
        logging.info(f"{len(finished_processes)}/{len(processes)} | Code 0: {len(code_0)} | Code 1: {len(code_1)} | {time.time() - start_time:.2f} seconds elapsed")
        time.sleep(10)

    logging.info(f"All processes finished in {time.time() - start_time:.2f} seconds.")