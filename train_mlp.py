import math
from time import sleep
import subprocess
import itertools
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch
from torch import nn
from torch.optim import SGD, Adam
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import argparse
import torch.multiprocessing as mp

def chunk_jobs(jobs, n_chunks):
    """Split a list of jobs into n_chunks as evenly as possible, tagging each job with a unique index."""
    chunk_sizes = [len(jobs) // n_chunks] * n_chunks
    for i in range(len(jobs) % n_chunks):
        chunk_sizes[i] += 1

    chunks = []
    start = 0
    idx = 0
    for size in chunk_sizes:
        chunk = []
        for job in jobs[start:start + size]:
            chunk.append((idx, job[0], job[1]))
            idx += 1
        chunks.append(chunk)
        start += size

    return chunks

def get_available_gpus(min_free_mem_gb=4):
    """Returns a list of GPU IDs with at least min_free_mem_gb available."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr}")
    
    free_memories = [int(x) for x in result.stdout.strip().split('\n')]
    return [i for i, mem in enumerate(free_memories) if mem >= min_free_mem_gb * 1024]


def get_available_gpus(min_free_mem_gb=4, max_utilization=10):
    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=memory.total,memory.used,utilization.gpu',
            '--format=csv,nounits,noheader'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr}")

    available_gpus = []
    for i, line in enumerate(result.stdout.strip().split('\n')):
        total_str, used_str, util_str = map(str.strip, line.split(','))
        total = int(total_str)  # in MB
        used = int(used_str)    # in MB
        util = int(util_str)    # in %

        free_mem_gb = (total - used) / 1024
        if free_mem_gb >= min_free_mem_gb and util < max_utilization:
            available_gpus.append(i)

    return available_gpus


def preload_subset(batch_size, subset_percentage, return_dataset=False):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_ds = datasets.CIFAR10(root='/tmp', train=True, download=True, transform=transform)

    torch.manual_seed(0)
    np.random.seed(0)
    subset_size = int(len(train_ds) * subset_percentage)
    indices = np.random.choice(len(train_ds), subset_size, replace=False)
    train_subset = torch.utils.data.Subset(train_ds, indices)
    xs = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
    ys = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])
    preloaded_dataset = torch.utils.data.TensorDataset(xs, ys)
    preloaded = torch.utils.data.DataLoader(preloaded_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    if return_dataset:
        return preloaded, preloaded_dataset

    return preloaded

class SP_MLP(nn.Module):
    """Initialized according to Table1 from TP4 -- the most similar training behavior to the plots"""
    def __init__(self, width=128, num_classes=10):
        super().__init__()
        self.width = width
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width,  width,  bias=False)
        self.fc_3 = nn.Linear(width,  num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc_1.weight, std=1.0)
        nn.init.normal_(self.fc_2.weight, std=self.width**(-0.5))
        nn.init.normal_(self.fc_3.weight, std=self.width**(-0.5))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc_1(x))
        h = F.relu(self.fc_2(h))
        return self.fc_3(h)
    
class NTK_MLP(nn.Module):
    """Initialized according to Table1 from TP4"""
    def __init__(self, width=128, num_classes=10):
        super().__init__()
        self.width = width
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width,  width,  bias=False)
        self.fc_3 = nn.Linear(width,  num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc_1.weight, std=self.width**(0))
        nn.init.normal_(self.fc_2.weight, std=self.width**(0))
        nn.init.normal_(self.fc_3.weight, std=self.width**(0))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = F.relu(self.fc_1(x))
        h = F.relu(self.fc_2(h) * self.width**(-0.5))
        return self.fc_3(h) * self.width**(-0.5)

class demoMLP(nn.Module):
    """SP model from the muP demo example jupyternotebook -- doesnt show expected train behavior"""
    def __init__(self, width=128, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
        super().__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, num_classes, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
        self.fc_1.weight.data /= self.input_mult**0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in')
        nn.init.zeros_(self.fc_3.weight)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.nonlin(self.fc_1(x) * self.input_mult**0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out) * self.output_mult

class MLP(nn.Module):
    """Standard MLP model -- does not show SP expected training behavior"""
    def __init__(self, width=128, num_classes=10):
        super().__init__()
        self.width = width
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, num_classes, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.fc_1(x)
        h = F.relu(h)
        h = self.fc_2(h)
        h = F.relu(h)
        h = self.fc_3(h)
        return h
    
class muMLPTab9(nn.Module):
    """muP initialized MLP model, according to Table9 from TP5 (thanks to dvruette)"""
    def __init__(self, width=128, num_classes=10):
        super().__init__()
        self.width = width
        self.input_mult = self.width**0.5
        self.output_mult = self.width**-0.5
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, num_classes, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.fc_1.weight, std=self.width**-0.5) # ? 1/fanout
        nn.init.normal_(self.fc_2.weight, std=self.width**-0.5)
        nn.init.normal_(self.fc_3.weight, std=self.width**-0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.input_mult * self.fc_1(x)
        h = self.fc_2(F.relu(h))
        h = self.output_mult * self.fc_3(F.relu(h))
        return h

    def get_parameter_groups(self, learning_rate, optimizer):
        '''
        SGD specific muP learning rates (Table 9, TP5)
        *IMPORTANT* SGD in muP just takes the LR that you pass
        This is only here for implementation completeness
        '''
        if optimizer == SGD:
            return [
                {'params': self.fc_1.parameters(), 'lr': learning_rate},
                {'params': self.fc_2.parameters(), 'lr': learning_rate},
                {'params': self.fc_3.parameters(), 'lr': learning_rate}
            ]
        elif optimizer == Adam:
            '''Adam specific muP learning rates (Table 9, TP5)'''
            return [
                {'params': self.fc_1.parameters(), 'lr': learning_rate/self.width**0.5},
                {'params': self.fc_2.parameters(), 'lr': learning_rate/self.width**0.5},
                {'params': self.fc_3.parameters(), 'lr': learning_rate/self.width}
            ]
        
class customMLP(nn.Module):
    """muP initialized MLP model, according to Table9 from TP5 (thanks to dvruette)"""
    def __init__(self, width=128, num_classes=10):
        super().__init__()
        self.width = width
        self.input_mult = self.width**0.5
        self.output_mult = self.width**-0.5
        self.fc_1 = nn.Linear(3072, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, num_classes, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.fc_1.weight, std=self.width**-0.5) # ? 1/fanout
        nn.init.normal_(self.fc_2.weight, std=self.width**-0.5)
        nn.init.normal_(self.fc_3.weight, std=self.width**-0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.input_mult * self.fc_1(x)
        h = self.fc_2(F.relu(h))
        h = self.output_mult * self.fc_3(F.relu(h))
        return h

    def get_parameter_groups(self, learning_rate, optimizer):
        '''
        SGD specific muP learning rates (Table 9, TP5)
        *IMPORTANT* SGD in muP just takes the LR that you pass
        This is only here for implementation completeness
        '''
        if optimizer == SGD:
            return [
                {'params': self.fc_1.parameters(), 'lr': learning_rate},
                {'params': self.fc_2.parameters(), 'lr': learning_rate},
                {'params': self.fc_3.parameters(), 'lr': learning_rate}
            ]
        elif optimizer == Adam:
            '''Adam specific muP learning rates (Table 9, TP5)'''
            return [
                {'params': self.fc_1.parameters(), 'lr': learning_rate/self.width**0.5},
                {'params': self.fc_2.parameters(), 'lr': learning_rate/self.width**0.5},
                {'params': self.fc_3.parameters(), 'lr': learning_rate/self.width}
            ]

def train(model, train_dl, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            train_loss += loss.item() * data.size(0)
            loss.backward()
            optimizer.step()
        
    return train_loss / len(train_dl.dataset)

def run_chunk(jobs, device, shared_tensor, preloaded, seeds, model_class, optimizer, epochs):
    torch.cuda.set_device(device)
    for job in jobs:
        job_id, log2lr, width = job
        run_experiment(log2lr, width, seeds, job_id, device, shared_tensor, preloaded, model_class, optimizer, epochs)

def run_experiment(log2lr, width, seeds, job_id, device, shared_tensor, preloaded, model_class, optimizer, epochs):
    train_dl = preloaded
    losses = []
    print(f"Running job {job_id} on device {device} with log2lr={log2lr}, width={width}")
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_class(width=width).to(device)
        # custom parameter groups for muMLP, else just use model.parameters()
        parameters = model.get_parameter_groups(2**log2lr, optimizer) if hasattr(model, 'get_parameter_groups') else model.parameters()
        optimizer = optimizer(parameters, lr=2**log2lr)
        loss = train(model, train_dl, optimizer, num_epochs=epochs, device=device)
        
        losses.append(loss)

    loss = np.mean(losses)
    shared_tensor[job_id] = loss
    print(f"Width: {width}, Log2LR: {log2lr}, Loss: {loss:.4f}, Losses: {[round(ls, 3) for ls in losses]}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="Train MLP or muMLP model.")
    parser.add_argument('--model', type=str, choices=['MLP', 'muMLP', 'demoMLP', 'SPMLP'], required=True, help="Choose the model type: 'MLP', 'muMLP', 'SPMLP' or 'demoMLP'")
    parser.add_argument('--subset', type=float, default=0.2, help="Percentage of dataset to use for training (default: 0.2)")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"], help="Optimizer to use: 'SGD' or 'Adam'")
    parser.add_argument("--lr_range", type=float, nargs=2, default=[-12, -4], help="Range of log2 learning rates to use (default: [-16, -4])")
    args = parser.parse_args()

    if args.model == 'MLP':
        model_class = MLP
    elif args.model == 'muMLP':
        model_class = muMLPTab9
    elif args.model == 'demoMLP':
        model_class = demoMLP
    elif args.model == 'SPMLP':
        model_class = SP_MLP
    else:
        raise ValueError("Invalid model type. Choose 'MLP' or 'muMLP'.")
    print(f"Using model: {args.model}, subset: {args.subset*100}%")

    optimizer = SGD if args.optimizer == "SGD" else Adam
    print(f"Using optimizer: {args.optimizer}: {optimizer}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    data_dir = '/tmp'

    preloaded = preload_subset(batch_size, args.subset)
    print(f"Preloaded dataset with {len(preloaded.dataset)} samples.")

    min_lr, max_lr = args.lr_range
    print(f"Log2 learning rate range: {min_lr} to {max_lr}")

    epochs = 20
    seeds = [2137]
    # seeds = [0, 1, 2, 3, 4]
    log2lrs = np.linspace(min_lr, max_lr, 40)
    widths = [128, 256, 512, 1024, 2048, 4096, 8192] 
    # widths = [128]

    free_memory, max_utilization = 16, 50
    availage_gpus = get_available_gpus(min_free_mem_gb=free_memory, max_utilization=max_utilization)
    if len(availage_gpus) == 0:
        raise RuntimeError(f"No available GPUs found with at least {free_memory}GB free memory and utilization < {max_utilization}%")
    availage_gpus = [0, 1, 5, 6, 7]
    devices = [f"cuda:{i}" for i in availage_gpus]
    print(f"Available devices: {len(devices)}, {availage_gpus}")

    jobs = list(itertools.product(log2lrs, widths))
    jobs_chunks = chunk_jobs(jobs, len(devices))
    print(f"Jobs: {len(jobs)}, Chunks: {len(jobs_chunks)}")
    
    processes = []
    shared_tensor = torch.zeros(len(jobs)).to(device).share_memory_()
    pbar = tqdm(total=shared_tensor.numel(), desc="Processing", unit="item")
    for enum, job_chunk in enumerate(jobs_chunks):
        device = devices[enum]

        print(f"Starting process {enum} on {device} with {len(job_chunk)} jobs")
        p = mp.Process(target=run_chunk, args=(job_chunk, device, shared_tensor, preloaded, seeds, model_class, optimizer, epochs))
        processes.append(p)
        p.start()

    while any(p.is_alive() for p in processes):
        done = shared_tensor.count_nonzero().item()
        if done > pbar.n:
            pbar.n = shared_tensor.count_nonzero().item()
            pbar.set_postfix_str(f"Completed: {shared_tensor.count_nonzero().item()}/{len(shared_tensor)}")
            pbar.refresh()
        sleep(5)
    pbar.close()

    results_df = pd.DataFrame(index=log2lrs, columns=widths)
    for i, job in enumerate(jobs):
        log2lr, width = job
        loss = shared_tensor[i].item()
        results_df.loc[log2lr, width] = loss

    plt.figure(figsize=(8, 4))
    for width in widths:
        plt.plot(results_df.index, results_df[width], label=f'Width {width}')
    plt.xlabel('Log2LR')
    plt.ylabel('Loss')
    plt.title(f'{args.model}, {args.subset*100}% of CIFAR\nLoss vs Log2LR for different widths')
    plt.xlim(np.floor(results_df.index.min())-0.5, np.ceil(results_df.index.max())+0.5)
    plt.legend()
    plt.grid()
    plt.savefig(f'results/loss_vs_log2lr_{args.model}_{args.subset}_{args.optimizer}.png')
    results_df.to_csv(f'results/loss_vs_log2lr_{args.model}_{args.subset}_{args.optimizer}.csv')
    plt.show()
