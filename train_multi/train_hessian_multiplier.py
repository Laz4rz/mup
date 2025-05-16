import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
import math

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from train_mlp import get_cifar, muMLPTab9, SP_MLP, NTK_MLP, get_cifar_toy

MODELS = {
    "mup": muMLPTab9,
    "sp": SP_MLP,
    "ntp": NTK_MLP,
}

CRITERIONS = {
    "CEL": nn.CrossEntropyLoss(),
    "MSE": nn.MSELoss(),
}


def train_epoch_eigen(model, train_dl, criterion, optimizer, device, sharp_x, sharp_y, hessian_iter=10, verbose=False):
    model.train()
    train_loss = 0
    iteration = 0
    lambdas = []
    for data, target in train_dl:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item() * data.size(0)
        loss.backward()
        optimizer.step()
        iteration += 1
        if iteration % hessian_iter == 0 or iteration == len(train_dl):
            hess = hessian(model, criterion=criterion, data=(sharp_x, sharp_y), cuda=True)
            top_eigenvalue = hess.eigenvalues(maxIter=400, top_n=1)[0][0]
            lambdas.append(top_eigenvalue)
            if verbose:
                print(f"Iteration {iteration}, Top Eigenvalue: {top_eigenvalue}")
    return train_loss / len(train_dl.dataset), lambdas

def train_model(model, width, criterion, optimizer, lr, classes, subset, epochs, batch_size, device, hessian_eval_per_epoch, log_epochs, verbose, verbose_epoch, seed):
    train_dl, _ = get_cifar(batch_size=batch_size, num_classes=classes, subset=subset, MSE=(str(criterion) == "MSELoss()"), on_gpu=True, device=device)
    sharp_x, sharp_y = next(iter(train_dl))
    hessian_iter = int(len(train_dl) / hessian_eval_per_epoch)
    print(f"Batches per epoch: {len(train_dl)}")
    print(f"Number of iterations for hessian eval: {hessian_iter}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model(width, classes).to(device)
    model_name = model.get_name()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0)

    lambdas_dict = {}
    losses_dict = {}
    lambdas = []
    losses = []

    for epoch in range(epochs):
        train_loss, top_eigenvalue = train_epoch_eigen(model, train_dl, criterion, optimizer, device, sharp_x, sharp_y, hessian_iter=hessian_iter, verbose=verbose)
        losses.append(train_loss)
        lambdas += top_eigenvalue

        if math.isnan(train_loss) or math.isnan(top_eigenvalue[-1]):
            print(f"NaN detected at epoch {epoch} for lr {lr} and width {width}")
            # losses += [float("nan")] * (epochs - len(losses)) 
            # lambdas += [float("nan")] * int((epochs * (len(train_dl) / hessian_eval_per_epoch)) - len(lambdas))
            break

        if (epoch % log_epochs == 0 or epoch == epochs - 1) and verbose_epoch:
            print(f"Epoch {epoch}, Top Eigenvalue: {top_eigenvalue[0]:.5f}, Loss: {train_loss:.5f}, lr: {lr}, width: {width}, #lambdas: {len(lambdas)}")

    lambdas_dict[(model_name, classes, subset, criterion, lr, width)] = lambdas
    losses_dict[(model_name, classes, subset, criterion, lr, width)] = losses

    df_lambdas = pd.DataFrame(lambdas_dict)
    df_losses = pd.DataFrame(losses_dict)
    df = pd.concat([df_lambdas, df_losses], axis=1, keys=["lambdas", "losses"])

    df.to_csv(f"/home/Mikolaj/mup-repo/results_multi/{model_name}_{criterion}_{classes}_{subset}_{epochs}_{width}_{lr}_not.csv", index=False)

def train_toy_cifar(model, width, criterion, lr, epochs, device, seed, sam, sam_rho, multiplier, hessian_iter=100, validation=False):
    torch.set_default_dtype(torch.float64)

    train_dl = get_cifar_toy(device=device)
    sharp_x, sharp_y = next(iter(train_dl))
    if validation:
        val_dl = get_cifar_toy(device=device, validation=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model(width, 1, multiplier=multiplier).to(device)
    model_name = model.get_name()
    if sam:
        optimizer = SAM(model.get_parameter_groups(lr, "SGD"), torch.optim.SGD, lr=lr, rho=sam_rho, momentum=0, weight_decay=0)
    else:
        optimizer = torch.optim.SGD(model.get_parameter_groups(lr, "SGD"), lr=lr, momentum=0, weight_decay=0)

    if validation:
        df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([("losses", model_name, 2, 0.005, criterion, lr, width, multiplier, sam, sam_rho), ("val", model_name, 2, 0.005, criterion, lr, width, multiplier, sam, sam_rho)]))
    else:
        df = pd.DataFrame(columns=pd.MultiIndex.from_tuples([("losses", model_name, 2, 0.005, criterion, lr, width, multiplier, sam, sam_rho), ("lambdas", model_name, 2, 0.005, criterion, lr, width, multiplier, sam, sam_rho)]))

    for epoch in range(epochs):
        xb, yb = next(iter(train_dl))
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        if sam:
            optimizer.first_step(zero_grad=True)
            loss_adv = criterion(model(xb), yb)
            loss_adv.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()
        if (epoch % hessian_iter == 0 or epoch == epochs - 1) and not validation:
            if multiplier:
                hess = hessian(model, criterion=criterion, data=(sharp_x, sharp_y), cuda=True)
            else:
                hess = hessian(model, criterion=criterion, data=(sharp_x, sharp_y), cuda=True, base_lr=lr, layer_lrs=model.layer_lrs)
            top_eigenvalue = hess.eigenvalues(maxIter=400, top_n=1)[0][0]
            print(f"{model_name}: Epoch {epoch}, Top Eigenvalue: {top_eigenvalue:.5f}, Loss: {loss.item():.5f}, lr: {lr}, width: {width}, sam: {sam}, sam_rho: {sam_rho}, multiplier: {multiplier}, device: {device}, seed: {seed}")
            df.loc[epoch] = [loss.item(), top_eigenvalue]
        elif not validation:
            df.loc[epoch] = [loss.item(), -2137.0]
        if validation:
            xb_val, yb_val = next(iter(val_dl))
            val_loss = criterion(model(xb_val), yb_val)
            df.loc[epoch] = [loss.item(), val_loss.item()]

    if validation:
        df.to_csv(f"/home/Mikolaj/mup-repo/results_multi/{model_name}_{criterion}_2_0.005_{epochs}_{width}_{lr}_toy_{sam}_{sam_rho}_{multiplier}_val.csv", index=False)
    else:
        df.to_csv(f"/home/Mikolaj/mup-repo/results_multi/{model_name}_{criterion}_2_0.005_{epochs}_{width}_{lr}_toy_{sam}_{sam_rho}_{multiplier}_{seed}.csv", index=False)


if __name__ == "__main__":
    # parameters from cli
    import argparse
    parser = argparse.ArgumentParser(description="Train a model and compute eigenvalues.")
    parser.add_argument("--model", type=str, default="mup", choices=MODELS.keys())
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--criterion", type=str, default="MSE", choices=CRITERIONS.keys())
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--classes", type=int, default=2)
    parser.add_argument("--subset", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="0") # pyhessian only supports "cuda", so below is export of CUDA_VISIBLE_DEVICES that cuda infers
    parser.add_argument("--hessian_evals_per_epoch", type=int, default=2)
    parser.add_argument("--log_epochs", type=int, default=10)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--verbose_epoch", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--toy_cifar", type=str, default="False")
    parser.add_argument("--multiplier", type=str, default="True")
    parser.add_argument("--sam", type=str, default="False")
    parser.add_argument("--sam_rho", type=float, default=0.05)
    parser.add_argument("--hessian_iter", type=int, default=100)
    parser.add_argument("--validation", type=str, default="False")
    args = parser.parse_args()
    print(args)

    args.toy_cifar = eval(args.toy_cifar)
    args.multiplier = eval(args.multiplier)
    args.sam = eval(args.sam)
    args.validation = eval(args.validation)

    if args.multiplier:
        from pyhessian import hessian
    else:
        from stability.reparamhessian import ReparamHessian as hessian

    if args.sam:
        from train_mlp import SAM
    else:
        args.sam_rho = 0.0

    model = MODELS[args.model]
    criterion = CRITERIONS[args.criterion]
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = "cuda"

    print(f"Model: {model}, criterion: {criterion}")

    if args.toy_cifar:
        train_toy_cifar(
            model=model,
            width=args.width,
            criterion=criterion,
            lr=args.lr,
            epochs=args.epochs,
            device=device,
            seed=args.seed,
            sam=args.sam,
            sam_rho=args.sam_rho,
            multiplier=args.multiplier,
            hessian_iter=args.hessian_iter,
            validation=args.validation
        )
    else:
        train_model(
            model=model,
            width=args.width,
            criterion=criterion,
            optimizer=optimizer,
            lr=args.lr,
            classes=args.classes,
            subset=args.subset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            hessian_eval_per_epoch=args.hessian_evals_per_epoch,
            log_epochs=args.log_epochs,
            verbose=args.verbose,
            verbose_epoch=args.verbose_epoch,
            seed=args.seed
        )
