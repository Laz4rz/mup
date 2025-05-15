import math
import torch
import numpy as np

from pyhessian.utils import (
    group_product,
    group_add,
    normalization,
    get_params_grad,
    hessian_vector_product,
    orthnormal,
)


class ReparamHessian:
    """Compute the re‑parameterisation‑invariant Hessian (Eq. 10)."""
    def __init__(self, model, criterion, data=None, dataloader=None, cuda=True, base_lr=None, layer_lrs=None):
        assert base_lr is not None and layer_lrs is not None, "provide base_lr and layer_lrs"
        from pyhessian import hessian as _BaseHessian

        self._base = _BaseHessian(
            model=model,
            criterion=criterion,
            data=data,
            dataloader=dataloader,
            cuda=cuda,
        )

        self.model = self._base.model  
        self.device = self._base.device
        self.full_dataset = self._base.full_dataset
        self.params = self._base.params
        self.gradsH = self._base.gradsH
        self.base_lr = base_lr
        self.layer_lrs = layer_lrs
        self.scalings = [math.sqrt(lr) for lr in layer_lrs]

        self.layer_ids = self._assign_layers()
        assert (
            max(self.layer_ids) + 1 == len(self.layer_lrs)
        ), (
            f"Detected {max(self.layer_ids)+1} layers with parameters, "
            f"but PER_LAYER_LRS has length {len(self.layer_lrs)}."
        )

    def _assign_layers(self):
        """Return a list of layer indices matching `self.params`."""
        layer_ids = []
        current_layer = -1
        last_root = None
        for name, _ in self.model.named_parameters():
            root = name.split(".")[0]  # e.g. "layer1.weight" → "layer1"
            if root != last_root:
                current_layer += 1
                last_root = root
            layer_ids.append(current_layer)
        return layer_ids

    def _apply_D(self, vec):
        """Element‑wise multiply list `vec` by D."""
        return [self.scalings[self.layer_ids[i]] * v_i for i, v_i in enumerate(vec)]

    def _apply_D_over_eta(self, vec):
        """Multiply by D and divide by the global step‑size η."""
        return [
            self.scalings[self.layer_ids[i]] / self.base_lr * v_i
            for i, v_i in enumerate(vec)
        ]

    def _Hv_reparam(self, v):
        """Compute (η⁻¹ D H D) v using autograd for the inner Hessian‑vector product."""
        Dv = self._apply_D(v) 
        if self.full_dataset:
            _, HDv = self._base.dataloader_hv_product(Dv)
        else:
            HDv = hessian_vector_product(self.gradsH, self.params, Dv)
        return self._apply_D_over_eta(HDv)

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """Return the top *top_n* eigenvalues / eigenvectors of Ĥ."""
        assert top_n >= 1
        eigenvalues, eigenvectors = [], []
        computed_dim = 0
        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size(), device=self.device) for p in self.params]
            v = normalization(v)
            for _ in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()
                tmp_eigenvalue, Hv = self._dataloader_or_single_batch_Hv(v)
                v = normalization(Hv)
                if eigenvalue is None:
                    eigenvalue = tmp_eigenvalue
                elif abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                    break
                else:
                    eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1
        return eigenvalues, eigenvectors

    def _dataloader_or_single_batch_Hv(self, v):
        if self.full_dataset:
            Hv = self._Hv_reparam(v)
            ev = group_product(Hv, v).cpu().item()
            return ev, Hv
        else:
            Hv = self._Hv_reparam(v)
            ev = group_product(Hv, v).cpu().item()
            return ev, Hv

