"""
This script is used to solve batched linear regression with L1 norm:
                      min | Ax - b |_1
where A and b are respectively independent and dependent variables.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings


class L1Minimizer(nn.Module):
    def __init__(self, y, X, device):
        super(L1Minimizer, self).__init__()
        self.y, self.X = y.to(device), X.to(device)
        self.beta = nn.Parameter(torch.ones((self.y.shape[0], self.X.shape[-1])).to(device).double())
        self.criterion = nn.L1Loss()

    def forward(self):
        return self.criterion(torch.bmm(self.X, self.beta.unsqueeze(2)).squeeze(), self.y)


def solve(y, X, device='cuda:0', lr=1e-1, max_iter=1000, tol=1e-5, verbose=-1):
    solver = L1Minimizer(y=torch.as_tensor(y), X=torch.as_tensor(X), device=device)
    optimizer = Adam(solver.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, max_iter)
    best_beta, best_loss, prev_loss = None, float('inf'), float('inf')
    for iteration in tqdm(range(max_iter)):
        optimizer.zero_grad()
        loss = solver()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            best_beta, best_loss = solver.beta.data.detach().cpu().numpy(), loss.item()
        if abs(loss.item() - prev_loss) <= tol:
            break
        else:
            prev_loss = loss.item()
    if iteration + 1 == max_iter:
        warnings.warn('reach maximum iteration without convergence!')
    elif verbose > 0:
        print('Converged in {} iterations.'.format(iteration))

    return best_beta


##--------------------------------------------------------------------------------


if __name__ == '__main__':

    """ generate synthetic data """
    bs, n_lights = 200000, 100
    y, X, n_gt = [], [], []
    for _ in range(bs):
        normal_gt = np.random.rand(3) - 0.5
        normal_gt = normal_gt / np.linalg.norm(normal_gt)
        L = np.random.rand(n_lights, 3)
        L[:, 0], L[:, 1] = L[:, 0] - 0.5, L[:, 1] - 0.5
        m = L @ normal_gt + np.random.normal(0, 0.01, n_lights)
        m[m < 0] = 0
        y.append(m)
        X.append(L)
        n_gt.append(normal_gt)
    y, X, n_gt = np.asarray(y), np.asarray(X), np.asarray(n_gt)

    """ solve the batched censored regression problem """
    torch_res = solve(y, X, device='cuda:0', lr=1e-1, max_iter=1000, verbose=1)

