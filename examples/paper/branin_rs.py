import torch 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import math 

from borc3.problem import Problem 
from borc3.surrogate import Surrogate, SurrogateIO
from borc3.acquisition import Acquisition
from borc3.bayesopt import Borc
from borc3.probability import DiscreteJoint
from borc3.utilities import tic, toc 

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Author: James Whiteley (github.com/jamesalexwhiteley)

def plotcontour(problem, borc):

    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'contour_branin.png')

    fig = plt.figure(figsize=(7, 6))

    # # ground truth 
    # steps = 1000
    # x = torch.linspace(0, 1, steps)
    # y = torch.linspace(0, 1, steps)
    # X, Y = torch.meshgrid(x, y, indexing='ij')
    # xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    # mu, prob = problem.rbo(xpts, nsamples=int(5e2), output=False, return_vals=True)
    # MU = mu.view(X.shape).detach()
    # PI = prob[0].view(X.shape).detach()

    # surrogate
    steps = 5
    x = torch.linspace(0, 1, steps)
    y = torch.linspace(0, 1, steps)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    xpts = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).to(device)

    tic()
    # list comprehension 
    mu, prob = zip(*[borc.rbo(x.unsqueeze(0), nsamples=int(2e2), output=False, return_vals=True) for x in xpts]) 
    MU = torch.tensor(mu).view(X.shape).detach()
    PI = torch.tensor(prob).view(X.shape).detach()
    # vector input 
    # mu, prob = borc.rbo(xpts, nsamples=int(2e2), output=False, return_vals=True) 
    # MU = mu.view(X.shape).detach().cpu()
    # PI = prob[0].view(X.shape).detach().cpu()
    toc()

    proxy = Line2D([0], [0], color='black', lw=1.5, label=r'\text{P}$[g(x,\xi)<0] = 1-\epsilon$')
    contour_mu = plt.contourf(X.numpy(), Y.numpy(), MU.numpy(), cmap='PuBu')
    contour_pi = plt.contour(X.numpy(), Y.numpy(), PI.numpy(), colors='black', linewidths=1, levels=torch.linspace(0.1, 0.9, 5))
    plt.clabel(contour_pi, inline=True, fontsize=8)
    plt.colorbar(contour_mu, shrink=0.8, pad=0.05)

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.legend([scatter, proxy], ['Optimal x', r'$\text{P}[\text{g}(x,\xi)\leq 0]$'], loc="lower left")
    plt.legend([proxy], [r'$\text{P}[\text{g}(x,\xi)\leq 0]$'], loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()

def branin_williams(x):
    """
    Williams, Brian Jonathan. 
    Sequential design of computer experiments to minimize integrated response functions. 
    Ohio State University, 2000.
    
    Parameters
    ----------
    x : torch.Tensor, shape=(4, n) 

    """
    def yb(u, v):
        return (v - 5.1 / (4 * math.pi ** 2) * u ** 2 + 5 / math.pi * u - 6) ** 2 + 10 * (1 - 1 / (8 * math.pi)) * torch.cos(u) + 10

    u1 = 15 * x[0, :] - 5
    v1 = 15 * x[1, :]
    u2 = 15 * x[2, :] - 5
    v2 = 15 * x[3, :]
    
    return yb(u1, v1) * yb(u2, v2)

class Model():
    def __call__(self, x): 
        self.x = x
        self.m = None

    def f(self):    
        return branin_williams(self.x[:, [0, 2, 3, 1]].T)
    
    def g(self):
        return torch.linalg.vector_norm(self.x, dim=1) - torch.tensor(3/2).sqrt()
    
def bayesopt(ninitial, iters, n):

    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    output_dir  = os.path.join(base_folder, "branin_rs")

    problem = Problem()
    model = Model()
    bounds = {"x1": (0, 1), "x4": (0, 1)}

    joint = torch.tensor([
        [0.0375, 0.0875, 0.0875, 0.0375],  # P(x2=0.25, x3=0.2), P(x2=0.25, x3=0.4), P(x2=0.25, x3=0.6), P(x2=0.25, x3=0.8) 
        [0.0750, 0.1750, 0.1750, 0.0750],  # P(x2=0.5, x3=0.2) ...
        [0.0375, 0.0875, 0.0875, 0.0375],  # P(x2=0.75, x3=0.2) ...
        ])
    
    x2_values = torch.tensor([0.25, 0.5, 0.75])
    x3_values = torch.tensor([0.2, 0.4, 0.6, 0.8])  
    dist = DiscreteJoint(joint, x2_values, x3_values)

    problem.set_bounds(bounds)
    problem.set_dist(dist)
    problem.add_model(model)
    problem.add_objectives([model.f])
    problem.add_constraints([model.g])
 
    xi = problem.sample_xi(nsamples=int(1e2)).to(device)
    surrogate = Surrogate(problem)
    acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.1)
    borc = Borc(surrogate, acquisition) 
    borc.cuda(device) 
    borc.initialize(nsamples=ninitial, sample_method="lhs", max_acq=torch.tensor([0.0])) 
    SurrogateIO.save(borc.surrogate, output_dir) 
    # borc.surrogate = SurrogateIO.load(output_dir) 

    # params=(torch.linspace(0.0, 1.0, steps=101), torch.linspace(0.0, 1.0, steps=101)) 
    # xopt, _ = problem.monte_carlo(params=params, nsamples=int(5e2), obj_type="mean", con_type="prob", con_eps=0.1) # [0, 0.75] 8336.85
    # _, _ = problem.rbo(xopt, nsamples=int(1e3), return_vals=True) 
    # plotcontour(problem, borc)

    # BayesOpt used to sequentially sample [x,xi] points 
    res = torch.ones(iters, ) 
    for i in range(iters): 

        # new_[x,xi] <- random search 
        borc.step(new_x=problem.sample()) 

        # argmax_x E[f(x,xi)] s.t. P[g(x,xi)<0]>1-epsilons
        if i % n == 0: 
            xopt, _ = borc.constrained_optimize_acq(iters=int(1e2), nstarts=4, optimize_x=True) 
            res[i], _ = problem.rbo(xopt, output=False, return_vals=True) # true E[f(x,xi)] 
            print(f"Max Objective: {res[i].item():.4f} | Optimal x : {xopt}") 

    return xopt, res 

if __name__ == "__main__": 
    ninitial, iters, n = 100, 10, 1 
    xopt, res = bayesopt(ninitial, iters, n) 