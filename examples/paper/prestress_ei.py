import torch 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import math 

from borc3.problem import Problem 
from borc3.surrogate import Surrogate, SurrogateIO
from borc3.gp import VariationalHomoscedasticGP, HomoscedasticGP
from borc3.acquisition import Acquisition
from borc3.bayesopt import Borc
from borc3.probability import MultivariateNormal
from borc3.utilities import tic, toc 

from prestress_rs import Model, plotcontour # type:ignore 
from pystressed.servicability import plot_magnel, optimize_magnel, optimize_and_plot_magnel 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# Author: James Whiteley (github.com/jamesalexwhiteley)

def bayesopt(ninitial, iters, n): 

    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    output_dir  = os.path.join(base_folder, "prestress_initial")

    problem = Problem()
    model = Model()
    bounds = {"P": (2.0, 15.0), "e": (-0.45, 0.45), 'd': (0.5, 1.5)} # for bayesopt 

    # Uncertain parameters: ground stiffness for two pile groups
    mu = torch.tensor([100.0, 100.0])                   # k0_1, k0_2 [kN/mm]                
    cov = torch.tensor([[    (20)**2,  0.5*(20)**2],    # COV = 30%, correlation = 0.5              
                        [0.5*(20)**2,      (20)**2]])
    dist = MultivariateNormal(mu, cov) 

    problem.set_bounds(bounds) 
    problem.set_dist(dist) 
    problem.add_model(model) 
    problem.add_objectives([model.f]) 
    problem.add_constraints([model.g1, model.g2, model.g3, model.g4]) # Transfer state  
    problem.add_constraints([model.g5, model.g6, model.g7, model.g8]) # Service state

    xi = problem.sample_xi(nsamples=int(1e3)).to(device)
    surrogate = Surrogate(problem, gp=HomoscedasticGP, ntraining=ninitial, nstarts=5) 
    acquisition = Acquisition(f="eMU", g="ePF", xi=xi, eps=0.01) 
    borc = Borc(surrogate, acquisition) 
    borc.cuda(device) 
    borc.initialize(nsamples=ninitial, sample_method="lhs", max_acq=torch.tensor([0.0])) 
    # borc.surrogate = SurrogateIO.load(output_dir) 

    # Monte Carlo solution 
    P_lower, P_upper = list(problem.param_bounds.values())[0]
    e_lower, e_upper = list(problem.param_bounds.values())[1]
    d_lower, d_upper = list(problem.param_bounds.values())[2]
    params=(torch.linspace(P_lower, P_upper, steps=20), torch.linspace(e_lower, e_upper, steps=4), torch.linspace(d_lower, d_upper, steps=4)) 

    # BayesOpt used to sequentially sample [x,xi] points 
    res = torch.ones(iters, ) 
    for i in range(iters): 

        # argmax_x E[f(x,xi)] s.t. P[g_i(x,xi)<0]>1-β, i=1,2...,m
        if i % n == 0: 
            xopt, _ = borc.surrogate.monte_carlo(params=params, nsamples=int(1e2), obj_type="mean", con_type="prob", con_eps=0.01, output=False)     
            problem.model(torch.cat([xopt, problem.sample_xi(nsamples=1).to(device)], dim=1)) # true E[f(x,xi)] = f(x) is simply determinisitc 
            res[i] = problem.objectives() 
            print(f"Max Objective: {res[i].item():.4f} | Optimal x : {xopt}") 

        # fbest = max_[x,xi] mu 
        borc.acquisition = Acquisition(f="MU") 
        _, borc.fbest = borc.batch_optimize_acq(iters=100, nstarts=5) 

        # new_x = argmax_[x,xi] EI x PF 
        borc.acquisition = Acquisition(f="EI", g="PF", xi=xi, eps=1.0) 
        new_x, _ = borc.batch_optimize_acq(iters=100, nstarts=5) 
        borc.step(new_x=new_x) 
        # print(f"new_x : {new_x}") 

    return xopt, res 

if __name__ == "__main__": 

    ninitial, iters, n = 10, 10, 5 
    xopt, res = bayesopt(ninitial, iters, n) 