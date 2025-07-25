import torch 
from matplotlib import pyplot as plt

from borc3.problem import Problem 
from borc3.surrogate import Surrogate
from borc3.acquisition import Acquisition
from borc3.bayesopt import Borc

# Author: James Whiteley (github.com/jamesalexwhiteley)

plt.rcParams['font.size'] = 9 

def plot1d(problem, borc):

    fig = plt.figure(figsize=(8, 6))

    # underlying function 
    fig.add_subplot(4, 1, 1)
    x = torch.linspace(problem.param_bounds["x"][0], problem.param_bounds["x"][1], 1000).unsqueeze(1)
    y = (model(x), model.f())[1]
    plt.plot(x, y, color='b', linewidth=1)
    plt.ylabel('f(x)') 
    plt.axhline(y=15, color='grey', linestyle='-.', label='Constraint')

    # objective gp
    x = x.squeeze(1)
    pred = borc.surrogate.predict_objectives(x.unsqueeze(-1), return_std=True, grad=False)[0]
    mu, std = pred.mu, pred.std
    low, high = mu - 2 * std, mu + 2 * std 
    plt.plot(x, mu, label='GP posterior mean', color='b')
    plt.fill_between(x, low, high, where=(high > low), interpolate=True, color='b', alpha=0.15, label=r'GP posterior 95% bounds')
    # sampled points 
    gp = borc.surrogate.objective_gps[0]
    train_x, train_y = gp.get_training_data()
    plt.scatter(train_x.flatten(), train_y.flatten(), color='k', label='{} training points'.format(len(train_x)), marker='o', s=35)
    plt.legend(loc=0)
    
    # EI acquisition function 
    fig.add_subplot(4, 1, 2)    
    a = borc.eval_acqf(x.unsqueeze(1)).detach()
    plt.plot(x.detach().cpu(), a.cpu(), label='EI Acquisition function', color='k')
    plt.legend(loc=0)

    # PF acquisition function 
    fig.add_subplot(4, 1, 3) 
    a = borc.eval_acqg(x.unsqueeze(1)).detach()
    plt.plot(x.detach().cpu(), a.cpu(), label='PF Acquisition function', color='k')
    plt.legend(loc=0)

    # EI*PF Acquisition function 
    fig.add_subplot(4, 1, 4)
    a = borc.eval_acquisition(x.unsqueeze(1)).detach()
    plt.plot(x, a, label='EI*PF Acquisition function', color='k')
    plt.axvline(x=new_x, color='m', linestyle='--')
    plt.xlabel('x')
    plt.legend(loc=0)

    plt.show()

class Model():
    def __call__(self, x):
        self.m = x**3 + torch.sin(2 * x) + 10 

    def f(self):
        return self.m.flatten()
    
    def g(self):
        return self.m.flatten() / 15 - 1 # f(x)<15 <=> g(x)<0

if __name__ == "__main__": 

    problem = Problem()
    model = Model()
    bounds = {"x": (-3, 3)}

    problem.set_bounds(bounds)
    problem.add_model(model)
    problem.add_objectives([model.f])
    problem.add_constraints([model.g])

    surrogate = Surrogate(problem) 
    acquisition = Acquisition(f="EI", g="PF") 
    borc = Borc(surrogate, acquisition) 
    borc.initialize(nsamples=4) 

    iters = 2
    for i in range(iters): 
        print(f"Iter: {i + 1}/{iters} | Max Objective: {borc.surrogate.fbest.detach().numpy()},  Optimal x : {borc.surrogate.xbest.detach().numpy()}")  
        new_x, max_acq = borc.batch_optimize_acq(iters=100)   
        plot1d(problem, borc) 
        borc.step(new_x=new_x) 
    
    borc.acquisition = Acquisition(f="MU", g="PF") 
    new_x, max_acq = borc.constrained_optimize_acq(iters=100) 
    plot1d(problem, borc) 
