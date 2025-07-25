import torch 
from matplotlib import pyplot as plt
import os 

from borc3.problem import Problem 
from borc3.surrogate import Surrogate, SurrogateIO
from borc3.acquisition import Acquisition
from borc3.bayesopt import Borc

# Author: James Whiteley (github.com/jamesalexwhiteley)

def plot2d(problem, borc):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    # # underlying function 
    steps = 40
    shape = (steps, steps)
    x1 = torch.linspace(problem.param_bounds["x1"][0], problem.param_bounds["x1"][1], steps)
    x2 = torch.linspace(problem.param_bounds["x2"][0], problem.param_bounds["x2"][1], steps)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    x = torch.stack([X1.flatten(), X2.flatten()], dim=-1)
    y = (problem.model(x), problem.objectives().flatten())[1].reshape(shape)
    ax.plot_surface(X1.numpy(), X2.numpy(), y.numpy(), cmap='viridis', alpha=0.5)
    ax.set_title('Himmelblau Function and GP')

    # gp 
    pred = borc.surrogate.predict_objectives(x, return_std=True, grad=False)[0]
    mu, std = pred.mu, pred.std 
    low, high = mu - 2 * std, mu + 2 * std 
    mu = mu.reshape(X1.shape)
    ax.plot_surface(X1, X2, mu, cmap='cividis', alpha=0.75)
    # ax.plot_surface(X1, X2, low.reshape(X1.shape), alpha=0.45, color='lightgrey')
    # ax.plot_surface(X1, X2, high.reshape(X1.shape), alpha=0.45, color='lightgrey')
    ax.scatter(X1.ravel(), X2.ravel(), mu.ravel(), label='GP prediction', color='b', s=1)
    # sampled points 
    gp = borc.surrogate.objective_gps[0]
    train_x, train_y = gp.get_training_data()
    train_x1, train_x2 = train_x[:, 0], train_x[:, 1]
    ax.scatter(train_x1.ravel(), train_x2.ravel(), train_y.ravel(), label='{} training points'.format(len(train_x)), color='r', s=25)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.legend(loc=0, fontsize=9)
    
    # acquisition function 
    ax = fig.add_subplot(1, 2, 2, projection='3d') 
    a = borc.eval_acquisition(x).detach().reshape(shape)
    ax.plot_surface(X1, X2, a, cmap='cividis', alpha=0.75)
    ax.scatter(new_x[0][0], new_x[0][1], max_acq, label='Max acquisition found', color='m', s=30, marker='D')
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_title('Acquisition Function')
    plt.legend(loc=0, fontsize=9)
    plt.show()

class Model():
    def __call__(self, x): # himmelblau function
        x1, x2 = x.T
        self.m = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2

    def f(self):
        return -self.m # min f, so negative  

if __name__ == "__main__": 

    base_folder = os.path.join(os.getcwd(), "models")
    os.makedirs(base_folder, exist_ok=True)
    output_dir  = os.path.join(base_folder, "borc_2d")

    problem = Problem()
    model = Model()

    bounds= {
        "x1": (-6, 6),
        "x2": (-6, 6),
        }

    problem.set_bounds(bounds)
    problem.add_model(model)
    problem.add_objectives([model.f])

    surrogate = Surrogate(problem) 
    acquisition = Acquisition(f="EI") 
    borc = Borc(surrogate, acquisition) 
    borc.initialize(nsamples=100) 

    SurrogateIO.save(borc.surrogate, output_dir) 
    borc.surrogate = SurrogateIO.load(output_dir) 

    iters = 2 
    for i in range(iters): 
        print(f"Iter: {i + 1}/{iters} | Max Objective: {borc.surrogate.fbest.detach().numpy()},  Optimal x : {borc.surrogate.xbest.detach().numpy()}") 
        new_x, max_acq = borc.batch_optimize_acq(iters=100, nstarts=10) 
        plot2d(surrogate.problem, borc)
        borc.step()