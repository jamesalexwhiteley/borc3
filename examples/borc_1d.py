import torch 
from matplotlib import pyplot as plt
import numpy as np
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

from borc3.surrogate import Surrogate
from borc3.problem import Problem 
from borc3.acquisition import Acquisition
from borc3.bayesopt import Borc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

def plot1d(problem, borc):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    fig.tight_layout(pad=3.0) 

    # fig.suptitle('Gaussian Process Regression and Acquisition Function', fontsize=18, y=0.95)
    plt.subplots_adjust(top=0.92)  

    x_tensor = torch.linspace(problem.param_bounds["x"][0], problem.param_bounds["x"][1], 1000).unsqueeze(1)
    y = (model(x_tensor), model.f())[1]

    x = x_tensor.squeeze(1).cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    ax1.plot(x, y, label='Objective Function', color='k', linestyle='-', linewidth=1)

    # GP Predictions
    pred = borc.surrogate.predict_objectives(x_tensor.to(device), return_std=True, grad=False)[0]
    pred.cuda('cpu')
    mu, std = pred.mu, pred.std
    low, high = mu - 2 * std, mu + 2 * std 

    ax1.plot(x, mu, color='b', linestyle='-', linewidth=2, label=r'GP $\mu$', alpha=0.6)
    ax1.fill_between(x, low, high, where=(high > low), interpolate=True, color='b', alpha=0.05, label=r'GP $\mu \pm 2\sigma$')
    # ax1.plot(x, low, color='b', linestyle='--', linewidth=1)
    # ax1.plot(x, high, color='b', linestyle='--', linewidth=1)

    # training points
    gp = borc.surrogate.objective_gps[0]
    train_x, train_y = gp.get_training_data(device='cpu')
    ax1.scatter(train_x.flatten(), train_y.flatten(), label=f'{len(train_x)} Training Points', s=75, marker='.', color='red', edgecolors='black', zorder=5)
    ax1.set_ylabel(r'$f(x)$', fontsize=14)
    # ax1.set_title('Gaussian Process', fontsize=16)
    ax1.legend(loc='upper left', frameon=True, shadow=True)

    # Acquisition function 
    a = borc.eval_acquisition(x_tensor.to(device)).detach().cpu().numpy()
    max_acq = a.max()
    new_x_index = np.argmax(a)
    new_x = x[new_x_index]
    ax2.plot(x, a, label='Acquisition Function', color='black', linestyle='-', linewidth=2)
    ax2.scatter(new_x, max_acq, color='magenta', label='Max Acquisition', s=100, marker='*', edgecolors='k', zorder=10)
    ax2.axvline(new_x, color='gray', linestyle='--', linewidth=1, label='Selected Point')

    ax2.set_xlabel(r'$x$', fontsize=14)
    ax2.set_ylabel(r'$\alpha(x)$', fontsize=14)
    # ax2.set_title('Acquisition Function', fontsize=16)
    ax2.legend(loc='upper left', frameon=True, shadow=True)

    # maximum acquisition point
    ax2.annotate('Maximum Acquisition',
                xy=(new_x, max_acq),
                xytext=(new_x + (x[-1]-x[0])*0.05, max_acq + (a.max()-a.min())*0.1),
                arrowprops=dict(facecolor='magenta', arrowstyle='->', linewidth=2),
                fontsize=12,
                color='magenta')

    plt.savefig('bayesopt.png', dpi=600, bbox_inches='tight')  
    plt.show()


class Model():
    def __call__(self, x):
        self.m = x + x**2 + 4*torch.sin(3/2 * x)

    def f(self):
        return self.m.flatten() 

if __name__ == "__main__": 

    problem = Problem()
    model = Model()
    bounds = {"x": (-3, 3)}
    problem.set_bounds(bounds)
    problem.add_model(model)
    problem.add_objectives([model.f])
    
    surrogate = Surrogate(problem)
    acquisition = Acquisition(f="PI")
    borc = Borc(surrogate, acquisition) 
    borc.cuda(device)
    borc.initialize(nsamples=4, sample_method="lhs") 

    iters = 2 
    for i in range(iters): 
        print(f"Iter: {i + 1}/{iters} | Max Objective: {borc.surrogate.fbest.cpu()},  Optimal x : {borc.surrogate.xbest.cpu()}") 
        new_x, max_acq = borc.batch_optimize_acq(iters=20, nstarts=10) 
        plot1d(problem, borc) 
        borc.step(new_x=new_x) 
       
