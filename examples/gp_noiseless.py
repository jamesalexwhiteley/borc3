import torch
from matplotlib import pyplot as plt
from borc3.gp import NoiselessGP

# Author: James Whiteley (github.com/jamesalexwhiteley)

def plot1d(x, y, mu, low, high, train_x, train_y):
    # underlying function 
    plt.figure(figsize=(8, 4))
    x, mu = x.cpu(), mu.cpu()
    plt.plot(x, y, label='Underlying function', color='k', linewidth=1)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend() 
    # gp
    plt.plot(x, mu, label='GP posterior mean', color='b')
    plt.fill_between(x, low, high, where=(high > low), interpolate=True, color='b', alpha=0.15, label=r'GP posterior 95% bounds')
    # sampled points 
    plt.scatter(train_x, train_y, color='k', label='{} randomly sampled training points'.format(nsamples), marker='o', s=35)
    plt.legend(loc=0)
    plt.show()

def plot2d(x, y, mu, low, high, train_x, train_y):
    # underlying function 
    fig = plt.figure()
    shape = (steps, steps)
    ax = fig.add_subplot(111, projection='3d')
    y = y.reshape(shape)
    X1, X2 = x[:, 0].reshape(shape), x[:, 1].reshape(shape)
    ax.plot_surface(X1.numpy(), X2.numpy(), y.numpy(), cmap='viridis', alpha=0.5)
    ax.set_title('Himmelblau Function')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')
    # gp 
    mu = mu.reshape(X1.shape)
    ax.plot_surface(X1, X2, mu, cmap='cividis', alpha=0.75)
    # ax.plot_surface(X1, X2, low.reshape(X1.shape), alpha=0.45, color='lightgrey')
    # ax.plot_surface(X1, X2, high.reshape(X1.shape), alpha=0.45, color='lightgrey')
    ax.scatter(X1.ravel(), X2.ravel(), mu.ravel(), label='GP prediction', color='b', s=1)
    # sampled points 
    train_x1, train_x2 = train_x[:, 0], train_x[:, 1]
    ax.scatter(train_x1.ravel(), train_x2.ravel(), train_y.ravel(), label='{} randomly sampled training points'.format(nsamples), color='r', s=25)
    plt.legend(loc=0, fontsize=10)
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": 

    # 1d test function 
    def f(x):
        return x**2 + 3 * torch.sin(3 * x)
    x = torch.linspace(-3, 3, 100)
    y = f(x)
    nsamples = 6
    ind = torch.randperm(int(len(x)))[:nsamples]
    train_x, train_y = x[ind], y[ind]

    gp = NoiselessGP(train_x.unsqueeze(-1), train_y)
    gp.fit() 
    pred = gp.predict(x.unsqueeze(-1), return_std=True) 
    low, high = pred.mu - 2 * pred.std, pred.mu + 2 * pred.std 
    plot1d(x, y, pred.mu, low, high, train_x, train_y) 

    # 2d test function   
    def himmelblau(x, y): 
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    steps = 40
    x1 = torch.linspace(-6, 6, steps)
    x2 = torch.linspace(-6, 6, steps)
    X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
    x = torch.stack([X1.flatten(), X2.flatten()], dim=-1)
    y = himmelblau(x[:, 0], x[:, 1])
    nsamples = 20
    ind = torch.randperm(int(len(x)))[:nsamples]
    train_x, train_y = x[ind], y[ind]

    gp = NoiselessGP(train_x, train_y)
    gp.fit() 
    pred = gp.predict(x, return_std=True)
    low, high = pred.mu - 2 * pred.std, pred.mu + 2 * pred.std
    plot2d(x, y, pred.mu, low, high, train_x, train_y) 