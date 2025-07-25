import torch 
from torch.distributions import Normal 
from borc3.utilities import to_device, gen_batch_data

# Author: James Whiteley (github.com/jamesalexwhiteley)

class Acquisition():
    def __init__(self, 
                 f="EI", 
                 g="None",
                 **kwargs):
        """
        Parameters
        ----------
        f : string 
            keyword for the acquisition function to use for the objectives 
        g : string 
            keyword for the acquisition function to use for the constraints 

        """
        self.name = f
        options = {
            # standard acquisition functions
            'MU' : self.posterior_mean, 
            'UCB' : self.upper_confidence_bound,
            'EI' : self.expected_improvement,
            'PI': self.probability_of_improvement, 
            'E' : self.entropy,
            'EF' : self.expected_feasibility,
            'PF': self.probability_of_feasibility,
            'None' : self.identity, 
            # acquisition functions with environmental variables, i.e., E[f(x,xi)]
            'eMU' : self.posterior_mean_monte_carlo,
            'eUCB' : self.upper_confidence_bound_monte_carlo, 
            'eEI' : self.expected_improvement_monte_carlo,
            'ePF' : self.probability_of_feasibility_monte_carlo,
            'eMSE' : self.mean_squared_error,
            'eWMSE' : self.weighted_mean_squared_error, 
        } 

        # default params 
        self.k = 10
        self.eps = 1.0
        self.efs = 1.0 
        self.beta = 1.5
        self.big = 1e6
        self.aleatoric = False 

        self.normal = Normal(0, 1)
        self.min = 1e-6 # prevent divide by zero 
        self.f = options[f]
        self.g = options[g]
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    def cuda(self, device):
        for attr, value in self.__dict__.items():
            setattr(self, attr, to_device(value, device))
        self.device = device 

    def gp_predict(self, x, gp):
        """ 
        Allow std_aleatoric to be used if self.aleatoric == True   
        
        """
        pred = gp.predict(x, return_std=True, grad=True)

        if self.aleatoric:
            mu, std = pred.mu, pred.std_aleatoric
        else: 
            mu, std = pred.mu, pred.std_epistemic

        return mu, torch.clamp(std, min=self.min) 
    
    # ================================= standard acquisiton functions ================================= #
    def posterior_mean(self, x, gp, fbest):
        """
        μ(x)
        
        """
        mu, _ = self.gp_predict(x, gp)
        return mu 
    
    def upper_confidence_bound(self, x, gp, fbest):
        """
        μ(x) + βσ(x)

        """
        mu, std = self.gp_predict(x, gp)
        return mu + self.beta * std

    def expected_improvement(self, x, gp, fbest):
        """
        E[max(f(x)-f*, 0)]

        """
        mu, std = self.gp_predict(x, gp) 
        z = (mu - fbest) / torch.clamp(std, min=self.min) 
        return (mu - fbest) * self.normal.cdf(z) + std * torch.exp(self.normal.log_prob(z))
    
    def probability_of_improvement(self, x, gp, fbest):
        """
        P[f(x)>f*]

        """
        mu, std = self.gp_predict(x, gp)
        z = (mu - fbest) / torch.clamp(std, min=self.min) 
        return self.normal.cdf(z)
    
    def entropy(self, x, gp, fbest): 
        """
        H(x) = .5 * log(2 * pi * e * std**2)

        """
        _, std = self.gp_predict(x, gp)
        var = std**2
        return 0.5 * torch.log(2 * torch.pi * torch.e * var)
    
    def expected_feasibility(self, x, gp, fbest):
        """
        E[g(x)=0]

        See Bichon et al. "Efficient global reliability analysis for nonlinear implicit performance functions." AIAA. 2008

        """ 
        mu, std = self.gp_predict(x, gp)
        eps = self.efs * std 
        z0, z1, z2 = -mu / std, -(eps + mu) / std, (eps - mu) / std 
        a = 2 * self.normal.cdf(z0) - self.normal.cdf(z1) - self.normal.cdf(z2) 
        b = 2 * torch.exp(self.normal.log_prob(z0)) - torch.exp(self.normal.log_prob(z1)) - torch.exp(self.normal.log_prob(z2))
        c = self.normal.cdf(z2) - self.normal.cdf(z1)
        return -(mu*a + std*b + eps*c)
        
    def probability_of_feasibility(self, x, gp, fbest):
        """
        P[g(x)<0]

        """
        mu, std = self.gp_predict(x, gp)
        z = (0.0 - mu) / torch.clamp(std, min=self.min) 
        return self.normal.cdf(z) - 1 + self.eps
    
    def identity(self, x, gp, fbest):
        """
        1

        """
        return torch.tensor([1.0]) 
    
    # ================================= # acquisiton functions with environmental variables, i.e., E[f(x,xi)] ================================= #   
    def posterior_mean_monte_carlo(self, x, gp, fbest): # eMU
        """
        μ(x) = E[f(x,xi)]
        
        """
        batch_x = gen_batch_data(x, self.xi) 
        mu, _ = self.gp_predict(batch_x, gp) 
        return mu.mean(dim=1) 
    
    def upper_confidence_bound_monte_carlo(self, x, gp, fbest): # eUCB
        """
        μ(x) + βσ(x), where 
        μ(x) = E[f(x,xi)], σ^2(x) = Var[f(x,xi)]

        """
        batch_x = gen_batch_data(x, self.xi)
        pred = gp.predict(batch_x, return_std=True, grad=True) 
        mu, std = pred.mu.mean(dim=1), pred.std.mean(dim=1)  
        return mu + self.beta * std 

    def expected_improvement_monte_carlo(self, x, gp, fbest): # eEI
        """
        E[max(μ(x)-μ*, 0)]

        See Arendt et al. "Objective-Oriented Sequential Sampling for Simulation Based Robust Design Considering Multiple Sources of Uncertainty." J Mec Des. 2013.

        """
        batch_x = gen_batch_data(x, self.xi) 
        pred = gp.predict(batch_x, return_std=True, grad=True) 
        mu, std = pred.mu.mean(dim=1), pred.std.mean(dim=1) 
        z = (mu - fbest) / torch.clamp(std, min=self.min) 
        return (mu - fbest) * self.normal.cdf(z) + std * torch.exp(self.normal.log_prob(z)) 

    def probability_of_feasibility_monte_carlo(self, x, gp, fbest): # ePF
        """
        P[g(x,xi)<0]  = '\'int Phi(-mu/std) p(xi)dxis

        """
        batch_x = gen_batch_data(x, self.xi)
        pred = gp.predict(batch_x, return_std=True, grad=True)         
        mu, std = pred.posterior() 
        p = torch.distributions.Normal(mu, std).cdf(torch.tensor([0.0]).to(mu.device)) 
        return p.mean(dim=1) - 1 + self.eps 
    
    def mean_squared_error(self, xi, gp, fbest): # eMSE
        """
        σ^2(xi) 
        
        """       
        eval_x = gen_batch_data(self.x, xi).squeeze(0)
        return gp.predict(eval_x, return_std=True, grad=True).std**2
    
    def weighted_mean_squared_error(self, xi, gp, fbest): # eWMSE
        """ 
        p(xi) σ^2(xi) 
        
        """ 
        eval_x = gen_batch_data(self.x, xi).squeeze(0)
        w = self.dist.log_prob(xi.cpu()).exp().to(xi.device)
        return w * (gp.predict(eval_x, return_std=True, grad=True).std**2)