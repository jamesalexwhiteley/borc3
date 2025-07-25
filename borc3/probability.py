import torch

# Author: James Whiteley (github.com/jamesalexwhiteley)

class DiscreteJoint:
    def __init__(self, joint_p, *value_grids):
        """
        Initialize an n-dimensional joint probability distribution
        
        Parameters
        ----------
        joint_p : torch.Tensor, shape=(d1, d2, ..., dn)
            n-dimensional tensor representing the joint probabilities of x1, x2, ..., xn.
        *value_grids : torch.Tensor
            A sequence of 1D tensors, each representing the possible values of one variable x1, x2, ..., xn.
            There should be one tensor per dimension, and each tensor should have length matching the corresponding 
            dimension size in joint_p.

        """
        self.joint_p = joint_p
        self.value_grids = value_grids  # list of value tensors for each dimension
        
        self.ndim = len(value_grids)
        self.joint_p_flat = joint_p.flatten()
        self.joint_distribution = torch.distributions.Categorical(probs=self.joint_p_flat)
        self.cumulative_p = torch.cumsum(self.joint_p_flat, dim=0)

    def sample(self, nsamples=(1,)):
        """
        Sample from the n-dimensional joint probability distribution
        
        Parameters
        ----------
        nsamples : int
            The number of samples to draw.
        
        Returns
        -------
        torch.Tensor
            Tensor of sampled values, shape (nsamples, ndim).

        """      
        sampled_indices = self.joint_distribution.sample(nsamples)
        sample_shape = self.joint_p.size()
        multi_indices = torch.unravel_index(sampled_indices, sample_shape)
        sampled_values = [self.value_grids[i][multi_indices[i]] for i in range(self.ndim)]
        
        return torch.stack(sampled_values, dim=-1)

    def log_prob(self, x):
        """
        Compute the log of the probability p(x) for a given set of values.
        
        Parameters
        ----------
        x : torch.Tensor, shape=(nsamples, ndim)
            Tensor representing the values for which the log probability needs to be computed.
        
        Returns
        -------
        torch.Tensor
            The log probabilities of the input values.

        """
        if x.shape[1] != self.ndim:
            raise ValueError(f"x should have {self.ndim} dimensions, but got {x.shape[1]}")
        
        # maintain cpu/gpu compatibility
        device = x.device
        
        # find value in grid closest [each element in] x
        indices = []
        for i in range(self.ndim):
            index = torch.argmin(torch.abs(self.value_grids[i].unsqueeze(0).to(device) - x[:, i].unsqueeze(1)), dim=1)
            indices.append(index)
        flattened_indices = torch.zeros_like(indices[0])
        
        multiplier = torch.tensor([1]).to(device)
        for i in reversed(range(self.ndim)):
            flattened_indices += indices[i] * multiplier
            multiplier *= self.joint_p.shape[i]

        log_probs = self.joint_distribution.log_prob(flattened_indices.cpu())
        return log_probs.to(device)
    
    def mean_vector(self):
        """
        Compute the mean vector for each dimension of the joint distribution.
        
        """
        means = []
        for axis in range(self.ndim):
            marginal = self.joint_p
            dims_to_sum = [i for i in range(self.ndim) if i != axis]
            marginal = marginal.sum(dim=dims_to_sum)
            
            mean = torch.dot(self.value_grids[axis], marginal)
            means.append(mean)
        
        return torch.tensor(means)

    def transform_uniform_samples(self, samples):
        """
        Map uniform samples to discrete joint distribution.
        
        Parameters
        ----------
        samples : torch.Tensor, shape=(nsamples, ndim)
            Uniform samples in the range [0, 1].
        
        Returns
        -------
        torch.Tensor, shape=(nsamples, ndim)
            Tensor of mapped values corresponding to the joint distribution.

        """
        num_samples = samples.size(0)
        indices = torch.searchsorted(self.cumulative_p, samples[:, 0].contiguous())

        sample_shape = self.joint_p.size()
        multi_indices = torch.unravel_index(indices, sample_shape)
        sampled_values = [self.value_grids[i][multi_indices[i]] for i in range(self.ndim)]

        return torch.stack(sampled_values, dim=-1).view(num_samples, self.ndim)
    
    def bounds(self):
        """
        Find the minimum and maximum x values for each dimension.
        
        Returns
        -------
        torch.Tensor, shape=(ndim, 2)
            A tensor containing the min and max value for each dimension.

        """
        min_values = []
        max_values = []
        
        for i in range(self.ndim):
            min_values.append(self.value_grids[i].min())  
            max_values.append(self.value_grids[i].max())  
        
        min_values = torch.tensor(min_values)
        max_values = torch.tensor(max_values)
        
        return torch.stack((min_values, max_values), dim=1)

class MultivariateNormal(torch.distributions.MultivariateNormal):
    def __init__(self, mu, cov):
        """
        Initialize a multivariate normal probability distribution
        
        Parameters
        ----------
        mu : torch.Tensor, shape=(ndim,)
            1d tensor representing the mean vector 
        cov : torch.Tensor, shape=(ndim, ndim)
            2d tensor representing the covariance matrix of the distribution

        """
        super().__init__(mu, cov)
        self.mu = mu 
        self.cov = cov 
        self.ndim = len(mu)

    def mean_vector(self):
        """
        Return the mean vector.
        
        """
        return self.mu

    def transform_uniform_samples(self, samples):
        """
        Map uniform samples to discrete joint distribution
        
        Parameters
        ----------
        samples : torch.Tensor, shape=(nsamples, ndim)
            uniform samples in the range [0, 1]
        
        Returns
        -------
        torch.Tensor, shape=(nsamples, ndim)
            tensor of mapped values 

        """
        # transform uniform to standard normal  
        standard_normal = torch.distributions.Normal(0.0, 1.0) 
        standard_normal_samples = standard_normal.icdf(samples) 

        # transform standard normal to multivariate normal 
        L = torch.linalg.cholesky(self.cov)
        standard_normal_samples = standard_normal_samples.to(L.dtype)
        xi = self.mu + standard_normal_samples @ L.T

        return xi 
    
    def bounds(self):
        """
        Find the minimum and maximum x values for each dimension.
        
        Returns
        -------
        torch.Tensor, shape=(ndim, 2)
            A tensor containing the min and max value for each dimension.
            
        """
        min_values = []
        max_values = []
        
        for i in range(self.ndim):
            min_values.append(self.mu[i] - 3 * self.cov[i][i])  
            max_values.append(self.mu[i] + 3 * self.cov[i][i])   
        
        min_values = torch.tensor(min_values)
        max_values = torch.tensor(max_values)
        
        return torch.stack((min_values, max_values), dim=1)


if __name__ == "__main__":

    # 3d joint probability distribution
    joint_p = torch.rand(3, 4, 5)  
    joint_p /= joint_p.sum()
    x1_values = torch.tensor([0, 1, 2])
    x2_values = torch.tensor([0, 1, 2, 3])
    x3_values = torch.tensor([0, 1, 2, 3, 4])

    # sample
    dist = DiscreteJoint(joint_p, x1_values, x2_values, x3_values)
    samples = dist.sample((10,))
    print(samples)

    # p(x)
    log_probs = dist.log_prob(samples)
    print(log_probs.exp())