import torch 
import time

# Author: James Whiteley (github.com/jamesalexwhiteley)

class GaussianScaler():
    def __init__(self, data, dim=1, keepdim=True):
        """
        Gaussian scaler standardizes data to zero mean and unit standard deviation  

        """
        self.mean = data.mean(dim=dim, keepdim=keepdim)
        self.std = data.std(dim=dim, keepdim=keepdim)

    def to(self, device):
        for attr, value in self.__dict__.items():
            setattr(self, attr, to_device(value, device))
        return self
    
    def standardize(self, data):
        return (data - self.mean) / self.std

    def unstandardize(self, data):
        return data * self.std + self.mean
    
    def unscale(self, data):
        """ data * self.std """
        return data * self.std
    
    def unscale_var(self, data):
        """ data * self.std^2 """
        return data * self.std**2 
    
    def unscale_cov_matrix(self, data):
        """ data_{ij} * self.std_{i} * self.std_{j} """
        if self.std.ndim == 0:
            return data * self.std**2 
        else:
            outer_product_std = self.std.unsqueeze(0) * self.std.unsqueeze(1)
            return data * outer_product_std
    
class NormalScaler():
    def __init__(self, data, dim=1, keepdim=True):
        """
        Normal scaler normalises data to [0,1]

        Parameters 
        ---------- 
        dim : int 
            dim=None for array, dim=0 for column-wise, dim=1 for row-wise
        """
        self.min, _ = torch.min(data, dim=dim, keepdim=keepdim)
        self.max, _ = torch.max(data, dim=dim, keepdim=keepdim)

    def to(self, device):
        for attr, value in self.__dict__.items():
            setattr(self, attr, to_device(value, device))
        return self
    
    def normalize(self, data):
        return (data - self.min) / (self.max - self.min)

    def unnormalize(self, data):
        return data * (self.max - self.min) + self.min

def to_device(obj, device):
    """
    Automating object.to(device) for tensor, list and dict  
    
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [to_device(o, device) for o in obj]
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    else:
        return obj

_tic_start_time = None 

def tic():
    """Start the timer."""
    global _tic_start_time
    _tic_start_time = time.time()

def toc():
    """Stop the timer and print the elapsed time."""
    if _tic_start_time is None:
        print("Timer has not been started. Please call tic() first.")
        return
    elapsed_time = time.time() - _tic_start_time
    print(f"Elapsed time: {elapsed_time / 60:.4f} minutes")

def extract_blocks(matrix, h, w):
    """
    Extracts blocks from a matrix.
    
    Parameters:
    -----------
    matrix : torch.Tensor
        input matrix 
    h : int
        block height
    w : int
        block width 
        
    Returns:
    --------
    torch.Tensor
        new matrix (num_blocks, block_size)
    """
    num_rows, num_cols = matrix.shape
    
    assert num_rows % h == 0, f"num_rows {num_rows} must be divisible by block height h={h}"
    assert num_cols % w == 0, f"num_cols {num_cols} must be divisible by block width w={w}"

    num_blocks = num_rows // h
    
    blocks = []
    for i in range(num_blocks):
        row_block = matrix[i*h:(i+1)*h, :w] 
        blocks.append(row_block.reshape(-1))

    return torch.stack(blocks)

def gen_batch_data(x, xi): 
    """
    Associate each row vector in x with the set of xi points

    Parameters:
    -----------
    x : torch.Tensor, shape=(n, m)
        input matrix
    xi : torch.Tensor, shape=(p, q)
        input matrix

    Returns:
    --------
    torch.Tensor, shape=(n, p, m+q)

    """
    X = x.unsqueeze(1).repeat(1, xi.size(0), 1)  
    XI = xi.unsqueeze(0).repeat(x.size(0), 1, 1) 
    return torch.cat((X, XI), dim=-1)
        