import torch
import nevergrad as ng
import numpy as np
import threading

import warnings
warnings.filterwarnings("ignore", message=r".*Initial solution argument*")  
warnings.filterwarnings("ignore", message=r".*sigma change np.exp*") 
warnings.filterwarnings("ignore", message=r".*orphanated injected solution*") 
warnings.filterwarnings("ignore", message=r".*Bounds are 1.0 sigma away from each other*") 

# Author: James Whiteley (github.com/jamesalexwhiteley)

# ============================================= 
# pytorch 
# =============================================  
def LBFGS(f, x, iters, bounds, lr=0.1):     
    """
    Unconstrained optimisation using torch.optim.LBFGS with bounds 

    """
    return torch_optim(f, x, iters, bounds, optimiser='LBFGS', lr=lr)

def ADAM(f, x, iters, bounds, lr=0.1):
    """
    Unconstrained optimisation using torch.optim.ADAM with bounds 

    """
    return torch_optim(f, x, iters, bounds, optimiser='ADAM', lr=lr)

def torch_optim(f, x, iters, bounds, optimiser='ADAM', lr=0.1):

    # adjust bounds for batch mode 
    with torch.no_grad():
        b0, b1 = bounds[:, 0], bounds[:, 1]
        b0 = b0.unsqueeze(0).expand_as(x)
        b1 = b1.unsqueeze(0).expand_as(x)

    x.requires_grad_(True) 
    
    if optimiser == 'LBFGS':
        optimizer = torch.optim.LBFGS([x], lr=lr) 
    elif optimiser == 'ADAM':
        optimizer = torch.optim.Adam([x], lr=lr)

    def closure():
        optimizer.zero_grad()  
        loss = f(x)            
        loss = -loss.sum()     
        loss.backward()   
        # loss.backward(retain_graph=True) 
        return loss

    for _ in range(int(iters)):
        optimizer.step(closure)

    with torch.no_grad():
        x = torch.min(torch.max(x, b0), b1)

    torch.cuda.empty_cache()
    return x, f(x)

# ============================================= 
# nevergrad  
# ============================================= 
def CMA_ES(f, g, x, iters, bounds, timeout_seconds=300):
    """
    Simple timeout wrapper - kills hanging optimizer after timeout_seconds
    """
    result = [None]  # Use list so thread can modify it
    
    def run_optimizer():
        try:
            device = x.device
            b0, b1 = bounds[:, 0].cpu().flatten().numpy(), bounds[:, 1].cpu().flatten().numpy()
            x_numpy = x.cpu().flatten().numpy()
            x_numpy = np.clip(x_numpy, b0, b1)
            
            def f2(x):
                x_tensor = torch.tensor(x, device=device).unsqueeze(0)
                return -f(x_tensor).detach().flatten().item()
            
            def g2(x):
                x_tensor = torch.tensor(x, device=device).unsqueeze(0)
                return (g(x_tensor).detach() >= 0).all().item()
            
            parametrization = ng.p.Array(init=x_numpy).set_bounds(lower=b0, upper=b1)
            parametrization.register_cheap_constraint(g2)
            optimizer = ng.optimizers.CMA(parametrization, budget=iters)
            res = optimizer.minimize(f2)
            
            x_result = torch.tensor(res.value).unsqueeze(0).to(device)
            result[0] = (x_result, f(x_result))
        except Exception as e:
            print(f"Optimizer error: {e}")
            result[0] = (x, f(x))  # Return original point if error
    
    # Start optimizer in thread
    thread = threading.Thread(target=run_optimizer)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    
    # Wait for timeout
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        print(f"Optimizer timed out after {timeout_seconds}s")
        return x, f(x)  # Return original point
    
    return result[0] if result[0] else (x, f(x))
