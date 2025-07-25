import torch
import warnings
import os 
import pickle 
from gpytorch.utils.warnings import GPInputWarning
warnings.filterwarnings("ignore", category=GPInputWarning)
warnings.filterwarnings("ignore", message=r".*added jitter of*")

from borc3.gp import NoiselessGP, HomoscedasticGP, GPModelIO
from borc3.utilities import to_device 

# Author: James Whiteley (github.com/jamesalexwhiteley)

class SurrogateIO: 
    @staticmethod 
    def save(surrogate, output_folder):
        """
        Save the Surrogate model, including its objective and constraint GPs.

        """
        os.makedirs(output_folder, exist_ok=True)
        
        objective_paths = []
        for i, gp in enumerate(surrogate.objective_gps):
            path = os.path.join(output_folder, f"objective_gp_{i}.pth")
            GPModelIO.save(gp, path)
            objective_paths.append(path)
        
        constraint_paths = []
        for i, gp in enumerate(surrogate.constraint_gps):
            path = os.path.join(output_folder, f"constraint_gp_{i}.pth")
            GPModelIO.save(gp, path)
            constraint_paths.append(path)

        save_dict = {
            'surrogate_class': surrogate.__class__,
            'objective_gp_paths': objective_paths, 
            'constraint_gp_paths': constraint_paths,  
            'x': surrogate.x, 
            'm': surrogate.m,  
            'f': surrogate.f, 
            'g': surrogate.g, 
            'fbest': surrogate.fbest, 
            'xbest': surrogate.xbest, 
            'device': surrogate.device
        }

        torch.save(surrogate.problem, os.path.join(output_folder, "problem.pth"))
        torch.save(save_dict, os.path.join(output_folder, "surrogate_state.pth"))

    @staticmethod
    def load(output_folder):
        """
        Load the Surrogate model and its GPs.

        """
        surrogate_state = torch.load(os.path.join(output_folder, "surrogate_state.pth"), weights_only=False)
        
        objective_gps = []
        for path in surrogate_state['objective_gp_paths']:
            gp = GPModelIO.load(path)
            objective_gps.append(gp)
        
        constraint_gps = []
        for path in surrogate_state['constraint_gp_paths']:
            gp = GPModelIO.load(path)
            constraint_gps.append(gp)
        
        problem = torch.load(os.path.join(output_folder, "problem.pth"), weights_only=False)
        surrogate_class = surrogate_state.get('surrogate_class')                
        surrogate = surrogate_class(problem)

        surrogate.objective_gps = objective_gps
        surrogate.constraint_gps = constraint_gps
        surrogate.x = surrogate_state.get('x')
        surrogate.m = surrogate_state.get('m')
        surrogate.f = surrogate_state.get('f')
        surrogate.g = surrogate_state.get('g')
        surrogate.fbest = surrogate_state.get('fbest')
        surrogate.xbest = surrogate_state.get('xbest')
        surrogate.device = surrogate_state.get('device')

        return surrogate

class Surrogate():
    def __init__(self, 
                 problem,
                 gp=NoiselessGP,
                 gp_con=None, 
                 ntraining=100, 
                 nstarts=5,
                 sample_method='sobol',
                 dtype=torch.float, 
                 training_padding=0.1):
        """
        A GP surrogate model for problem class.  

        gp : class 
            the gp model to use for regression 
        ntraining : int 
            number of iterations used for optimizing GP hyperparameters 
        nstarts : int 
            number of random starting points to try when optimizing, 
        
        """
        self.gp_obj = gp 
        if gp_con != None:
            self.gp_con = gp_con
        else:
            self.gp_con = gp

        self.ntraining = ntraining 
        self.nstarts = nstarts 
        self.sample_method = sample_method
        self.batch_pred = None 
        self.name = "GP"
        self.device = "cpu"
        self.dtype = dtype 
        self.training_padding=training_padding
        self.x = None 
        self.m = None 
        self.f = None 
        self.g = None 

    # def add_problem(self, problem):
    #     """ 
    #     Gaussian Process surrogate for optimization problem

    #     (GP created for each function and each constraint) 

    #     """ 
        self.problem = problem 
        # self.batch_gp = [None] # batch gp
        self.objective_gps = [None for _ in range(len(self.problem.obj_fun))] # objective gps 
        self.constraint_gps = [None for _ in range(len(self.problem.con_fun))] # constraint gps 

    def cuda(self, device):
        self.device = device 
        if self.x != None:
            self.x = self.x.to(self.device)
            self.m = self.m.to(self.device)
            self.f = self.f.to(self.device)
            self.g = self.g.to(self.device)

    def build_obj_gp(self, data_x, data_y, normalize_x=True, standardize_y=True):
        """
        Build a objective GP  

        """ 
        gp = self.gp_obj(data_x, data_y, normalize_x=normalize_x, standardize_y=standardize_y, ntraining=self.ntraining, nstarts=self.nstarts) 
        gp.cuda(self.device)
        gp.fit() 
        return gp 
    
    def build_con_gp(self, data_x, data_y, normalize_x=True, standardize_y=True):
        """
        Build a constraint GP  

        """ 
        gp = self.gp_con(data_x, data_y, normalize_x=normalize_x, standardize_y=standardize_y, ntraining=self.ntraining, nstarts=self.nstarts) 
        gp.cuda(self.device)
        gp.fit() 
        return gp 

    def run_model_initial(self, nsamples, method):
        """
        Initialise data from model  

        """
        self.x = self.problem.sample(nsamples=nsamples, method=method, dtype=self.dtype)
        self.m = self.problem.model(self.x) 
        self.f = self.problem.objectives(self.m) 
        self.g = self.problem.constraints(self.m) 

        if self.device != 'cpu':
            self.x = to_device(self.x, self.device)
            self.m = to_device(self.m, self.device)
            self.f = to_device(self.f, self.device)
            self.g = to_device(self.g, self.device)
    
    def build(self, nsamples, normalize_x=True, standardize_y=True, sample_method="sobol"):
        """
        Build GPs of problem objectives and constraints by running the model 

        Parameters
        ----------
        nsamples : int
            number of input points used to model gps 
        normalise_x : bool
            whether to normalise input features
        standardise_y : bool  
            whether to standardise output targets
        sample_method : str
            sampling method for generating training points

        """
        self.fbest, self.xbest = None, None 
        
        # Temporarily use padded bounds for training data generation
        original_bounds = self.problem.param_bounds.copy()
        padded_bounds = self.problem._get_padded_bounds(padding=self.training_padding)
        self.problem.param_bounds = padded_bounds
        
        with torch.no_grad(): 
            # look for a feasible point to initialise archive 
            for i in range(10):
                self.run_model_initial(nsamples=nsamples, method=sample_method)
                self.get_best()
                if self.fbest != None:
                    break 
                elif i == 9: 
                    raise ValueError("After 10 attempts, could not find an initialization with at least one feasible point")
        
        # Restore original bounds for optimisation
        self.problem.param_bounds = original_bounds
                
        for i, _ in enumerate(self.problem.obj_fun): 
            y = self.f[:, i]
            if y.dim() != 1:
                raise ValueError("Objective train_y needs to be 1d to be able to fit gps")
            self.objective_gps[i] = self.build_obj_gp(self.x, y, normalize_x=normalize_x, standardize_y=standardize_y)
            
        for i, _ in enumerate(self.problem.con_fun):
            y = self.g[:, i]
            if y.dim() != 1:
                raise ValueError("Constraint train_y needs to be 1d to be able to fit gps")
            self.constraint_gps[i] = self.build_con_gp(self.x, y, normalize_x=normalize_x, standardize_y=standardize_y)

        return self.fbest, self.xbest 
    
    def get_best(self):
        """
        Get best x, f point  

        """
        _, max_ind = torch.max(self.f, dim=0)
        self.fbest = self.f[max_ind] 
        self.xbest = self.x[max_ind]

    def run_model_update(self, new_x):
        """
        Generate data from model

        """    
        m = self.problem.model(new_x)
        f = self.problem.objectives(m) 
        g = self.problem.constraints(m) 
        return f, g

    def update(self, new_x):
        """
        Add new training data to the objective and constraint GPs 

        Parameters
        ----------
        new_x : torch.Tensor, shape=(nsamples, nparam)  
            new input test data 
        new_y : torch.Tensor, shape=(nsamples, nparam)  
            new output test data 
        
        """         
        # device 
        new_x = new_x.to(self.device)
        self.new_x = new_x

        f, g = self.run_model_update(self.new_x)
        g = to_device(g, self.device) 
        f = f.to(self.device)

        for i, gp in enumerate(self.objective_gps): 
            gp.update(self.new_x, f[:, i]) 

        for i, gp in enumerate(self.constraint_gps): 
            gp.update(self.new_x, g[:, i]) 

        # update training data 
        self.x = torch.cat((self.x, self.new_x), dim=0)
        self.f = torch.cat((self.f, f), dim=0)
        if len(self.g) != 0:
            self.g = torch.cat((self.g, g), dim=0)

        with torch.no_grad(): 
            return self.fbest, self.xbest 
    
    def predict_objectives(self, x, return_std=True, return_var=False, return_cov=False, grad=False):
        """
        Predict objectives based on GPs 

        Parameters
        ----------
        x : torch.Tensor, shape=(nsamples, nparam)
            points to evaluate 

        """
        pred = [gp.predict(x, return_std=return_std, return_var=return_var, return_cov=return_cov, grad=grad) for gp in self.objective_gps]
        return pred

    def predict_constraints(self, x, return_std=True, return_var=False, return_cov=False, grad=False):
        """
        Predict constraints based on GPs 

        Parameters
        ----------
        x : torch.Tensor, shape=(nsamples, nparam)
            points to evaluate 

        """ 
        pred = [gp.predict(x, return_std=return_std, return_var=return_var, return_cov=return_cov, grad=grad) for gp in self.constraint_gps]
        return pred 
        
    def monte_carlo(self, 
                    params, 
                    nsamples=int(1e4), 
                    obj_type="det",
                    obj_ucb=[1],
                    con_type="prob", 
                    con_ucb=[1],
                    con_eps=0.1,
                    output=False):
        
        return self.problem._monte_carlo( 
                    obj_fun=lambda x : self.predict_objectives(x)[0].mu.unsqueeze(1),
                    con_fun=lambda x : [cons.mu.unsqueeze(1) for cons in self.predict_constraints(x)], 
                    surrogate=True,
                    params=params, 
                    nsamples=nsamples, 
                    obj_type=obj_type,
                    obj_ucb=obj_ucb,
                    con_type=con_type, 
                    con_ucb=con_ucb,
                    con_eps=con_eps,
                    output=output,
                    device=self.device)   