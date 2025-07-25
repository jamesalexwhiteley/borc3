import os 
import torch 
import gpytorch 
import warnings
warnings.filterwarnings("ignore", message=r".*Negative variance values detected.*") 
from gpytorch.utils.warnings import GPInputWarning
warnings.filterwarnings("ignore", category=GPInputWarning) 

from borc3.utilities import NormalScaler, GaussianScaler 
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy

# Author: James Whiteley (github.com/jamesalexwhiteley) 

class Posterior():
    def __init__(self):
        self.mu = None 
        self.std = None 
        self.std_epistemic = None 
        self.std_aleatoric = None 
        self.var = None 
        self.cov = None 

    def posterior(self, return_epistemic=False, return_aleatoric=False):
        if return_epistemic: 
            return self.mu, self.std_epistemic
        elif return_aleatoric:
            return self.mu, self.std - self.std_epistemic
        else: 
            return self.mu, self.std

    def cuda(self, device):
        for attr, value in self.__dict__.items():
            if value != None:
                setattr(self, attr, value.to(device))

class GPModelIO:
    @staticmethod
    def save(model, output_path, metadata=None):
        """
        Save GP model and optional metadata.
        Only saves attributes that exist on the model.
        """
        # Core required attributes
        save_dict = {
            'model_class': model.__class__,
            'model_state_dict': model.state_dict(),
            'likelihood_state_dict': model.likelihood.state_dict(),
            'train_x': model.train_x,
            'train_y': model.train_y,
            'device': model.device,
        }
        
        # Optional model initialization arguments
        model_init_args = {}
        init_attrs = ['normalize_x', 'standardize_y', 'ntraining', 'nstarts', 'num_inducing']
        for attr in init_attrs:
            if hasattr(model, attr):
                model_init_args[attr] = getattr(model, attr)
        
        if model_init_args:
            save_dict['model_init_args'] = model_init_args
        
        # Optional scalers
        for scaler_attr in ['scaler_x', 'scaler_y']:
            if hasattr(model, scaler_attr):
                save_dict[scaler_attr] = getattr(model, scaler_attr)

        if metadata:
            save_dict['metadata'] = metadata

        torch.save(save_dict, output_path)

    @staticmethod
    def load(output_path):
        """
        Load GP model.
        """
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"No file found at {output_path}")
        
        checkpoint = torch.load(output_path, weights_only=False)

        model_class = checkpoint['model_class']
        train_x = checkpoint['train_x']
        train_y = checkpoint['train_y']
        
        # Use model_init_args if available, otherwise create model with just train data
        model_init_args = checkpoint.get('model_init_args', {})
        model = model_class(train_x=train_x, train_y=train_y, **model_init_args)

        # Load state dictionaries
        model.load_state_dict(checkpoint['model_state_dict'])
        model.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        
        # Restore scalers if they were saved
        for scaler_attr in ['scaler_x', 'scaler_y']:
            if scaler_attr in checkpoint:
                setattr(model, scaler_attr, checkpoint[scaler_attr])
        
        # Set device and mode
        model.device = checkpoint['device']
        model.cuda(model.device)
        model.eval()

        metadata = checkpoint.get('metadata', {})
        if metadata:
            print(f"Loaded metadata: {metadata}")

        return model
    
class HomoscedasticGP(gpytorch.models.ExactGP):
    def __init__(self, 
                 train_x, 
                 train_y, 
                 normalize_x=True,
                 standardize_y=True, 
                 ntraining=100, 
                 nstarts=5):
        """
        gpytorch.models.ExactGP
        Homoscedastic noise y = f(x) + N(0, std^2)

        Parameters
        ----------
        train_x : torch.Tensor, shape=(nsamples, nparam)  
            input test data 
        train_y : torch.Tensor, shape=(nsamples, )  
            output test data    
    
        """
        self.name = 'HomoscedasticGP'
        self.jitter = 1e-4
        self.normalize_x = normalize_x   
        self.standardize_y = standardize_y 
        self.ntraining = ntraining 
        self.nstarts = nstarts  
        self.fbest = torch.max(train_y) 
        self.xbest = train_x[list(torch.where(train_y == self.fbest))] 
        self.device = 'cpu'

        if self.normalize_x:
            self.scaler_x = NormalScaler(train_x, dim=0) 
            train_x = self.scaler_x.normalize(train_x) 

        if self.standardize_y:
            self.scaler_y = GaussianScaler(train_y, dim=0) 
            train_y = self.scaler_y.standardize(train_y) 

        self.train_x = train_x 
        self.train_y = train_y        

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.GreaterThan(1e-4))
        # likelihood.raw_noise.data = torch.tensor(1e-3)  # Start with higher noise

        super().__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(1))) # use seperate hyperparameter for each dimension 
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.size(1))) 

    def forward(self, x): 
        """ 
        Compute the mean and covariance functions at the point x 

        Parameters
        ----------
        x : torch.Tensor 
            point to evaluate 

        """
        mu = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mu, cov)
    
    def cuda(self, device):
        self.to(device)
        self.likelihood.to(device)
        self.train_x = self.train_x.to(device)
        self.train_y = self.train_y.to(device)
        self.device = device 

    def train(self, bool=True): 
        """
        Switch the model to training mode (for learning hyperparameters)

        """
        super().train(bool)
        self.likelihood.train(bool)

    def eval(self):
        """
        Switch the model to evaluation mode (for making predictions)

        """
        super().eval()
        self.likelihood.eval()

    def optimize_hyp(self):
        """
        Optimize the hyperparameters by maximising the log marginal likelihood with a multi start approach 
        
        """
        best_state_dict, min_loss = None, torch.tensor(float('inf'))
        for _ in range(self.nstarts):

            self.initialize()
            optimizer = torch.optim.Adam(self.parameters(), lr=0.05)  
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self) 

            for _ in range(self.ntraining):
                optimizer.zero_grad()
                res = self(self.train_x)
                loss = -mll(res, self.train_y).sum() 
                loss.backward() 

                # clip gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step() 

            final_loss = float(loss.detach()) 
            if final_loss < min_loss: 
                min_loss = final_loss 
                best_state_dict = self.state_dict() 

        self.load_state_dict(best_state_dict) 

    def fit(self):
        """
        Fit the gp model to the data 

        """
        self.train() 
        self.optimize_hyp()
        self.eval()

    def posterior(self, pred):  
        """
        Get posterior distribution p(y|D) = likelihood(model(x))

        """            
        mu = pred.mean
        if self.standardize_y:
            mu = self.scaler_y.unstandardize(mu)
        self.pred.mu = mu   

        if self.return_std:
            std = pred.stddev
            if self.standardize_y:
                std = self.scaler_y.unscale(std)
            self.pred.std = std 
            self.pred.std_epistemic = std 
            self.pred.std_aleatoric = torch.tensor(0.0)

        if self.return_var:
            var = pred.variance
            if self.standardize_y:
                var = self.scaler_y.unscale_var(var)
            self.pred.var = var 
        
        if self.return_cov:
            cov = pred.covariance_matrix
            if self.standardize_y:
                cov = self.scaler_y.unscale_cov_matrix(cov)
            self.pred.cov = cov 

        return self.pred

    def predict(self, test_x, return_std=False, return_var=False, return_cov=False, grad=False):
        """
        Make predictions at point(s) x 

        """
        x = test_x.float()
        device = next(self.parameters()).device # run on same device as self.model by default
        self.to(device)

        self.pred = Posterior()
        self.return_std, self.return_var, self.return_cov = return_std, return_var, return_cov
        
        if self.normalize_x:
            x = self.scaler_x.normalize(x)

        with gpytorch.settings.max_preconditioner_size(10): 
            with gpytorch.settings.cholesky_jitter(self.jitter):
                with gpytorch.settings.max_cholesky_size(512): 
                    if grad:
                        with gpytorch.settings.fast_pred_var():
                            self.posterior(self.likelihood(self(x))) 
                            return self.pred    

                    else:
                        with torch.no_grad(), gpytorch.settings.fast_pred_var():
                            self.posterior(self.likelihood(self(x))) 
                            return self.pred 
                    
    def get_training_data(self, device=None):
        """
        Return the internal training data in the input scale 
        
        """
        x = self.train_x 
        y = self.train_y 

        if self.normalize_x:
            x = self.scaler_x.unnormalize(x)

        if self.standardize_y:
            y = self.scaler_y.unstandardize(y)

        if device != None:
            x, y = x.to(device), y.to(device)

        return x, y
    
    def update(self, new_x, new_y):
        """
        Add new training points to the gp 

        Parameters
        ----------
        new_x : torch.Tensor, shape=(nsamples, nparam)  
            new input test data 
        new_y : torch.Tensor, shape=(nsamples, nparam)  
            new output test data 

        """
        x, y = self.get_training_data()
        device = x.device  
        new_x = new_x.to(device)
        new_y = new_y.to(device)

        with torch.no_grad():

            train_x = torch.cat((x, new_x), dim=-2)
            train_y = torch.cat((y, new_y), dim=-1)

            assert train_x.shape[0] == train_y.shape[0]

            self.fbest = torch.max(train_y) 
            self.xbest = train_x[list(torch.where(train_y == self.fbest))] 

            if self.normalize_x:
                self.scaler_x = NormalScaler(train_x, dim=0) 
                train_x = self.scaler_x.normalize(train_x) 

            if self.standardize_y:
                self.scaler_y = GaussianScaler(train_y, dim=0) 
                train_y = self.scaler_y.standardize(train_y) 

            super().set_train_data(inputs=train_x, targets=train_y, strict=False)

        self.train_x = train_x 
        self.train_y = train_y 

        # check for duplicate points
        distances = torch.cdist(train_x, train_x)
        min_distance = distances[distances > 0].min()
        if min_distance < 1e-6:
            print("Warning: Near-duplicate training points detected")
            # Remove duplicates or add small noise
            train_x = train_x + torch.randn_like(train_x) * 1e-2
        
        # check condition number
        if self.normalize_x:
            # Check if data spans reasonable range
            data_range = train_x.max(dim=0).values - train_x.min(dim=0).values
            if (data_range < 1e-6).any():
                print("Warning: Some dimensions have very small range")

        # try to refit the model
        self.fit() 
        self.eval()
        # try:
        #     self.fit()  
        # except Exception as e:
        #     print(f"Warning: GP retraining failed. Using existing hyperparameters. Error: {e}")
        #     with torch.no_grad():
        #         super().set_train_data(inputs=self.train_x, targets=self.train_y, strict=False)           
        #         self._clear_cache()
        #         self.eval()

class VariationalHomoscedasticGP(ApproximateGP):
    def __init__(self, 
                 train_x, 
                 train_y, 
                 normalize_x=True,
                 standardize_y=True, 
                 ntraining=200, 
                 nstarts=5,
                 num_inducing=100):
        """
        Variational GP that exactly matches HomoscedasticGP API
        """
        self.name = 'VariationalHomoscedasticGP'
        self.jitter = 1e-4
        self.normalize_x = normalize_x   
        self.standardize_y = standardize_y 
        self.ntraining = ntraining 
        self.nstarts = nstarts
        self.num_inducing = min(num_inducing, len(train_x))
        self.device = 'cpu'
        
        # Store original data
        self.original_train_x = train_x.clone()
        self.original_train_y = train_y.clone()
        self.fbest = torch.max(train_y) 
        self.xbest = train_x[list(torch.where(train_y == self.fbest))] 

        # Apply scaling
        if self.normalize_x:
            self.scaler_x = NormalScaler(train_x, dim=0) 
            train_x = self.scaler_x.normalize(train_x) 

        if self.standardize_y:
            self.scaler_y = GaussianScaler(train_y, dim=0) 
            train_y = self.scaler_y.standardize(train_y) 

        self.train_x = train_x 
        self.train_y = train_y

        # Inducing points 
        inducing_points = train_x[:self.num_inducing, :]
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VariationalHomoscedasticGP, self).__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def cuda(self, device):
        self.to(device)
        self.likelihood.to(device)
        self.train_x = self.train_x.to(device)
        self.train_y = self.train_y.to(device)
        self.original_train_x = self.original_train_x.to(device)
        self.original_train_y = self.original_train_y.to(device)
        self.device = device 

    def train(self, bool=True): 
        super().train(bool)
        self.likelihood.train(bool)

    def eval(self):
        super().eval()
        self.likelihood.eval()

    def fit(self):
        """
        See https://docs.gpytorch.ai/en/v1.13/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
        """
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=0.02)

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=self.train_y.size(0))

        for i in range(self.ntraining):
            optimizer.zero_grad()
            output = self(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()

        self.eval()
        self.likelihood.eval()

    def posterior(self, pred):  
        mu = pred.mean
        if self.standardize_y:
            mu = self.scaler_y.unstandardize(mu)
        self.pred.mu = mu   

        if self.return_std:
            std = pred.stddev
            if self.standardize_y:
                std = self.scaler_y.unscale(std)
            self.pred.std = std 
            self.pred.std_epistemic = std 
            self.pred.std_aleatoric = torch.tensor(0.0)

        if self.return_var:
            var = pred.variance
            if self.standardize_y:
                var = self.scaler_y.unscale_var(var)
            self.pred.var = var 
        
        if self.return_cov:
            cov = pred.covariance_matrix
            if self.standardize_y:
                cov = self.scaler_y.unscale_cov_matrix(cov)
            self.pred.cov = cov 

        return self.pred

    def predict(self, test_x, return_std=False, return_var=False, return_cov=False, grad=False):
        x = test_x.float()
        device = next(self.parameters()).device
        self.to(device)

        self.pred = Posterior()
        self.return_std, self.return_var, self.return_cov = return_std, return_var, return_cov
        
        if self.normalize_x:
            x = self.scaler_x.normalize(x)

        with gpytorch.settings.max_preconditioner_size(100): 
            with gpytorch.settings.cholesky_jitter(self.jitter):
                if grad:
                    with gpytorch.settings.fast_pred_var():
                        self.posterior(self.likelihood(self(x))) 
                        return self.pred    
                else:
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        self.posterior(self.likelihood(self(x))) 
                        return self.pred 
                    
    def get_training_data(self, device=None):
        x = self.train_x 
        y = self.train_y 

        if self.normalize_x:
            x = self.scaler_x.unnormalize(x)

        if self.standardize_y:
            y = self.scaler_y.unstandardize(y)

        if device != None:
            x, y = x.to(device), y.to(device)

        return x, y
    
    def update(self, new_x, new_y):
        """Simple update"""
        x, y = self.get_training_data()
        device = x.device  
        new_x = new_x.to(device)
        new_y = new_y.to(device)

        # Combine data
        train_x = torch.cat((x, new_x), dim=-2)
        train_y = torch.cat((y, new_y), dim=-1)
        
        self.fbest = torch.max(train_y) 
        self.xbest = train_x[list(torch.where(train_y == self.fbest))] 

        # Create new model with updated data
        new_model = VariationalHomoscedasticGP(
            train_x, train_y,
            normalize_x=self.normalize_x,
            standardize_y=self.standardize_y,
            ntraining=self.ntraining,
            nstarts=self.nstarts,
            num_inducing=self.num_inducing
        )
        
        # Copy the new model's attributes to self
        self.__dict__.update(new_model.__dict__)
        
        try:
            self.fit()
        except Exception as e:
            print(f"Warning: Retraining failed: {e}")
            self.eval()

class NoiselessGP(HomoscedasticGP): 
    def __init__(self, 
                 train_x, 
                 train_y, 
                 normalize_x=True,
                 standardize_y=True, 
                 ntraining=10, 
                 nstarts=5):
        """
        gpytorch.models.ExactGP
        Noiesless observations y = f(x)

        """  
        super().__init__(train_x=train_x, train_y=train_y, normalize_x=normalize_x, standardize_y=standardize_y, ntraining=ntraining, nstarts=nstarts) 
        self.name = 'NoiselessGP'
        noise = 1e-4 * torch.ones_like(train_y)
        noise.requires_grad_(False) # ensure noise is not altered when optimizing hyperparameters
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=noise, learn_additional_noise=False)

class HeteroscedasticGP(HomoscedasticGP):
    def __init__(self, 
                train_x, 
                train_y, 
                noise, # noise variance 
                normalize_x=True,
                standardize_y=True, 
                ntraining=20, 
                nstarts=5):
        """
        gpytorch.models.ExactGP
        Heteroscedastic noise y = f(x) + N(0,std(x)^2) 

        """
        super().__init__(train_x=train_x, train_y=train_y, normalize_x=normalize_x, standardize_y=standardize_y, ntraining=ntraining, nstarts=nstarts)
        self.noise = noise # overwrite the noise hyperparameter 
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=self.noise, learn_additional_noise=False) 
