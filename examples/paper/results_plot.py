import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# Set up plotting style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams.update({'font.size': 12})

def plotdata(d, x, data, names, name, y_optimal):
    output_dir = 'figures'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f'{name}_bayesopt_{names[d]}.png')
    
    colours = ['#0072B2', '#009E73', '#D55E00']
    fontsize = 16 

    # Data for plotting 
    error = torch.abs(data - y_optimal)
    valid_mask = ~torch.any(torch.isnan(error), dim=1)
    error = error[valid_mask]

    # fig definition
    plt.figure(figsize=(6.25, 6))
    label = fr'$\alpha_{{\text{{{names[d]}}}}}$' if d != 1 else fr'$\tilde\alpha_{{\text{{{names[d]}}}}}$'

    # Individual runs 
    n_show = min(15, data.shape[0])
    show_alpha = 0.15 if data.shape[0] > 10 else 0.3
    for i in range(n_show):
        individual_error = error[i].numpy()
        label_run = 'Sample runs' if i == 0 else None
        plt.plot(x, individual_error, '-', color=colours[d], 
                alpha=show_alpha, linewidth=1, label=label_run)

    # Percentile bands
    p25 = torch.quantile(error, 0.25, dim=0).numpy()
    p50 = torch.quantile(error, 0.50, dim=0).numpy()
    p75 = torch.quantile(error, 0.75, dim=0).numpy()
    plt.fill_between(x, p25, p75, color=colours[d], alpha=0.2, label=f'{label} (IQR)')
    plt.plot(x, p50, color=colours[d], linewidth=2.5, alpha=1.0, label=f'{label} (median)')
    
    # Markers
    plt.scatter(x, p50, color=colours[d], s=30, edgecolor=colours[d])

    # Background lines 
    plt.axhline(y=0.0, color='#666666', linestyle='-', linewidth=1, alpha=0.7)
    plt.grid(True, linestyle='-', alpha=0.2, color='gray')

    # Labels 
    plt.xlabel("Iterations", fontsize=fontsize)
    plt.ylabel(r"Absolute Error $\quad$ $|y^* - \mathbb{E}[f(x^*, \xi)]|$", fontsize=fontsize)
    plt.legend(fontsize=fontsize-2, loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.show()

def main(): 
    # Configuration 
    names = ['RS', 'EIxPF-WSE', 'EIxPF'] 
    name = 'branin' 
    y_optimal = 8336.85
    N = 2 
    
    # Load the data 
    file_path = 'data/branin.pt'
    if not os.path.exists(file_path):
        print(f"Error: No data file found at {file_path}")
        exit(1)
    
    try:
        DATA = torch.load(file_path, weights_only=False)
    except TypeError:
        DATA = torch.load(file_path)
    
    # print(f"Loaded data shape: {DATA.shape}")
    ITERS = DATA.shape[2]
    
    # Extract only the computed points (every N-th iteration starting from 0)
    indices = list(range(0, ITERS, N))
    DATA_computed = DATA[:, :, indices]
    
    # print(f"Extracted computed points at iterations: {indices}")
    # print(f"New data shape: {DATA_computed.shape}")
    
    # Generate x-axis values for the computed points
    x = torch.tensor(indices)
    
    # Plot each method
    for d in range(DATA_computed.shape[0]):
        data = DATA_computed[d]  # Shape: (runs, computed_points)
        plotdata(d, x, data, names, name, y_optimal)
    
if __name__ == "__main__":
    main()