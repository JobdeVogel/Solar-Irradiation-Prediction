import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import sys
import os

def expected_cm(dataloader):
    pass

def evaluate_sample_rmses(sample_rmses, bin_interval=10, image_dir='', show=False, save=True):
    """ 
    Loop over all samples, plot the RMSE for each individual sample
    """
    if show:
        matplotlib.use('TkAgg')
    
    bins = (torch.ceil(torch.max(sample_rmses) / bin_interval)).int().item()
    bin_edges = torch.linspace(0, torch.ceil(torch.max(sample_rmses) / bin_interval) * bin_interval, steps=bins + 1)

    avg_rmse = torch.mean(sample_rmses)
    for i, edge in enumerate(bin_edges):
        if avg_rmse < edge:
            avg_bin = i - 1
            break

    err_names = [str(int(bin_edges[i].item())) + '-' + str(int(bin_edges[i+1].item())) for i in range(len(bin_edges[:-1]))]
    hist = torch.histc(sample_rmses, bins=bins, min=0, max=torch.ceil(torch.max(sample_rmses) / bin_interval) * bin_interval).long().cpu()

    # Plotting the histogram
    colors = ['blue' for orange in range(bins)]
    colors[avg_bin] = 'orange'
    
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(err_names, hist, color=colors)
    ax.set_xlabel('RMSE [kWh/m2]')
    ax.set_ylabel('Sample Frequency')

    plt.title('RMSE Error Distribution over Test Samples')
    
    if save:
        path = os.path.join(image_dir, 'sample_rmses')
    
        plt.savefig(path, dpi=300)
        print(f"Saved sample rmses figure in {path}")
    
    if show:
        plt.show()
    
    plt.close()
    return

def evaluate_point_rmses(all_logits, all_targets, bin_interval=50, image_dir='', show=False, save=True):
    """
    Loop overal all points, plot the RMSE for each individual point
    """   
    if show:
        matplotlib.use('TkAgg')
    
    abs_errors = torch.abs(all_targets - all_logits).cpu()
    
    bins = (torch.ceil(torch.max(abs_errors) / bin_interval)).int().item()
    bin_edges = torch.linspace(0, torch.ceil(torch.max(abs_errors) / bin_interval) * bin_interval, steps=bins + 1)

    avg_error = torch.mean(abs_errors)
    for i, edge in enumerate(bin_edges):
        if avg_error < edge:
            avg_bin = i - 1
            break

    err_names = [str(int(bin_edges[i].item())) + '-' + str(int(bin_edges[i+1].item())) for i in range(len(bin_edges[:-1]))]
    hist = torch.histc(abs_errors, bins=bins, min=0, max=torch.ceil(torch.max(abs_errors) / bin_interval) * bin_interval).long().cpu()

    # Plotting the histogram
    colors = ['blue' for _ in range(bins)]
    colors[avg_bin] = 'orange'
    
    fig, ax = plt.subplots(figsize=(18, 5))
    
    ax.bar(err_names, hist, color=colors)
    ax.set_xlabel('Absolute Error [kWh/m2]')
    ax.set_ylabel('Point Frequency')
    ax.set_yscale("log")
    
    max_height = torch.max(hist)
    for xpos, ypos, yval in zip(err_names, hist, hist):
        plt.text(xpos, ypos, "N=%d"%yval, ha="center", va="bottom")
    
    plt.title('Absolute Error Distribution over Test Points')
    
    if save:
        path = os.path.join(image_dir, 'point_rmses')
    
        plt.savefig(path, dpi=300)
        print(f"Saved point accuracy figure in {path}")
    
    if show:
        plt.show()
    
    return plt

def evaluate_bin_accuracy(confusion_matrix, gt_confusion_matrix, irr_names,  normalize_bars=True, image_dir='', show=False, save=True):
    """
    Loop over confusion matrix, plot the accuracy of each individual irradiance bin 
    """
    if show:
        matplotlib.use('TkAgg')
    
    true_bars = np.array([])
    pred_bars = np.array([])
    one_off_bars = np.array([])
    
    for i in range(confusion_matrix.shape[0]):
        pred_num_class = confusion_matrix[i, i]
        true_num_class = gt_confusion_matrix[i, i]
        
        # Check if not the last bin
        if i == 0:
            one_off_num_class = confusion_matrix[i, i + 1]
        elif i != confusion_matrix.shape[0] - 1:
            one_off_num_class = confusion_matrix[i, i - 1] + confusion_matrix[i, i + 1]
        else:
            one_off_num_class = confusion_matrix[i, i - 1]
    
        true_bars = np.append(true_bars, true_num_class)
        pred_bars = np.append(pred_bars, pred_num_class)
        one_off_bars = np.append(one_off_bars, one_off_num_class)
    
    wrong_bars = true_bars - pred_bars - one_off_bars
    fig, ax = plt.subplots(figsize=(13, 5))
    
    n = np.copy(true_bars)
    
    if normalize_bars:        
        pred_bars = (pred_bars/true_bars) * 100
        one_off_bars = (one_off_bars/true_bars) * 100
        wrong_bars = (wrong_bars/true_bars) * 100
        true_bars = (true_bars/true_bars) * 100
    
    wrong = ax.bar(irr_names, wrong_bars, bottom=one_off_bars+pred_bars, color='red', label='Wrong')
    off = ax.bar(irr_names, one_off_bars, bottom=pred_bars, color='orange', label='One off')
    pred = ax.bar(irr_names, pred_bars, color='green', label='Correct')
    
    ax.bar_label(wrong, labels=np.round(wrong_bars/true_bars, 2), label_type='center')
    ax.bar_label(off, labels=np.round(one_off_bars/true_bars, 2), label_type='center')
    ax.bar_label(pred, labels=np.round(pred_bars/true_bars, 2), label_type='center')
    
    max_height = np.max(true_bars)
    for xpos, ypos, yval in zip(irr_names, true_bars, n):
        plt.text(xpos, ypos + max_height * 0.05, "N=%d"%yval, ha="center", va="bottom")
    
    plt.legend(bbox_to_anchor=(1.01,0.5), loc='center left')
    
    if save:
        path = os.path.join(image_dir, 'bin_accuracy')
    
        fig.savefig(path, dpi=300, width=2000)
        print(f"Saved bin accuracy figure in {path}")
        
    if show:
        plt.show() 
    
    return plt, ax