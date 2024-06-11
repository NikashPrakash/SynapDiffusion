import os
import numpy as np
import matplotlib.pyplot as plt


def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, "config"):
        with open("/scratch/eecs448w24_class_root/eecs448w24_class/shared_data/brainWiz/config.json") as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split("."):
        node = node[part]
    return node


def make_training_plot(stats, name="Fine-Tuning"): #TODO ADAPT TO CURRENT DATA
    """Set up an interactive matplotlib graph to log metrics during training."""
    # plt.ion()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    plt.suptitle(name)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Recall")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("Precision")

    epoch_set = np.arange(len(stats))
    for i in range(4):
        axes[i].plot(epoch_set,stats[:,0+i], 'r--', marker="o", label="Validation")
        axes[i].plot(epoch_set,stats[:,4+i], 'b--', marker="o", label="Training")
        axes[i].legend()
    save_training_plot()
    
def make_minibatch_training_plot(stats, name="Training Loss"):
    """Set up an interactive matplotlib graph to log metrics during training."""
    # plt.ion()
    fig, axes = plt.subplots()
    plt.title(name)
    axes.set_xlabel("Minibatch")
    axes.set_ylabel("Loss")
    print(stats)
    minibatchSet = np.arange(len(stats))
    axes.plot(minibatchSet,stats, 'b--', marker="o", label="Training")
    axes.legend()
    save_training_plot(name)

def save_training_plot(name):
    """Save the training plot to a file."""
    plt.savefig(f"Trained_{name}.png", dpi=200)
