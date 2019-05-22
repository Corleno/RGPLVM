#!/user/bin/env python3
'''
Create, 05/15/2019

@author: Rui Meng
'''
import matplotlib.pyplot as plt
import numpy as np

# Define a function for the line plot with intervals
def lineplotCI(x_data, y_data, sorted_x, low_CI, mid_CI, upper_CI, x_label, y_label, title=None, save_dir=None, save_name=None, Z = None):
    # Create the plot object
    fig, ax = plt.subplots()

    # Plot the data, set the linewidth, color and transparency of the
    # line, provide a label for the legend
    ax.scatter(x_data, y_data, lw = 1, color = '#539caf', alpha = 1, label = 'Original')
    # Plot the prediction
    ax.plot(sorted_x, mid_CI, lw = 1, color = 'b', alpha = 1, label = 'Fit')
    # Shade the confidence interval
    ax.fill_between(sorted_x, low_CI, upper_CI, color = '#539caf', alpha = 0.4, label = '95% CI')
    if Z is not None:
        ax.scatter(Z, np.zeros_like(Z), lw = 1, color = 'b', alpha = 1, marker = 'x', label = 'IP')
    # Label the axes and provide a title
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(sorted_x[0], sorted_x[-1])

    # Display legend
    ax.legend(loc = 'best')
    if save_name is not None:
        print (save_dir + save_name)
        plt.savefig(save_dir + save_name)
        plt.close(fig)

if __name__ == "__main__":
    pass