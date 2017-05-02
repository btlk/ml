import os, sys, random
import numpy as np
import csv

from argparse import ArgumentParser
from random import shuffle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
# matplotlib.rcParams['backend'] = "Qt5Agg"
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def get_rand_color():
  return '#%06X' % (random.randint(0, 256**3 - 1))


def draw_tsne(original_data, results_data):
  tsne = TSNE(n_components = 2)

  plt.subplot(121)

  orig_tsne = tsne.fit_transform(original_data[:, 1:])
  orig_classes = int(original_data[:, 0].max()) + 1
  for i in range(0, orig_classes):
    pts = orig_tsne[original_data[:, 0] == i]
    plt.plot(pts, color = get_rand_color(), 
      marker = 'o', linestyle = '', alpha = 0.7)

  plt.subplot(122)

  res_tsne = tsne.fit_transform(results_data[:, 1:])
  res_classes = int(results_data[:, 0].max()) + 1
  for i in range(1, res_classes):
    pts = res_tsne[results_data[:, 0] == i]
    plt.plot(pts, color = get_rand_color(), 
      marker = 'o', linestyle = '', alpha = 0.7)

  plt.show()


def draw_2d(original_data, results_data):
  plt.subplot(121)

  orig_classes = int(original_data[:, 0].max()) + 1
  for i in range(0, orig_classes):
    pts = original_data[original_data[:, 0] == i][:, 1:]
    plt.plot(pts, color = get_rand_color(), 
      marker = 'o', linestyle = '', alpha = 0.7)

  plt.subplot(122)

  res_classes = int(results_data[:, 0].max()) + 1
  for i in range(1, res_classes):
    pts = results_data[results_data[:, 0] == i][:, 1:]
    plt.plot(pts, color = get_rand_color(), 
      marker = 'o', linestyle = '', alpha = 0.7)

  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser(description = 'Dataset generator')
  parser.add_argument('--original', '-o',
                      type = str,
                      help = 'Path to original data csv',
                      required = True)
  parser.add_argument('--result', '-r',
                      type = str,
                      help = 'Path to reuslts csv',
                      required = True)

  args = parser.parse_args()

  original_path = args.original
  results_path = args.result

  original_data = np.loadtxt(original_path, delimiter = ',')
  results_data = np.loadtxt(results_path, delimiter = ',')

  data_dim = original_data.shape[-1] - 1

  if data_dim > 2:
    draw_tsne(original_data, results_data)
  else:
    draw_2d(original_data, results_data)

