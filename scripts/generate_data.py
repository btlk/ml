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


class ClassSpace:
  num_classes = 0

  def __init__(self, format_str = None):
    decoded = list(map(float, format_str[1:-1].split(',')))
    self.center = decoded[:-1]
    self.radius = decoded[-1]
    self.num = ClassSpace.num_classes
    self.dim = len(self.center)
    self.points = []

    ClassSpace.num_classes += 1

  def __str__(self):
    return str(self.center + [self.radius])

  def __repr__(self):
    return str(self.center + [self.radius])

  def populate(self, count):
    self.points = []
    normal_deviates = np.random.normal(size = (count, self.dim))
    radius = np.sqrt((normal_deviates**2).sum(axis = 0))
    self.points = normal_deviates / radius * 10.0

    self.points *= self.radius
    self.points += self.center


def norm_classes(cl_list):
  pts = np.concatenate([x.points for x in cl_list])
  p_min = pts.min()
  p_max = pts.max()

  pts -= p_min
  pts /= (p_max - p_min)

  start_id = 0
  for cl in cl_list:
    cl.points = pts[start_id:start_id + len(cl.points)]
    start_id += len(cl.points)

  return cl_list


def draw_classes(cl_list):
  def get_rand_color():
    return '#%06X' % (random.randint(0, 256**3 - 1))

  tsne = TSNE(n_components=2, random_state=0)
  pts = np.concatenate([x.points for x in cl_list])

  plot_pts = tsne.fit_transform(pts)

  start_id = 0
  for cl in cl_list:
    plt.plot(plot_pts[start_id:start_id + len(cl.points), 0], 
             plot_pts[start_id:start_id + len(cl.points), 1], 
      color = get_rand_color(), marker = 'o', linestyle = '', alpha = 0.7)
    start_id += len(cl.points)

  plt.show()


def dump_classes(cl_list, output_file, train_part = 1.0):
  points = np.concatenate([x.points for x in cl_list])
  labels = np.concatenate([[x.num]*len(x.points) for x in cl_list]).reshape([-1, 1])
  data = np.append(labels, points, axis = len(points.shape) - 1)

  if train_part < 1.0:
    data_size = len(data)
    test_part = int((1.0 - train_part) * data_size)
    test_indexes = np.random.choice(range(data_size), test_part, replace = False)
    train_indexes = set(range(data_size)).difference(set(test_indexes))
  
    train_data = [data[x] for x in train_indexes]
    shuffle(train_data)
    test_data = [data[x] for x in test_indexes]

    output_file = output_file.split('.')
    np.savetxt(output_file[0] + '_train.' + output_file[1], train_data, delimiter = ",", fmt = '%.9f')
    np.savetxt(output_file[0] + '_test.' + output_file[1], test_data, delimiter = ",", fmt = '%.9f')
  else:
    np.savetxt(output_file, shuffle(data), delimiter = ",", fmt = '%.9f')


def generate_spaces(classes, space):
  params = list(map(float, space[1:-1].split(',')))
  spaces = []
  for i in range(classes):
    fmt_s = '('
    for j in range(len(params) - 1):
      cc = np.random.uniform(-params[j], params[j])
      fmt_s += str(cc) + ','
    rad = np.random.uniform(1, params[-1])
    fmt_s += str(rad) + ')'
    spaces.append(ClassSpace(fmt_s))

  return spaces


if __name__ == '__main__':
  parser = ArgumentParser(description = 'Dataset generator')
  parser.add_argument('--classes', '-c',
                      type = int,
                      help = 'Count of classes to generate',
                      required = True)
  parser.add_argument('--points', '-p',
                      type = str,
                      help = 'String representing points count per class or single int for equal count',
                      required = True)
  parser.add_argument('--space', '-s',
                      type = str,
                      help = 'String containing tuples representing each class center coords and radius',
                      required = True)
  parser.add_argument('--dump', '-d',
                      type = str,
                      help = 'Dump file path',
                      default = 'dump.csv',
                      required = False)
  parser.add_argument('--train', '-t',
                      type = float,
                      help = 'Train part',
                      default = 1.0,
                      required = False)
  args = parser.parse_args()

  num_classes = int(args.classes)
  num_points = list(map(int, args.points.split(' ')))
  space = args.space.split(' ')
  if len(space) > 1:
    space = list(map(ClassSpace, space))
  else:
    space = generate_spaces(num_classes, space[0])
  dump = args.dump
  train_part = args.train

  if len(num_points) == 1:
    num_points = [num_points[0]] * num_classes
  elif len(num_points) != num_classes:
    raise RuntimeError(
      'Length of \'--points\' arg should match \'--classes\' or be \'1\'')

  for i in range(num_classes):
    space[i].populate(num_points[i])

  space = norm_classes(space)
  # draw_classes(space)
  dump_classes(space, dump, train_part)

