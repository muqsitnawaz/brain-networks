import matplotlib.pyplot as plt
import numpy as np

import sys
import glob
import errno

def parse_file(fname):
    f = open(fname, 'r')
    lines = f.readlines()

    data = {}
    for line in lines:
        if "Runtime of LP" in line:
            data['runtime'] = eval(line[line.index(':')+1:])
            continue
        if "Cost of LP" in line:
            data['cost'] = eval(line[line.index(':')+1:])
        if "whose capacity was increased" in line:
            data['num_incr'] = eval(line[line.index(':')+1:])

    f.close()
    return data

def parse_filename(fname):
    params = fname[fname.rindex('/')+1:].split('-')
    return {'g': params[1], 'k': params[3]}

def plot_g_vs_stats(exp_data):
    x = list(range(4, 16))
    y1, y2, y3 = [], [], []
    for i in range(4, 16):
        y1.append(sum(list(map(lambda d: d['runtime'], exp_data[str(i)])))/30)
        y2.append(sum(list(map(lambda d: d['cost'], exp_data[str(i)])))/30)
        y3.append(sum(list(map(lambda d: d['num_incr'], exp_data[str(i)])))/30)

    y2 = [y/160 for y in y2]
    y3 = [y/20 for y in y2]

    plt.title('Effect of Increasing Graph Size on LP Solution')
    plt.xticks(np.arange(4,16))
    plt.xlabel('Size of Graph')
    plt.ylabel('LP Solution')

    plt.plot(x, y1, linestyle='--', marker='o', color='b', label='runtime (s)')
    plt.plot(x, y2, linestyle='--', marker='o', color='r', label='avg. lp cost / 160')
    plt.plot(x, y3, linestyle='--', marker='o', color='g', label='avg. edges increased / 20')

    plt.legend(bbox_to_anchor=(0.025, 0.80), loc="lower left", borderaxespad=0.)
    plt.show()

def plot_k_vs_stats(exp_data):
    x = list(range(1, 19))
    y1, y2, y3 = [], [], []
    for i in range(1, 19):
        y1.append(sum(list(map(lambda d: d['runtime'], exp_data[str(i)])))/30)
        y2.append(sum(list(map(lambda d: d['cost'], exp_data[str(i)])))/30)
        y3.append(sum(list(map(lambda d: d['num_incr'], exp_data[str(i)])))/30)

    y2 = [y/10 for y in y2]

    plt.title('Effect of Increasing s-t Pairs on a 10 by 10 Graph')
    plt.xticks(np.arange(1,19))
    plt.xlabel('Number of s-t Pairs')
    plt.ylabel('LP Solution')

    plt.plot(x, y3, linestyle='--', marker='o', color='b', label='runtime (s)')
    plt.plot(x, y1, linestyle='--', marker='o', color='r', label='avg. lp cost / 10')
    plt.plot(x, y2, linestyle='--', marker='o', color='g', label='avg. edges increased')

    plt.legend(bbox_to_anchor=(0.025, 0.80), loc="lower left", borderaxespad=0.)
    plt.show()

def main():
    # Runtime experiments
    path = './results/runtime/*.exp'  
    filenames = glob.glob(path)

    exp_data = {}
    for filename in filenames:
        exp_data[parse_filename(filename)['g']] = exp_data.get(parse_filename(filename)['g'], []) + [parse_file(filename)]

    plot_g_vs_stats(exp_data)

    # St-pairs experiments
    path = './results/st-pairs/*.exp'  
    filenames = glob.glob(path)

    exp_data = {}
    for filename in filenames:
        exp_data[parse_filename(filename)['k']] = exp_data.get(parse_filename(filename)['k'], []) + [parse_file(filename)]

    plot_k_vs_stats(exp_data)

if __name__ == "__main__":
    main()