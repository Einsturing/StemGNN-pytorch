import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import math


def figsave(df, label1=[], label2=[], v=0, name='-dual.png'):
    print('-----------------------', str(label1) +
          '.png', '--------------------------')
    plt.subplots(figsize=(17, 8.5))
    plt.scatter(df.x, df.y, s=1, c=df[label1], cmap='Blues_r')
    # valididx = np.where(df[label2] > 0)
    valididx = np.where(df[label1] - df[label2] > v)
    print(valididx[0], df[label2][valididx[0]] - df[label1][valididx[0]])
    gt = plt.scatter(df.x[valididx[0]], df.y[valididx[0]], s=150,
                     c=df[label1][valididx[0]] - df[label2][valididx[0]], cmap='Greens')
    valididx = np.where(df[label1] - df[label2] < -1 * v)
    print(valididx[0], df[label1][valididx[0]] - df[label2][valididx[0]])
    lt = plt.scatter(df.x[valididx[0]], df.y[valididx[0]], s=150,
                     c=df[label2][valididx[0]] - df[label1][valididx[0]], cmap='Oranges')

    hc = plt.colorbar(gt)
    plt.clim(0, 1)
    hc.set_label('lower')
    lc = plt.colorbar(lt)
    plt.clim(0, 1)
    lc.set_label('higher')
    plt.title(str(label2) + '-d-' + str(v))
    plt.savefig(str(label2) + '-d-' + str(v) + name)


def display(df, label1=[], i=0, v=0, name='-p.png'):
    plt.subplots(figsize=(17, 8.5))
    gt = plt.scatter(df.x, df.y, s=250, c=df[label1], cmap='rainbow')
    plt.clim(-1, 1)
    hc = plt.colorbar(gt)
    hc.set_label('higher')
    plt.title(str(label1) + '-d-' + str(v))
    plt.savefig(str(label1) + name)
