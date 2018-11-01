#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 10:04:28 2018

@author: haotian
"""
# import necessary packages and init
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
sns.set_style({
    'font.family': '.PingFang SC',
    #    'font.family': 'STSong',
    'axes.unicode_minus': False
})
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# read data

orgData = pd.read_excel('data.xlsx', 'data',
                        index_col=None, na_values=['#NAME?'])
category = pd.read_excel('data.xlsx', 'category_info', index_col='编号')
projData = pd.merge(orgData, category[['主题']], left_on=[
                    '群类别'], right_index=True)
if not category['数量'].equals(orgData['群类别'].value_counts().sort_index()):
    print('Verification fails!')
    exit()

# task 2

plt.figure()
sns.boxplot(x=projData['主题'], y=projData['平均年龄'])
plt.title('平均年龄在主题上分布箱图')
plt.savefig('report/figure/task2-boxplot.png', dpi=300)

# task 3


def plot_pdf(choice):
    plt.clf()
    sns.distplot(projData[choice])
    plt.title('%sPDF图' % choice)
    plt.xlabel('%s/岁' % choice)


plot_pdf('平均年龄')
plt.savefig('report/figure/task3-pdf.png', dpi=300)


def doThreeTest(group, choice):
    ksD, ksP = stats.kstest(
        (group[choice] - group[choice].mean()) / group[choice].std(), 'norm')
    ntN, ntP = stats.normaltest(
        (group[choice] - group[choice].mean()) / group[choice].std())
    spW, spP = stats.shapiro(group[choice])
    resStr = 'Skew and Kurtosis Test: N=%s, P=%s' % (ntN, ntP)
    return resStr, (ksD, ksP), (ntN, ntP), (spW, spP)


with open('report/task3q1.log', 'w', encoding='utf8') as resFile:
    print(doThreeTest(projData, '平均年龄')[0], file=resFile)

with open('report/task3q2.log', 'w', encoding='utf8') as resFile:
    for name, group in projData.groupby('群类别'):
        print('Group %s' % name, file=resFile)
        print(doThreeTest(group, '平均年龄')[0], file=resFile)

with open('report/task3q3.log', 'w', encoding='utf8') as resFile:
    lm = ols('平均年龄 ~ C(群类别)', data=projData).fit()
    reportTable = sm.stats.anova_lm(lm, typ=1)
    print(reportTable, file=resFile)

with open('report/task3std.log', 'w', encoding='utf8') as resFile:
    task5std = projData.groupby('群类别')['平均年龄'].std()
    print(task5std, file=resFile)
    print('Max std: %.3f, type %d, %s' % (task5std.max(),
                                          task5std.idxmax(),
                                          category.loc[task5std.idxmax()]['主题']),
          file=resFile)
    print('Min std: %.3f, type %d, %s' % (task5std.min(),
                                          task5std.idxmin(),
                                          category.loc[task5std.idxmin()]['主题']),
          file=resFile)
    print('MaxStd / MinStd = %.3lf' % (task5std.max() / task5std.min()),
          file=resFile)

# task 4

choices = ['性别比', '无回应比例', '图片比例']

with open('report/task4norm.log', 'w', encoding='utf8') as resFile:
    for c in choices:
        # pdf plot
        plot_pdf(c)
        plt.savefig('report/figure/task4-%s-pdf.png' % c, dpi=300)
        resStr = doThreeTest(projData, c)[0]
        print('%s %s' % (c, resStr), file=resFile)
        plt.clf()

with open('report/task4zerocount.log', 'w', encoding='utf8') as resFile:
    print('Zero count', file=resFile)
    print((projData[choices] == 0).sum(), file=resFile)

with open('report/task4lognorm0.log', 'w', encoding='utf8') as res0File:
    with open('report/task4lognorm.log', 'w', encoding='utf8') as resFile:
        for c in choices:
            # log pdf plot
            sns.distplot(np.log(1e-6 + projData[c]))
            plt.title('%s log PDF图' % c)
            plt.xlabel('%s/岁' % c)
            plt.savefig('report/figure/task4-%s-logpdf.png' % c, dpi=300)
            plt.clf()
            # log without 0 pdf plot
            cdt = projData.loc[projData[c] != 0]
            sns.distplot(np.log(cdt[c]))
            plt.title('%s 去零 log PDF图' % c)
            plt.xlabel('%s/岁' % c)
            plt.savefig('report/figure/task4-%s-0logpdf.png' % c, dpi=300)
            plt.clf()
            # log normal test
            ntN, ntP = stats.normaltest(np.log(1e-6 + projData[c]))
            print('%s Skew and Kurtosis Test: N=%s, P=%s' %
                  (c, ntN, ntP), file=resFile)
            # log without 0 normal test
            ntN, ntP = stats.normaltest(np.log(cdt[c]))
            print('%s Skew and Kurtosis Test: N=%s, P=%s' %
                  (c, ntN, ntP), file=res0File)
