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
    # 'font.family': '.PingFang SC',
    'font.family': 'STSong',
    'axes.unicode_minus': False
})
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
import itertools

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
        # TODO: put them in subplots
        plot_pdf(c)
        plt.savefig('report/figure/task4-%s-pdf.png' % c, dpi=300)
        resStr = doThreeTest(projData, c)[0]
        print('%s %s' % (c, resStr), file=resFile)
        plt.clf()
    for c in choices:
        print('%s MaxStd / MinStd = %.2f' %
              (c, projData.groupby('主题')[c].std().max(
              ) / projData.groupby('主题')[c].std().min()),
              file=resFile)

with open('report/task4zerocount.log', 'w', encoding='utf8') as resFile:
    print('Zero count', file=resFile)
    print((projData[choices] == 0).sum(), file=resFile)

with open('report/task4lognorm0.log', 'w', encoding='utf8') as res0File:
    with open('report/task4lognorm.log', 'w', encoding='utf8') as resFile:
        for c in choices:
            # log pdf plot
            # TODO: put them in subplots
            sns.distplot(np.log(1e-6 + projData[c]))
            plt.title('%s log PDF图' % c)
            plt.xlabel('%s/岁' % c)
            plt.savefig('report/figure/task4-%s-logpdf.png' % c, dpi=300)
            plt.clf()
            # log without 0 pdf plot
            # TODO: put them in subplots
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
        for c in choices:
            lcdt = projData.groupby('主题')[c].apply(
                lambda d: np.log(d + 1e-6).std())
            print('%s MaxStd / MinStd = %.2f' %
                  (c, lcdt.max() / lcdt.min()),
                  file=resFile)
            lcdt = projData.loc[projData[c] != 0].groupby(
                '主题')[c].apply(lambda d: np.log(d).std())
            print('%s MaxStd / MinStd = %.2f' %
                  (c, lcdt.max() / lcdt.min()),
                  file=res0File)

# task 5
with open('report/task5kwtest.log', 'w', encoding='utf8') as resFile:
    for c in choices:
        gp = projData.groupby('主题')[c]
        gpl = list()
        for gpn in gp.groups:
            gpl.append(gp.get_group(gpn))
        kwS, kwP = stats.kruskal(*gpl)
        print('%s K-W Test: s=%s, p=%s' % (c, kwS, kwP), file=resFile)

for c in choices:
    sns.violinplot(x='主题', y=c, data=projData)
    plt.title('%s在主题上分布小提琴图' % c)
    plt.savefig('report/figure/task5-%s-boxplot.png' % c, dpi=300)
    plt.clf()

# task 6

choices = ['性别比', '无回应比例', '图片比例', '平均年龄']


def ftest_theme(dt, c):
    gp = dt.groupby('主题')[c]
    gpl = list()
    for gpn in gp.groups:
        gpl.append(gp.get_group(gpn))
    fvalue, pvalue = stats.f_oneway(*gpl)
    return fvalue, pvalue

dt = {}
for c in choices:
    randfs = list()
    groupfs = list()
    weightfs = list()
    gwfs = list()
    for t in range(10):
        org_sample = projData
        rand_sample = projData.sample(frac=0.1)
        group_sample = projData.groupby('主题').apply(
            lambda d: d.sample(frac=0.1))
        weight_sample = projData.sample(frac=0.1, weights='群人数')
        gw_sample = projData.groupby('主题').apply(
            lambda d: d.sample(frac=0.1, weights='群人数'))
        rand_f, rand_p = ftest_theme(rand_sample, c)
        group_f, group_p = ftest_theme(group_sample, c)
        weight_f, weight_p = ftest_theme(weight_sample, c)
        gw_f, gw_p = ftest_theme(gw_sample, c)
        randfs.append(rand_f)
        groupfs.append(group_f)
        weightfs.append(weight_f)
        gwfs.append(gw_f)
    res = pd.DataFrame({'org': orgfs,
                'rand':randfs,
                'group': groupfs,
                'weight': weightfs,
                'group-weight': gwfs})
    dt[c] = res.var()

res = pd.DataFrame(dt)
with open('report/task6-fvar.log', 'w', encoding='utf8') as resFile:
    print(res, file=resFile)
res = res.apply(lambda d: (d - d.mean()) / d.std())
res.transpose().plot(kind='bar')
plt.savefig('report/figure/task6-favr.png', dpi=300)


# task 7

def get_groups(gpdata, groups):
    res = pd.DataFrame()
    for g in groups:
        res = res.append(gpdata.get_group(g))
    return res

testRatio = 0.1
classes = ['同学会', '业主', '投资理财', '行业交流', '游戏']
features = ['性别比', '群人数', '消息数', '稠密度', '年龄差', '平均年龄', '地域集中度', '手机比例', '会话数', '无回应比例', '夜聊比例', '图片比例']
# norm_cols = ['年龄差', '消息数', '群人数', '平均年龄']
norm_cols = ['性别比', '群人数', '消息数', '稠密度', '年龄差', '平均年龄', '地域集中度', '手机比例', '会话数', '无回应比例', '夜聊比例', '图片比例']
normData = projData.apply(lambda d: (d - d.mean()) / d.std() if d.name in norm_cols else d)

lrdata = get_groups(normData.groupby('主题'), classes)
X_train, X_test, y_train, y_test = train_test_split(lrdata[features], lrdata['群类别'], test_size=testRatio)
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
clfsvm = svm.SVC(C=1.5).fit(X_train, y_train)
with open('report/task6-multi.log', 'w', encoding='utf8') as resFile:
    print('Logistic Regression accur. = %.2f%%' % (100 * np.mean(y_pred == y_test)), file=resFile)
    print('Support Vector Machine accur. = %.2f%%' % (clfsvm.score(X_test, y_test) * 100), file=resFile)

with open('report/task6-two.log', 'w', encoding='utf8') as resFile:
    for fs in itertools.combinations(classes, 2):
        lrdata = get_groups(normData.groupby('主题'), fs)
        X_train, X_test, y_train, y_test = train_test_split(lrdata[features], lrdata['群类别'], test_size=testRatio)
        clf = LogisticRegression().fit(X_train, y_train)
        clfsvm = svm.SVC(C=1.5).fit(X_train, y_train)
        print(fs, file=resFile)
        print('Logistic Regression accur. = %.2f%%, Support Vector Machine accur. = %.2f%%' % (clf.score(X_test, y_test) * 100, clfsvm.score(X_test, y_test) * 100), file=resFile)
