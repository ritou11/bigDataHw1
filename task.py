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
sns.set_style('ticks')
sns.set_style({
    'font.family': '.PingFang SC',
    # 'font.family': 'STSong',
    'axes.unicode_minus': False })
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
plt.savefig('report/meta/fig/task2-boxplot.png', dpi=300)
plt.clf()

# task 3


def plot_pdf(choice):
    sns.distplot(projData[choice])
    plt.title('%sPDF图' % choice)
    plt.xlabel('%s/岁' % choice)


plot_pdf('平均年龄')
plt.savefig('report/meta/fig/task3-pdf.png', dpi=300)


def doThreeTest(group, choice):
    ksD, ksP = stats.kstest(
        (group[choice] - group[choice].mean()) / group[choice].std(), 'norm')
    ntN, ntP = stats.normaltest(
        (group[choice] - group[choice].mean()) / group[choice].std())
    spW, spP = stats.shapiro(group[choice])
    resStr = 'Skew and Kurtosis Test: N=%s, P=%s' % (ntN, ntP)
    return resStr, (ksD, ksP), (ntN, ntP), (spW, spP)


with open('report/meta/task3q1.log', 'w', encoding='utf8') as resFile:
    print(doThreeTest(projData, '平均年龄')[0], file=resFile)

with open('report/meta/task3q2.log', 'w', encoding='utf8') as resFile:
    for name, group in projData.groupby('群类别'):
        print('Group %s' % name, file=resFile)
        print(doThreeTest(group, '平均年龄')[0], file=resFile)

with open('report/meta/task3q3.tex', 'w', encoding='utf8') as resFile:
    lm = ols('平均年龄 ~ C(群类别)', data=projData).fit()
    reportTable = sm.stats.anova_lm(lm, typ=1)
    print(reportTable.to_latex(), file=resFile)

gp = projData.groupby('主题')['平均年龄']
figure = plt.figure()
for gpn in gp.groups:
    sns.kdeplot(data=gp.get_group(gpn), shade=False, label=gpn)
sns.kdeplot(data=projData['平均年龄'], shade=True, label='总体')
plt.title('不同群类别平均年龄经验概率分布')
plt.xlabel('年龄/岁')
plt.tight_layout(h_pad=2)
plt.savefig('report/meta/fig/task3q3.png', dpi=300)
ylim = plt.ylim()
xlim = plt.xlim()
plt.clf()

x = np.linspace(5, 50, 1000)
for gpn in gp.groups:
    y = stats.norm.pdf(x, gp.get_group(gpn).mean(), gp.get_group(gpn).std())
    sns.lineplot(x, y, label=gpn)
    plt.plot(np.ones(10) * gp.get_group(gpn).mean(), np.linspace(np.max(y) - 0.02, np.max(y) + 0.02, 10), 'k-.')
y = stats.norm.pdf(x, projData['平均年龄'].mean(), projData['平均年龄'].std())
sns.lineplot(x, y, label='总体')
plt.fill_between(x, y, color=sns.color_palette()[5], alpha=0.3)
plt.plot(np.ones(10) * projData['平均年龄'].mean(), np.linspace(*ylim, 10), 'k--')
plt.ylim(ylim)
plt.xlim(xlim)
plt.title('不同群类别平均年龄参数正态分布')
plt.xlabel('年龄/岁')
plt.savefig('report/meta/fig/task3q3-norm.png', dpi=300)
plt.clf()

with open('report/meta/task3std.log', 'w', encoding='utf8') as resFile:
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

with open('report/meta/task4norm.log', 'w', encoding='utf8') as resFile:
    N = len(choices)
    f, axs = plt.subplots(1, N, sharey=False, figsize=(12, 5))
    for i, c in enumerate(choices):
        # pdf plot
        sns.distplot(projData[c], ax=axs[i])
        axs[i].set_title('%sPDF图' % c)
        plt.xlabel('%s/岁' % c)
        resStr = doThreeTest(projData, c)[0]
        print('%s %s' % (c, resStr), file=resFile)
    plt.tight_layout(h_pad=2)
    plt.savefig('report/meta/fig/task4-pdf.png', dpi=300)
    plt.clf()
    for c in choices:
        print('%s MaxStd / MinStd = %.2f' %
              (c, projData.groupby('主题')[c].std().max(
              ) / projData.groupby('主题')[c].std().min()),
              file=resFile)

with open('report/meta/task4zerocount.tex', 'w', encoding='utf8') as resFile:
    zeros = (projData[choices] == 0).sum()
    zeros.name = '零值数量'
    print(zeros.to_latex(), file=resFile)

with open('report/meta/task4lognorm0.log', 'w', encoding='utf8') as res0File:
    with open('report/meta/task4lognorm.log', 'w', encoding='utf8') as resFile:
        f, axs = plt.subplots(2, N, sharey=False, figsize=(14, 10))
        for i, c in enumerate(choices):
            # log pdf plot
            sns.distplot(np.log(1e-6 + projData[c]), ax=axs[0][i])
            axs[0][i].set_title('%s log PDF图' % c)
            axs[0][i].set_xlabel('%s/岁' % c)
            # log without 0 pdf plot
            cdt = projData.loc[projData[c] != 0]
            sns.distplot(np.log(cdt[c]), ax=axs[1][i])
            axs[1][i].set_title('%s 去零 log PDF图' % c)
            axs[1][i].set_xlabel('%s/岁' % c)
            # log normal test
            ntN, ntP = stats.normaltest(np.log(1e-6 + projData[c]))
            print('%s Skew and Kurtosis Test: N=%s, P=%s' %
                  (c, ntN, ntP), file=resFile)
            # log without 0 normal test
            ntN, ntP = stats.normaltest(np.log(cdt[c]))
            print('%s Skew and Kurtosis Test: N=%s, P=%s' %
                  (c, ntN, ntP), file=res0File)
        plt.tight_layout(h_pad=2)
        plt.savefig('report/meta/fig/task4-logpdf.png', dpi=300)
        plt.clf()
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
with open('report/meta/task5kwtest.log', 'w', encoding='utf8') as resFile:
    for c in choices:
        gp = projData.groupby('主题')[c]
        gpl = list()
        for gpn in gp.groups:
            gpl.append(gp.get_group(gpn))
        kwS, kwP = stats.kruskal(*gpl)
        print('%s K-W Test: s=%s, p=%s' % (c, kwS, kwP), file=resFile)


f, axs = plt.subplots(N, 1, sharey=False, figsize=(10, 7))
for i, c in enumerate(choices):
    sns.violinplot(x='主题', y=c, data=projData, ax=axs[i])
    axs[i].set_title('%s在主题上分布小提琴图' % c)
plt.tight_layout(h_pad=2)
plt.savefig('report/meta/fig/task5-boxplot.png', dpi=300)
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
dtm = {}
for c in choices:
    randfs = list()
    groupfs = list()
    weightfs = list()
    gwfs = list()
    for t in range(10):
        rand_sample = projData.sample(frac=0.1, random_state=t)
        group_sample = projData.groupby('主题').apply(
            lambda d: d.sample(frac=0.1, random_state=t))
        weight_sample = projData.sample(frac=0.1, weights='群人数', random_state=t)
        gw_sample = projData.groupby('主题').apply(
            lambda d: d.sample(frac=0.1, weights='群人数', random_state=t))
        rand_f, rand_p = ftest_theme(rand_sample, c)
        group_f, group_p = ftest_theme(group_sample, c)
        weight_f, weight_p = ftest_theme(weight_sample, c)
        gw_f, gw_p = ftest_theme(gw_sample, c)
        randfs.append(rand_f)
        groupfs.append(group_f)
        weightfs.append(weight_f)
        gwfs.append(gw_f)
    res = pd.DataFrame({'rand':randfs,
                'group': groupfs,
                'weight': weightfs,
                'group-weight': gwfs})
    dt[c] = res.var()
    dtm[c] = res.mean()

res = pd.DataFrame(dt)
with open('report/meta/task6-fvar.tex', 'w', encoding='utf8') as resFile:
    print(res.to_latex(), file=resFile)
res = res.apply(lambda d: (d - d.mean()) / d.std())
res.transpose().plot(kind='bar', rot=0)
plt.savefig('report/meta/fig/task6-fvar.png', dpi=300)
plt.clf()

res = pd.DataFrame(dtm)
with open('report/meta/task6-fmean.tex', 'w', encoding='utf8') as resFile:
    print(res.to_latex(), file=resFile)
# task 7

sigmoid_x = np.linspace(-10, 10, 100)
sigmoid_y = 1 / (1 + np.exp(- sigmoid_x))
sns.lineplot(sigmoid_x, sigmoid_y)
plt.xlabel('z')
plt.ylabel('Sigmoid(z)')
plt.title('Sigmoid函数图像')
plt.savefig('report/meta/fig/task7-sigmoid.png', dpi=300)
plt.clf()

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
X_train, X_test, y_train, y_test = train_test_split(lrdata[features], lrdata['群类别'], test_size=testRatio, random_state=2333)
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
clfsvm = svm.SVC(C=1.5).fit(X_train, y_train)
with open('report/meta/task7-multi.log', 'w', encoding='utf8') as resFile:
    print('Logistic Regression accur. = %.2f%%' % (100 * np.mean(y_pred == y_test)), file=resFile)
    print('Support Vector Machine accur. = %.2f%%' % (clfsvm.score(X_test, y_test) * 100), file=resFile)

with open('report/meta/task7-two.log', 'w', encoding='utf8') as resFile:
    print('Features: %s' % ','.join(features), file=resFile)
    for fs in itertools.combinations(classes, 2):
        lrdata = get_groups(normData.groupby('主题'), fs)
        X_train, X_test, y_train, y_test = train_test_split(lrdata[features], lrdata['群类别'], test_size=testRatio, random_state=2333)
        clf = LogisticRegression().fit(X_train, y_train)
        clfsvm = svm.SVC(C=1.5).fit(X_train, y_train)
        print(fs, file=resFile)
        print('Logistic accur. = %.2f%%, SVM accur. = %.2f%%' % (clf.score(X_test, y_test) * 100, clfsvm.score(X_test, y_test) * 100), file=resFile)
        print('Coef. = [%s]' % (','.join(map(lambda f: '%.2f' % f, clf.coef_[0]))), file=resFile)
