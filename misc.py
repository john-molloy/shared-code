# A handy place to put useful bits of code

#%%

os.listdir('.')
os.listdir('..')


#%%

def log_time():
    now = datetime.datetime.now()
    timestamp = '{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}_{6:02d}'.format(
        now.year, now.month, now.day,
        now.hour, now.minute, now.second,
        int(round(now.microsecond / 10000)))
    return timestamp


#%%

def jm_file(process: str = None):
    print(f'  process: [{process}]')

    if not process:
        print('    false - if not process')
    else:
        print('    true  - else:')


if __name__ == "__main__":
    jm_file()
    jm_file(True)
    jm_file(False)
    jm_file("")
    jm_file("abc")
    jm_file("True")
    jm_file("False")


#%%
# r: litteral
# f: interpret \n and substitute for {var}

dummy = 999

print(r'ab\ncde')                       # Exactly as-is: "ab\ncde"
print(f'ab\ncde')                       # With newline: "ab  cde"

print(r'ab\ncd{0}e'.format(dummy))      # "ab\ncd999e"
print(f'ab\ncd{0}e'.format(dummy))      # With newline: "ab  cd0e"

print(r'ab\ncd{dummy}e')                # "ab\ncd{dummy}e"
print(f'ab\ncd{dummy}e')                # With newline: "ab  cd999e"


#%%

d1 = {}
d1['k'] = ['a', 'b', 'c', 'd']
d1['c1'] = [1, np.nan, 3, 4]
d1['c2'] = [10, 20, 30, 40]
d1['c3'] = [100, 200, np.nan, 400]
d = pd.DataFrame(d1)
d
d.sum()
d.sum(axis=0)
d.sum(axis=1)


#%%

from datetime import datetime

print(datetime.now().strftime("%Y%m%d_%H%M%S"))


#%%
# That function...
# predict.kmeans <- function(centers, newdata) {
#   apply(newdata, 1, function(r) which.min(colSums((t(centers) - r)^2)))
# }
#
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.vq.html
# The features in obs should have unit variance, which can
# be achieved by passing them through the whiten function.
# [JM: I have used then'scale' function (or similar name).


import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
import numpy as np


d = {}
d['clu'] = [ 1,  2,  3]
d['frq'] = [10.5, 15.5, 10.5]
d['rec'] = [10.5, 15.5, 15.5]
cent = pd.DataFrame(d)
del(d)

d = {}
d['frq'] = 5 + 15 * np.random.ranf(10)
d['rec'] = 5 + 15 * np.random.ranf(10)
prep = pd.DataFrame(d)
del(d)

clusters, distances = vq(prep, cent[['frq', 'rec']])

plt.scatter(cent['frq'], cent['rec'], c='red', s=50)
plt.scatter(prep['frq'], prep['rec'], c=clusters, s=1)
plt.show()

# I'm not sure what this is doing, even though it looks correct.
cent
cent['clu']
cent['clu'][clusters]

prep['cluster'] = cent['clu'][clusters]

prep.drop('cluster', axis=1, inplace=True)


# BUT!!! We need a way of finding the cluster number
# rather than the index of the DataFrame.






#%%

d = {}
d['c1'] = ['a', 'b', 'c', 'd']
d['c2'] = [  1,   2,   3,   4]
d['c3'] = [ 10,  20,  30,  40]
d['c4'] = [100, 200, 300, 400]
df = pd.DataFrame(d)
print(df)

df['s1'] = df.sum(axis=1)
print(df)

col_list = ['c2', 'c4']

df['s2'] = df[col_list].sum(axis=1)
print(df)

df['s3'] = 0
df.loc[df[col_list].sum(axis=1) > 300, 's3'] = 1
print(df)

col_list = ['c1', 'c2', 's3']
df = df[col_list]
print(df)


#%%

d = {}
d['c1'] = ['a', 'b', 'c', 'd']
d['c2'] = [  1,   2,   np.nan,   4]
d['c3'] = [ 10,  20,  30,  40]
d['c4'] = [100, 200, 300, np.nan]
df = pd.DataFrame(d)
print(df)

df.dropna()
df.dropna(axis=0)
df.dropna(axis=1)

df.dropna(axis=0, inplace=True)
print(df)


#%%
# Review random numbers again.

np.random.rand()                # A single number.
np.random.rand(1)               # A single number as an aarray.
np.random.rand(10)              # Ten numbers in an array.
np.random.rand(5, 2)            # Ten numbers is a 5x2 array.
np.random.rand(5, 2, 2)         # Twenty numbers in a three dimensional array.

np.random.randint()             # Invalid.
np.random.randint(4)            # Integer in range 0 to 3.
np.random.randint(10, 14)       # Integer in range 10 to 13.
np.random.randint(0, 4, 5)      # Five integers in range 0 to 3 in an array.
np.random.randint(0, 4, (5, 2)) # Integers in range 0 to 3 in a 5x2 array.

q = ['a', 'b', 'c', 'd']
np.random.choice(q, 3)                      # Choose three, with replacement.
np.random.choice(q, 3, replace=True)        # Choose three, with replacement.
np.random.choice(q, 3, replace=False)       # Choose three, without replacement.
np.random.choice(q, 5, replace=False)       # Fails.
np.random.choice(q, 5, replace=True)        # Works, because there is replacement.
np.random.choice(q, 3, p=[0.7, 0.1, 0.1, 0.1])                  # With probability
np.random.choice(q, 3, replace=True,  p=[0.7, 0.1, 0.1, 0.1])   # Equivalent.
np.random.choice(q, 3, replace=False, p=[0.7, 0.1, 0.1, 0.1])   # Works with no replacement.  Determines the probable order, I reckon.
del q


np.random.randrange(11, 14)     # Invalid
np.random.randrange(4)


#%%

d = {}
d['c1'] = list("abcabc")
d['c2'] = list("aabbcc")
d['v1'] = [  1 * q for q in range(len(d['c1']))]
d['v2'] = [ 10 * q for q in d['v1']]
d['v3'] = [100 * q for q in d['v1']]
df = pd.DataFrame(d)
print(df)

df[df['c1'] == 'b']
df.loc[df['c1'] == 'b']
df.loc[(df['c1'] == 'b') | (df['c1'] == 'c')]
df.loc[(df['c1'] == 'c') & (df['c2'] == 'c')]


#%%
# https://stackoverflow.com/questions/47551557/glm-in-python-vs-r/47664290

import statsmodels.api as sm

df = pd.DataFrame({"Survived": [0,0,1,1,0],
                   "Sex": ["Male", "Female", "Female", "Male", "Male"]})

model = sm.formula.glm("Survived ~ C(Sex, Treatment(reference='Female'))",
                       family=sm.families.Binomial(), data=df).fit()

print(model.summary())


#%%

d = {}
d['y'] = list("0000011110")
d['x'] = list("mmmmmfffff")
df = pd.DataFrame(d)

model = sm.formula.glm(
                "y ~ C(x, Treatment(reference='f'))",
                family=sm.families.Binomial(),
                data=df).fit()

print(model.summary())


#%%

import pandas as pd
from scipy.stats import chisquare

d = {}
d['c1'] = list("aaaabbbb")
d['c2'] = list("xxyyyyzz")
df = pd.DataFrame(d)

pd.crosstab(df['c1'], df['c2'])

chisquare(pd.crosstab(df['c1'], df['c2']))


#%%

d = {}
d['c1'] = [5,2,5,3,3,2]
d['c2'] = [2,3,3,5,5,1]
d['c3'] = [2,2,4,2,6,4]
df = pd.DataFrame(d)

pd.crosstab(df['c1'], df['c2'])

chisquare(pd.crosstab(df['c1'], df['c2']))


#%%

import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf


#%%
# Very simple test.
# Just one independent variable.

d = {}
# Dependent variable - always equal numbers of each.
d['y'] = list("0000000011111111")
#d['y'] = list("2222222211111111")
#d['y'] = list("0000000099999999")
#d['y'] = list("MMMMMMMMWWWWWWWW")
#
# Change the freqs of the independent variable.
d['x'] = list("mmmmmmffmmffffff")
#
dftrain = pd.DataFrame(d)

# print(dftrain)

model = smf.glm(formula='y ~ x',
                data=dftrain,
                family=sm.families.Binomial()).fit()
print(model.summary())

d = {}
d['x'] = list("mf")
dftest = pd.DataFrame(d)
#print(dftest)

res = model.predict(dftest)

rescmp = dftest.copy()
rescmp['r'] = round(res, 2)

print(rescmp)


#%%
# Binary
# The specified length include the leading  '0b'.

for i in range(2**4):
    b = format(i, '#07b')
    bb = b[2:]
    print(b, bb, len(b), len(bb))


#%%

print("The quick brown fox.".find("The"))
print("The quick brown fox.".find("he"))
print("The quick brown fox.".find("fox"))
print("The quick brown fox.".find("qqq"))
print("The quick brown fox.".find("the"))


#%%
# Create the binary strings another way.
# format(i, '#013b')
# What am I doing wrong here?

def func(i):
    #return format(i, '#013b')
    return 2 * i

df = pd.DataFrame({'v': range(2**3)})
df['f'] = df.apply(func)
print(df)

df = pd.DataFrame({'v': range(2**3)})
df['f'] = df.apply(lambda i: 5 * i)
print(df)

df = pd.DataFrame({'v': range(2**3)})
df['f'] = df.apply(lambda i: func(i))
print(df)


df = pd.DataFrame({'v': np.linspace(0, 2 * 3.14159, 9)})
df['sin'] = df.apply(np.sin)
print(df)

func(0)
func(1)
func(5)


#%%
#
# Venn diagrams
#
# https://pypi.org/project/matplotlib-venn/
# https://towardsdatascience.com/how-to-create-and-customize-venn-diagrams-in-python-263555527305
#
# pip install matplotlib-venn
#

#Import libraries
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted
from matplotlib_venn import venn3, venn3_circles
from matplotlib import pyplot as plt
# %matplotlib inline

from matplotlib_venn import venn3_unweighted

import matplotlib.pyplot as plt


#%%

venn2(subsets = (30, 10, 5), set_labels = ('Group A', 'Group B'))
venn2_circles(subsets = (30, 10, 5))

venn2_unweighted(subsets = (30, 10, 5), set_labels = ('Group A', 'Group B'))
plt.savefig('test.png')

venn3(subsets = (1, 2, 3, 4, 5, 6, 7),
      set_labels = ('Group A', 'Group B', 'Group C'),
      alpha = 0.5);


#%%
# TODO: keep

df = pd.DataFrame()
df['value'] = list('wxyz')
df.index = list('abcd')
print(df)
df.reset_index(inplace=True, drop=True)
print(df)


#%%
# TODO: keep

df = pd.DataFrame()
df['v'] = list('abbccc')

df
df.duplicated()
df.duplicated(keep='first')
df.duplicated(keep='last')
df.duplicated(keep=False)


#%%
# TODO: keep_new

df = pd.DataFrame()
df['v'] = list('abbccc')
df['x'] = range(6)
print(df)

df.duplicated(subset=['v'])
df.duplicated(subset=['v'], keep='first')
df.duplicated(subset=['v'], keep='last')
df.duplicated(subset=['v'], keep=False)

df[~df.duplicated(subset=['v'], keep='first')]

# This was used in assessments/concerns.
# Keep only those where 'v' appears more
# than once.
df[df.duplicated(subset=['v'], keep=False)]

# Keep only the second entry.
df2 = df[df.duplicated(subset=['v'], keep='first')]
df2
df2[~df2.duplicated(subset=['v'], keep='first')]

del df, df2


#%%
# TODO: keep

df = pd.DataFrame()
df['v'] = range(10, 16)
print(df)
df['sh'] = df.shift(1)
print(df)


#%%
# TODO: keep

df = pd.DataFrame()
df['id'] = list('abbccc')
df['va'] = range(6)
print(df)

df.groupby('id')['va'].sum()

df['id']
df['id'].isin(['a', 'c'])
df.loc[df['id'].isin(['a', 'c'])]
df.loc[~df['id'].isin(['a', 'c'])]


#%%
# TODO: keep

import numpy as np
import pandas as pd

n = 10
df = pd.DataFrame()
df['id'] = np.random.choice(list('abc'), n, replace=True)
df['roll'] = np.random.choice(range(100*n), n, replace=False)
print(df)

df = df.sort_values(by = ['id', 'roll']).reset_index(drop=True)
print(df)

df['rank'] = df.groupby('id')['roll'].rank(ascending=1).astype(int)
print(df)

df.groupby('id')['rank'].unique()

df.groupby('id')['roll'].max()

del n, df


#%%
# TODO: keep
# Remove ids that date of last assessment was 30 days less after their first

Assessments_dd = Assessments_dup.loc[~Assessments_dup['id'].isin(days_diff.loc[days_diff['diff_days'] < 30]['id'].values.T.tolist())]

days_diff['diff_days'] < 30
days_diff.loc[days_diff['diff_days'] < 30]
days_diff.loc[days_diff['diff_days'] < 30]['id']
days_diff.loc[days_diff['diff_days'] < 30]['id'].values
days_diff.loc[days_diff['diff_days'] < 30]['id'].values.T
days_diff.loc[days_diff['diff_days'] < 30]['id'].values.T.tolist()

Assessments_dup.loc[~Assessments_dup['id'].isin(days_diff.loc[days_diff['diff_days'] < 30]['id'].values.T.tolist())]


#%%
# TODO: keep
# Remove assessments that have days gap larger than a year
Assessments_ddyr = Assessments_dd.loc[~((Assessments_dd['large_gap'].shift(-1)=='Y') | (Assessments_dd['large_gap']=='Y'))]

Assessments_dd.loc[
    ~(
        (Assessments_dd['large_gap'].shift(-1) == 'Y')
      | (Assessments_dd['large_gap'] == 'Y')
    )
]



#%%
# TODO: keep
# After a pd.merge()

Assessments_cnr['_merge'].value_counts()
# both          167277
# left_only       7272
# right_only         0      # Interesting - how does it know about right_only?
# Name: _merge, dtype: int64
Assessments_cnr['_merge'].unique()
# ['both', 'left_only']
# Categories (2, object): ['both', 'left_only']     # jm: Note 'Categories'.
Assessments_cnr['_merge'].nunique()
# 2


df = pd.DataFrame()
df['id'] = np.random.choice(list('abc'), 10, replace=True)
df['id'].value_counts()
# b    5
# a    3
# c    2
# Name: id, dtype: int64
df['id'].unique()
# array(['a', 'b', 'c'], dtype=object)
df['id'].nunique()
# 3


#%%

from math import sqrt
from scipy.stats import sem

q = [1, 3, 6, 6, 6, 6, 6]
qq = len(q)

print("std ", np.std(q))
print("std0", np.std(q, ddof=0))
print("std1", np.std(q, ddof=1))

print("sem ", sem(q))
print("xxx ", np.std(q, ddof=1) / np.sqrt(qq))


#%%
# TODO: keep
# Zoi
# Continue from here (28/05/2021)

import numpy as np
import pandas as pd
from numpy import mean
from scipy.stats import sem
from scipy.stats import t


# t-test for independent samples
# function for calculating the t-test for two independent samples
def ttest(data1, data2, alpha):
    # calculate means
    mean1, mean2 = mean(data1), mean(data2)
    # calculate standard errors
    se1, se2 = sem(data1), sem(data2)
    # standard error on the difference between the samples
    sed = sqrt(se1**2.0 + se2**2.0)
    # calculate the t statistic
    t_stat = (mean1 - mean2) / sed
    # degrees of freedom
    df = len(data1) + len(data2) - 2
    # calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
    # calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
    # return everything
    return t_stat, df, cv, p


#%%
# TODO: keep

res = pd.DataFrame(columns=['stdev', 'reject', 'result'])

n = 10

for q in np.linspace(0.0, 1.0, 11):

    data1 = np.linspace(0, 1, n)
    data2 = data1 + q
    #data2 = data1 + np.random.normal(0, q, n)

    alpha = 0.05
    t_stat, df, cv, p = ttest(data1, data2, alpha)

    if p > alpha:
        result = 'Accept null hypothesis that the means are equal.'
        reject = 'N'
    else:
        result = 'Reject the null hypothesis that the means are equal.'
        reject = 'Y'

    stats = {'stdev': q, 'reject': reject, 'result': result}

    res = res.append(stats, ignore_index=True)

print(res)


#%%
# TODO: keep
# Boolean indexing on a DataFrame where
# the length of the boolean array does
# not match the length of the DataFrame.
#
# The takeaway message is that it works
# if the boolean array is too long so
# long as it is a pandas Series.

df1 = pd.DataFrame({'a': range(4)})
df2 = pd.DataFrame({'a': range(7)})     # Longer.

print(df1)
print(df2)

df1.loc[df1['a'] <= 2]      # Works, as expected.
df1.loc[df2['a'] <= 2]      # This works even though df2 is longer than df1.

df1.loc[[True, True, True]]                 # Fails - too short.
df1.loc[[True, True, True, True]]           # The Goldilocks zone.
df1.loc[[True, True, True, True, True]]     # Fails - too long.

mask = df2['a'] <= 2
print(mask)                 # The length is 7.
df1.loc[mask]               # This works

df1.loc[pd.Series([True, True, True, True, True])]  # Ooh, this works even though it is too long.


#%%

p = np.nan

if p > 0.0:
    print("True")
else:
    print("False")


#%%

n1 = 10000
n2 = 10000
q = ['q1', 'q2', 'q3']
a = [1, 2, 3, 4]
p = [0.1, 0.2, 0.3, 0.4]

df1 = pd.DataFrame()
df1['q'] = np.random.choice(q, n1, replace=True)
df1['a'] = np.random.choice(a, n1, replace=True, p=p)

df2 = pd.DataFrame()
df2['q'] = np.random.choice(q, n2, replace=True)
df2['a'] = np.random.choice(a, n2, replace=True, p=p[::-1])


df1_counts = jm_hist_prep(df1)
df2_counts = jm_hist_prep(df2)

plt.bar(
    df1_counts[df1_counts['q'] == 'q1']['a'],
    df1_counts[df1_counts['q'] == 'q1']['v'],
    alpha=0.5)
plt.bar(
    df2_counts[df2_counts['q'] == 'q1']['a'],
    df2_counts[df2_counts['q'] == 'q1']['v'],
    alpha=0.5)
plt.plot()


#%%

plt.bar([1,2,3], [10,20,30])
plt.bar([1,3,2], [10,20,30])            # This is put in numerical order.
plt.bar(['a','b','c'], [10,20,30])
plt.bar(['a','c','b'], [10,20,30])      # This is displayed as presented.
plt.bar([1,'a','b'], [10,20,30])        # You can't mix int and str.


#df[['a', 'b']].plot.hist(alpha=0.5)


#%%
# TODO: keep_new
# The default (and numbering of zero) makes sense.
# Each column is a variable so it isn't always
# meaningful to add them.

q = pd.DataFrame()
q['age']    = range(4)
q['height'] = q['age'] * 10
q['weight'] = q['age'] * 100
q.index = ['person_' + str(qq) for qq in range(4)]

print(q)
print(np.sum(q))
print(np.sum(q, axis=0))    # The default.
print(np.sum(q, axis=1))


#%%
# TODO: Compare pd.pivot() to pd.pivot_table().

df = pd.DataFrame()
df['patient'] = list("aabbb")
df['code']    = list("qrqrr")
df['score']   = range(5)

print(df)
print("")

pd.pivot_table(df, index='patient', columns='code', values='score')
pd.pivot_table(df, index='patient', columns='code', values='score', aggfunc=sum)
pd.pivot_table(df, index='patient', columns='code', values='score', aggfunc=min)
#pd.pivot_table(df, index='patient', columns='code', values='score', aggfunc=count)


#%%
# Comparison chaining.  This is valid.

if 10 <= 20 <= 30:
    print("True")
else:
    print("False")


#%%

try:
    tmp;
except:
    print("Create tmp")
    tmp = []

tmp.append('abc')
tmp.append('def')

print(tmp)


#%%

p = np.nan

if np.isnan(p):
    print("NaN")
elif p > 0.05:
    print("bigger")
else:
    print("smaller")


#%%

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color="black", width=0.5),
      label = ["ppp", "eee", "rrr",
               "None",
               "No", "dd", "ee", "ss",
               "lll"],
      color = "blue"
    ),
    link = dict(
      source = [0, 1, 2, 3, 4, 5, 6, 7],
      target = [8, 8, 8, 8, 3, 3, 3, 3],
      value = [2, 3, 5,
               100,
               70, 10, 10, 10,
               110])
)])


#%%
# TODO: keep
# Ranks with duplicated rankings.

df = pd.DataFrame()
df['c1'] = list('aaabbb')
df['c2'] = [1, 1, 3, 4, 5, 6]

print(df)

df.groupby('c1')['c2'].rank(ascending=True)
df.groupby('c1')['c2'].rank(ascending=True, method='min')

df['c1'].rank()
df['c2'].rank()
df['c2'].rank(method='min')
df['c2'].rank(method='max')


#%%
#TODO: Keep

yr = 2020
mn = 4

start_date = datetime.date(yr, mn, 1)       # Make this inclusive.
end_date = datetime.date(yr + 1, mn, 1)     # Make this exclusive.

print(start_date)
print(end_date)

print(start_date + datetime.timedelta(days=10))     # This doesn't do months or years.


#%%
# TODO: keep

df = df.rename(columns=lambda x: 'dep_' + str(x))


#%%

n = 6

coeffs = np.array([1.0, 2.0, 3.0])

df = pd.DataFrame()
df['v1'] = np.random.random(n)
df['v2'] = np.random.random(n)
df['v3'] = np.random.random(n)
df['sum'] = df.apply(lambda row: np.sum(row), axis=1)
df['v1'] /= df['sum']
df['v2'] /= df['sum']
df['v3'] /= df['sum']
df.drop('sum', axis=1, inplace=True)


df['target'] = df.apply(lambda row: np.sum(row * coeffs), axis=1)
df['noisy'] = df['target'] + np.random.normal(0.0, 0.25, n)

print(df)


#%%

import re

"abc    def".replace(' ', '.')
"abc    def".replace('/ +/g', 'x')

t = "a\n\naabc\tdef"
print(t)
print(re.sub('[ \t\n]+', ' ', t))

re.sub('\.+', '.', 'abc...def ')


#%%

import re

"abcdef".find("a")
"abcabc".find("a")


# [0, 5, 10, 15]
[m.start() for m in re.finditer('test', 'test test test test')]


s = "aa"
t = "aaaabb"
[m.start() for m in re.finditer(s, t)]
[[m.start(), len(s)] for m in re.finditer(s, t)]

[m.start() for m in re.finditer('xxx', 'test test test test')]

s = 'testx'
print([[m.start(), len(s)] for m in re.finditer(s, 'test test test test')])
q = [[m.start(), len(s)] for m in re.finditer(s, 'test test test test')]


#%%

n, w, p = word_counter("landlord sofa sofa")
print("Results:")
print(n)
print(w)
print(p)


#%%

def foo():
    return 10, 20

tmp1, tmp2 = foo()
print(tmp1, tmp2)

tmp1, tmp1 = foo()
print(tmp1)

del tmp1, tmp2


#%% Column 'c' is added to df after appending q.

df = pd.DataFrame()
df['a'] = [1, 2, 3]
df['b'] = [10, 20, 30]
print(df)

q = {'a': 4, 'b': 40, 'c':400}
print(q)

df = df.append(q, ignore_index=True)
print(df)


#%%

import win32com.client

fn = 'word_document_test.docx'

word = win32com.client.Dispatch("Word.Application")
word.visible = False
wb = word.Documents.Open(fn)
doc = word.ActiveDocument
print(doc.Range().Text)

# How do you close the file?


#%% Read the text from a MS Word document.
# https://towardsdatascience.com/how-to-extract-data-from-ms-word-documents-using-python-ed3fbb48c122

#specific to extracting information from word documents
import os
import zipfile
#other tools useful in extracting the information from our document
import re
#to pretty print our xml:
import xml.dom.minidom


fn = 'word_document.docx'

# The name with the Word Document body
# text in it is ‘word/document.xml’
document = zipfile.ZipFile(fn)
document.namelist()
document.read('word/document.xml')

uglyXml = xml.dom.minidom.parseString(document.read('word/document.xml'))
uglyXml = uglyXml.toprettyxml(indent='  ')
text_re = re.compile('>\n\s+([^<>\s].*?)\n\s+</', re.DOTALL)
prettyXml = text_re.sub('>\g<1></', uglyXml)

print(prettyXml)


#%% bins using np.digitize()

bins = [0] + [10**n for n in range(3)]
#print(bins)

df = pd.DataFrame()
df['y'] = np.random.choice([2016, 2017], 100)
df['m'] = 100 * np.random.random_sample(100)**3
df['n'] = df['m'].astype('int32')
df['b'] = np.digitize(df['n'], bins)

binned = df.groupby(['y', 'b']).count().reset_index()
binned.rename(columns={'n': 'volume'}, inplace=True)

binned['bin'] = binned['b'].apply(lambda n: bins[n])


#%% Or using pandas cut()

dfbinned = pd.cut(df['n'], bins, labels=bins[:-1])
zz = pd.concat([df, dfbinned], axis=1)


#%%
# TODO

exec(open(r'C:\a\functions\all.py').read())


#%%

i = "original"

def print_i1():
    print(i)        # Prints "changed" when called below.

def print_i2(s=i):  # Default set at function creation, not at function call.
    print(s)        # Prints "original" when called below.


i = "changed"
print_i1()
print_i2()


#%%

with pd.ExcelWriter(os.path.join(params['outputs'], 'workbook.xlsx')) as writer:
    df1.to_excel(writer, sheet_name="sheet_df1", merge_cells=True)
    df2.to_excel(writer, sheet_name="sheet_df2", merge_cells=False)



#%%
# End Of File
#
