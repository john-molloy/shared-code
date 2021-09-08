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





#
# End Of File
#
