
# %%
import autoreload
%load_ext autoreload
%autoreload 2

import glob

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from fcsutils import FcsTrajectory
from fcsfit import *

# %%
files = glob.glob("/Users/cjw/Desktop/FCS_raw/*.raw")

df_list = list()
for f in files:
    print(f)
    fcs = FcsTrajectory(f, 2e-5)
    fcs()
    _df = pd.DataFrame({'tau':fcs.actime,
           'autocorr':fcs.autocorr,
           'file':f,
           })
    df_list.append(_df)

df = pd.concat(df_list, axis=0)

# _ = fcs.autocorrelate(1)


# %%
plt.figure(figsize=(8,6))
sns.lineplot(x='tau', y='autocorr', data=df,
             err_style="bars", hue='file', alpha=.1,
             legend=False, palette=sns.dark_palette("blue", n_colors=10))
sns.lineplot(x='tau', y='autocorr', data=df)
plt.xscale('log')
# %%

# %%

plt.plot(fcs.txdisplay, fcs.trajdisplay)
fcs.txdisplay[0:10]

# %%

fcs1 = fcs_1comp(p0 = [.1, .005, 0])
p, perr = fcs1(df.tau, df.autocorr)
p

# %%
sns.lineplot(x='tau', y='autocorr', data=df)
plt.plot(df.tau.unique(), fcs1.predict(df.tau.unique(), p))
plt.xscale('log')
# %%
fcs2 = fcs_2comp(p0=[.1, .5, .001, .0005, 0], bounds=None)
p2, perr2 = fcs2(df.tau, df.autocorr)

# %%
sns.lineplot(x='tau', y='autocorr', data=df)
plt.plot(df.tau.unique(), fcs2.predict(df.tau.unique(), p2))
plt.xscale('log')
# %%

# bounds = [(.01, .0001, 0.01, 25e-6, -.001),
#           (.2, .01, 0.5, 1e-3, 0.001)]
fcst = fcs_1comp_triplet(p0=[.1, .001, 0], bounds=None)
fcst.__xfunc__ = fcst.__func__
fcst.__func__ = lambda t, g0, td, offset:fcst.__xfunc__(
    t, g0, td, 0.05, 100e-6, offset)

fcst.__func__(.5, .1, .001, 0)
# %%
pt, perrt = fcst(df.tau, df.autocorr)

sns.lineplot(x='tau', y='autocorr', data=df)
plt.plot(df.tau.unique(), fcst.predict(df.tau.unique(), pt))
plt.xscale('log')
pt
# %%
import pandas as pd
jtest = pd.read_csv("/Users/cjw/Desktop/FCS_raw/Plot Values.csv")


# %%

plt.semilogx(fcs.actime, fcs.autocorr - 1)
plt.semilogx(jtest.X1[1:], jtest.Y1[1:])
plt.savefig("/Users/cjw/Desktop/fcscorr.png")
# %%
jtest.shape, fcs.actime.shape

# %%
fcs.actime.shape

# %%
files

# %%
