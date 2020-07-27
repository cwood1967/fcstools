
# %%
import glob

import numpy as np
from matplotlib import pyplot as plt

from rawutils import FcsTrajectory

import autoreload
%load_ext autoreload
%autoreload 2
# %%
files = glob.glob("/Users/cjw/Desktop/FCS_raw/*.raw")
fcs = rawutils.rawfcs(files[0])
fcs.readrawfile()
_ = fcs.bin(2e-5)

plt.plot(fcs.tx, fcs.traj)

# %%
len(fcs.tx // 1000) , len(fcs.tx) % 1000, 

# %%
dv = fcs.tx[:30000]
dv = dv.reshape((1000, -1))
dv = dv.min(axis=1)

dtj = fcs.traj[:30000]
dtj = dtj.reshape((1000, -1))
dtj = dtj.sum(axis=1)

# %%
plt.plot(dv, dtj)

# %%

ac = fcs.autocorrelate(1)

# %%
plt.semilogx(fcs.rawactime, fcs.rawautocorr, alpha=0.1)
plt.semilogx(fcs.actime, fcs.autocorr)



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
