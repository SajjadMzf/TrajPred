import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd

import read_csv as rc
import param as p
ttc_list = []
thw_list = []
min_ttc_list = []
min_thw_list = []
for file_itr, track_path in enumerate(p.track_paths):
    track_df = pd.read_csv(track_path)
    static_df = pd.read_csv(p.static_paths[file_itr])
    thw_list.append(track_df['thw'].values)
    min_thw_list.append(static_df['minTHW'].values)
    ttc_list.append(track_df['ttc'].values)
    min_ttc_list.append(static_df['minTTC'].values)
    
ttc_array = np.concatenate(ttc_list)
min_ttc_array = np.concatenate(min_ttc_list)
thw_array = np.concatenate(thw_list)
min_thw_array = np.concatenate(min_thw_list)

ttc_array = ttc_array[ttc_array!=0]
ttc_array = ttc_array[ttc_array<10]
thw_array = thw_array[thw_array!=0]
thw_array = thw_array[thw_array<10]

min_ttc_array = min_ttc_array[min_ttc_array != -1]
min_thw_array = min_thw_array[min_thw_array != -1]
min_ttc_array = min_ttc_array[min_ttc_array <10]
min_thw_array = min_thw_array[min_thw_array <10]


print(max(ttc_array))
print(max(ttc_array))
print(min(min_ttc_array))
print(min(min_ttc_array))
print(len(ttc_array))
print(len(min_ttc_array))
#print(np.sum(ttc_array>200))
#print(np.sum(min_ttc_array>200))

fig1, ax1 = plt.subplots()
ax1.hist(ttc_array,200,(0,10), density=True, cumulative = True)
ax1.set_title('TTC Histogram')
fig2, ax2 = plt.subplots()
ax2.hist(min_ttc_array,200,(0,10), density=True, cumulative = True)
ax2.set_title('Min TTC Histogram')

fig3, ax3 = plt.subplots()
ax3.hist(thw_array,200,(0,10), density=True, cumulative = True)
ax3.set_title('THW Histogram')
fig4, ax4 = plt.subplots()
ax4.hist(min_thw_array,200,(0,10), density=True, cumulative = True)
ax4.set_title('Min THW Histogram')
plt.show()























'''

# create 2 kernels
m1 = (50,50)
s1 = np.eye(2)*400
k1 = multivariate_normal(mean=m1, cov=s1)

m2 = (150,150)
s2 = np.eye(2)*400
k2 = multivariate_normal(mean=m2, cov=s2)

# create a grid of (x,y) coordinates at which to evaluate the kernels


x = np.arange(0,200)
y = np.arange(0,200)
xx, yy = np.meshgrid(x,y)

# evaluate kernels at grid points
xxyy = np.c_[xx.ravel(), yy.ravel()]

zz = np.zeros((xxyy.shape[0])).reshape(200,200)

x1lim = (0,100)
y1lim = (0,100)
x1res = x1lim[1] - x1lim[0]
y1res = x1res
x1 = np.linspace(x1lim[0], x1lim[1])
y1 = np.linspace(y1lim[0], y1lim[1])
xx1, yy1 = np.meshgrid(x1,y1)

# evaluate kernels at grid points
xxyy1 = np.c_[xx1.ravel(), yy1.ravel()]

zz1 = k1.pdf(xxyy1) + k2.pdf(xxyy1)
zz1 = zz1.reshape(50,50)
zz[100:150, 50:100] += zz1
plt.imshow(zz1); plt.show()

'''