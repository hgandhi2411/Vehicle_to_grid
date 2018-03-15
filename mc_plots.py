import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Sample = 2559
plt.rcParams['figure.figsize'] = (10.0, 8.0)
sns.set(style= "darkgrid", context="talk", palette='hls')
result_path = 'C:/Users/hetag/Desktop/Vehicle_to_grid/Results/2018-03-13/multi_city/'
states = ['Arizona', 'California', 'DC', 'Illinois', 'Massachusetts', 'New York']
best_sp = {'Arizona': 0.06, 'California': 0.06, 'DC': 0.12, 'Illinois': 0.06, 'Massachusetts': 0.18, 'New York':0.14}

final_results = {}

for s in states:
    data = pd.read_csv(result_path + s + '/data.csv')
    Annual_savings = data['Savings{}'.format(best_sp[s])]
    # commute_cost = data['Commute_cost']
    # commute_cycles = data['Commute_cycles']

    final_results[s] = Annual_savings

def mad_based_outlier(points, thresh=99):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)

final_results = pd.DataFrame.from_dict(final_results)

o, k = {}, {}
for s in states:
    outliers = percentile_based_outlier(final_results[s], threshold=96.25)
    o['{}\n SP = {}'.format(s, best_sp[s])] = final_results[s][outliers]
    not_outliers = outliers == False
    k['{}\n SP = {}'.format(s, best_sp[s])] = final_results[s][not_outliers]
    
sns.violinplot(data = pd.DataFrame.from_dict(k), bw = 0.5)
plt.yticks(np.arange(-200, 150, 25))
plt.ylabel('Savings from V2G($)')
sns.stripplot(data = pd.DataFrame.from_dict(o), edgecolor='gray', size = 5, jitter = True, linewidth=1)
plt.savefig(result_path + 'violin.svg')

# for s in states:
#     outliers = mad_based_outlier(final_results[s])
#     o[s] = final_results[s][outliers]
#     not_outliers = outliers == False
#     k[s] = final_results[s][not_outliers]

# sns.violinplot(data = pd.DataFrame.from_dict(k))
# sns.stripplot(data = pd.DataFrame.from_dict(o), color='k')
# plt.savefig(result_path + 'violin_mad.svg')