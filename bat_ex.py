import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import collections

plt.rcParams['figure.figsize'] = (10.0, 8.0)
sns.set(style= "darkgrid", context="talk", palette='hls')
result_path = 'C:/Users/hetag/Desktop/Vehicle_to_grid/Results/2018-03-13/bat_ex/'

sample = 2559.
year = np.arange(2010, 2051, 1)
x_predict  = np.zeros(len(year))
historical_battery_prices = np.array([1000, 800, 642, 599, 540, 350, 273, 162, 73, 50]) #$/kWh from bloomberg 2017 report
ys = np.arange(2010, 2018, 1)
ys = np.append(ys, 2030)
ys = np.append(ys, 2050)

x_predict[:len(historical_battery_prices)] = historical_battery_prices[:]
x_predict[20] = 73
x_predict[-1] = 50

future_costs = np.interp(year, ys, historical_battery_prices)
future_costs[6:] = pd.stats.moments.rolling_mean(future_costs[6:], window = 5, min_periods=1)
# print(future_costs)

cost_dict = {a:b for a,b in zip(year, future_costs)}
# print(cost_dict)

plt.plot(year, future_costs, 'o-')
plt.title('Forecast of battery capital cost over time')
plt.xlabel('Year')
# plt.xticks(rotation=30)
plt.ylabel('Battery Capital Cost ($/kWh)')
plt.savefig(result_path + 'battery_cost.png')
plt.close()

battery = 60
eff = 0.62
DoD = 0.9
lifecycles = 5300		#for lithium ion battery from Gerad's article
energy_throughput = 2*lifecycles*battery*DoD 		#Ln.Es.DoD (kWh)
states = ['Arizona', 'California', 'DC', 'Illinois', 'Massachusetts', 'New York']
best_sp = {'Arizona': 0.06, 'California': 0.06, 'DC': 0.12, 'Illinois': 0.06, 'Massachusetts': 0.18, 'New York':0.14}

cities = {'Arizona': 'Phoenix', 'California': 'SanFrancisco', 'DC': 'Washington', 'Illinois': 'Chicago', 'Massachusetts': 'Boston', 'New York':'NYC'}
cities_reversed = {a:b for b,a in zip(cities.keys(), cities.values())}
final_results = {}
final_no_outliers = {}
final_outliers = {}

def percentile_based_outlier(data, threshold=95):
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)

for s in states:
    data = pd.read_csv(result_path + s + '/data.csv')
    charging_cost = data['Charging{}'.format(best_sp[s])]
    charging_cycles = data['Cycles{}'.format(best_sp[s])]
    discharging_cost = data['Disharging{}'.format(best_sp[s])]
    commute_cost = data['Commute_cost']
    commute_cycles = data['Commute_cycles']

    final_results[s] = {}
    final_no_outliers[s] = {}
    final_outliers[s] = {}

    for key, value in cost_dict.items():
        bat_degradation = battery * value/energy_throughput
        #print(bat_degradation*eff*charging_cycles)
        annual_savings = discharging_cost - (charging_cost + charging_cycles * battery * bat_degradation * eff) - (0 - (commute_cost + commute_cycles * battery * bat_degradation * eff))
        final_results[s][key] = annual_savings
        # outliers = percentile_based_outlier(annual_savings, threshold = 90)
        # final_outliers[s][key] = annual_savings[outliers]
        # no_outliers = outliers == False
        # final_no_outliers[s][key] = annual_savings


for key in sorted(cities_reversed):
    value = cities_reversed[key]
    x = list(final_results[value].keys())
    y = [np.mean(a) for a in final_results[value].values()]
    yerr = np.sqrt(np.array([np.var(a, ddof = 1) for a in final_results[value].values()]) / sample)
    # plt.plot(x, y, '.', label = '{}'.format(key))
    plt.errorbar(x, y, yerr = yerr, fmt = '.', markeredgewidth=2, label = '{}'.format(key))


plt.xticks(rotation = 30)
plt.ylabel('Savings from V2G($)')
plt.xlabel('Year') 
plt.legend()
plt.tight_layout()
plt.savefig(result_path + 'future_cost.png')#.format(s))
plt.close()