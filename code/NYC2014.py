import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mlib
import datetime as dt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pandas.tseries.holiday import USFederalHolidayCalendar
import scipy.stats as ss
import os
from utils import *

prefix = 'C:/Users/hetag/Documents/White lab/Vehicle_to_grid/'
# Required numbers
Sample = 2559    #total EVs in NYC 2015
Actual_battery = 75	#kWh 60Kwh discontinued
battery = 60 #kWh
eff = 0.95 #roundtrip efficiency of Tesla S 
#reference: Shirazi, Sachs 2017
rated_dist = 240 #miles www.tesla.com
dist_one_kWh = rated_dist*eff/battery  	#miles/kWh
DoD = 0.90   #depth of discharge
charging_rate = 11.5		
# Considering AC Level2 Charging using SAE J1772 (240V/80A), standard on board chargers are capable of delivering 48A with the high amperage version being able to deliver 72A - tesla.com
# Super charger gives 17.2 kWh/hr charging rate
complete_charging_time = 420 	#minutes(7 hrs) for the Level2 charger to charge Tesla S60

#battery degradation cost
lifecycles = 5300		#for lithium ion battery from Gerad's article
energy_throughput = 2*lifecycles*battery*DoD 		#Ln.Es.DoD (kWh)
battery_cap_cost = 450			# $/kWh Nykvist and nilsson 2015, for 2014
# Alternately from https://electrek.co/2017/02/18/tesla-battery-cost-gigafactory-model-3/
bat_degradation = battery/energy_throughput			# using Gerad's equation, cb is the cost of current storage which will be introduced later.
lcos = 1142/1000			# $/kWh, Lazards LCOS V2.0 2016, Commercial lithium ion category average

working_days = 250
#pricetaker_cycles = working_days
#V2G_cycles = 25

#For optimal scenario we need a set selling price
SP = np.arange(0, 0.31, 0.01)		#$/kWh
x = len(SP)


#Extracting and histograming commute distance data for NYS from NHTS 
# Sampling using the probability vectors for distance and time 
dist = np.genfromtxt(prefix + 'PERV2PUB.csv', delimiter = ',', dtype = None, usecols = (103), skip_header = 1, filling_values = 0)	#commute distance
time = np.genfromtxt(prefix + 'PERV2PUB.csv', delimiter = ',', dtype = None, skip_header=1, usecols = (84), filling_values = 0)    #commute time
#work_time = np.genfromtxt('PERV2PUB.csv', delimiter = ',', dtype = None, usecols = (98), skip_header = 1, filling_values = 0)
state = np.genfromtxt(prefix + 'PERV2PUB.csv', delimiter = ',', dtype = None, usecols = (48), skip_header= 1, filling_values = 0)		#state in column 49 of PERV2PUB.csv
mask = np.asarray(['NY']*len(state), dtype = 'S2')


dist_sample = []
time_sample = []
for i in range(len(dist)):
	if(state[i] == mask[i]):
		if(dist[i] > 0 and dist[i] < 100 and time[i]> 0 ):
			dist_sample.append(dist[i])
			time_sample.append(time[i])
			

p_dist = np.asarray([1.0/len(dist_sample)]*len(dist_sample))
commute_dist, commute_time = sampling(data = dist_sample, N = Sample, data2=time_sample, correlated=True)

print('Commute distance = ',commute_dist)
print('Commute time = ', commute_time)

#Actual calculations of battery usage and selling
battery_used_for_travel = 2 * commute_dist/dist_one_kWh  #kWh
battery_left_to_sell = battery*DoD - battery_used_for_travel

result_path = prefix + 'Results/{}/'.format(dt.datetime.today().date()) + 'nyc/'
if not os.path.exists(result_path):
    # print('making a new dir!')
    os.makedirs(result_path)

# Sampling departure times
Time_depart_for_work = np.genfromtxt(prefix + 'Population_data/ss15pny.csv', delimiter = ',', filling_values = 0, skip_header = 1, usecols = (95), dtype = None)
depart_values, depart_keys = create_dict(Time_depart_for_work, bins = np.arange(1,151))
depart_time_decode = pd.read_csv(prefix + 'input.csv')
depart_time_start = depart_time_decode.Time_start
p_depart = depart_values/np.sum(depart_values)
sample_time_depart = np.random.choice(depart_keys, size = Sample, p = p_depart ) 

time_leave = pd.to_datetime(depart_time_decode.Time_start).dt.time
time_leave_home = []
for i in sample_time_depart:
	time_leave_home.append(time_leave[i-1])

time_leave_home = np.asarray(time_leave_home)
time_depart_from_home = []
for i in range(len(time_leave_home)):
	time_depart_from_home.append(dt.datetime(2014,1,1,time_leave_home[i].hour, time_leave_home[i].minute, 0))

time_arrival_work = addtime(np.asarray(time_depart_from_home), commute_time)

#sampling work daily work hrs for different people
work_hrs_per_week = np.genfromtxt(prefix+ 'Population_data/ss15pny.csv', delimiter = ',', filling_values = 0, skip_header = 1, usecols = (73), dtype = None)
work_hrs_values, work_hrs_keys = create_dict(work_hrs_per_week, bins = np.arange(1,100))
p_work_hrs = work_hrs_values/np.sum(work_hrs_values)
weekly_work_hrs = np.random.choice(work_hrs_keys, size = Sample, p = p_work_hrs)
daily_work_mins = weekly_work_hrs*60/5 
plt.clf()

#LBMP Data
data = pd.read_csv(prefix + 'LBMP/2014LBMP.csv', usecols = [0,1,2,3,4])
dates = pd.to_datetime(data.Time_Stamp)
price = np.asarray(data.LBMP)/1000		#LBMP in kWh

time_arrival_work = roundupTime(time_arrival_work)
final_discharge_cost = np.zeros((x,Sample))
final_charge_cost = np.zeros((x,Sample))
final_cdgdn = np.zeros((x, Sample))
final_commute_cdgdn = np.zeros(Sample) 
final_commute_cost = np.zeros(Sample)
battery_cycles = 0
k = 0
day = dt.datetime(2014,1,1,0,0)
count = 1
battery_left = np.array([battery] * Sample)
#Using holiday calender
holidays = USFederalHolidayCalendar().holidays(start = '2014-01-01', end = '2014-12-31').to_pydatetime()

while(count <= working_days):
	#check if it is a holiday
	if day in holidays:
		print('{} is a holiday'.format(day.date()))
		k+=24*12
		day+= dt.timedelta(days=1)
		for i in range(Sample):
			time_arrival_work[i] += dt.timedelta(days = 1)
		pass
			
	#determining if it is a weekday	
	if(day.weekday()<5):
		#Total money earned for discharging the battery
		#print(k)
		#print(day.date())
		date_set = np.asarray([i for i in dates[k:k+24*36]])
		price_set = np.asarray([i for i in price[k:k+24*36]])
		battery_used = np.zeros((x,Sample))
		time_discharge_starts = time_arrival_work

		for i in range(x):
			
			cost_discharging, battery_sold = cost_calc(state = 'discharging', dates = date_set, price = price_set, battery = battery, time_start = time_discharge_starts, time_stop = None, daily_work_mins = daily_work_mins, set_SP = SP[i], battery_left = battery_left, timedelta = 5)
			final_discharge_cost[i] += cost_discharging
			battery_used[i] = battery_sold + battery_used_for_travel
			time_leaving_work = addtime(time_arrival_work, daily_work_mins)
			time_reach_home = addtime(time_leaving_work, commute_time)
			time_charging_starts = roundupTime(time_reach_home)
			time_charging_stops = addtime(time_charging_starts, complete_charging_time*battery_used[i]/battery)
			time_charging_stops = roundupTime(time_charging_stops)
			cost_charging, battery_charged = cost_calc(state = 'charging', dates = date_set, price = price_set, battery = battery, time_start = time_charging_starts, time_stop = time_charging_stops, battery_left=battery - battery_sold, timedelta = 5) 
			final_cdgdn[i] += battery_charged[i] * bat_degradation * lcos * eff
			final_charge_cost[i] += cost_charging + battery_charged[i] * bat_degradation * lcos * eff

		#Cost of commute without V2G
		charge_commute_stop = addtime(time_charging_starts, complete_charging_time*battery_used_for_travel/battery)
		charge_commute_stop = rounddownTime(np.asarray(charge_commute_stop))
		cost_commute, battery_charged = cost_calc(state = 'charging', dates = date_set, price = price_set, battery = battery, time_start = time_charging_starts, time_stop = charge_commute_stop, battery_left = battery - battery_used_for_travel, timedelta = 5)
		final_commute_cdgdn += battery_used_for_travel * bat_degradation * lcos * eff
		final_commute_cost += cost_commute  + battery_used_for_travel * bat_degradation * lcos * eff

		for i in range(Sample):
			time_arrival_work[i] += dt.timedelta(days = 1)
		k += 24*12
		day+=dt.timedelta(days=1)
		battery_cycles += battery_used/battery
		count+=1
		
	else:
		k+=24*12
		day+= dt.timedelta(days=1)
		for i in range(Sample):
			time_arrival_work[i] += dt.timedelta(days = 1)


#print(day.date(), 'Battery cycles =',battery_cycles)
colors = ['#ffff00','#40E0D0','#ff0000', '#191970']
alpha = [1, 0.7, 0.5, 0.3]

sns.set(context = 'talk', style = 'darkgrid', palette='hls')
mlib.rcParams['figure.figsize'] = (10, 8)
cdgdn_commute = np.mean(final_commute_cdgdn)
cch_commute = np.mean(final_commute_cost - final_commute_cdgdn)
Annual_savings = np.zeros((x,Sample))


for i in range(x):
	Annual_savings[i] = final_discharge_cost[i] - final_charge_cost[i] - (0.0 - final_commute_cost)

np.set_printoptions(precision = 2, threshold = np.inf)

#Writing values to a file
output = result_path + 'data.csv'
results = {'Distance': commute_dist, 'Time':commute_time, 'Work hours': weekly_work_hrs, 'Work time': time_depart_from_home}
results.update({'Savings{}'.format(a):b for a,b in zip(SP, Annual_savings)})
results.update({'Cycle{}'.format(a):b for a,b in zip(SP, battery_cycles)})
results = pd.DataFrame.from_dict(results)
results.to_csv(output)
