#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import matplotlib as mlib
import datetime as dt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pandas.tseries.holiday import USFederalHolidayCalendar
from utils import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)
sns.set(style= "darkgrid", context="talk", palette='hls')

prefix = ''

# prefix = os.path.join(os.path.expanduser('~'),'Documents', 'Heta', 'White lab', 'Vehicle_to_grid')

Sample = 2813    #total EVs in NYC Sept2018
#Source: https://www.nyserda.ny.gov/All-Programs/Programs/ChargeNY/Support-Electric/Map-of-EV-Registrations

cars = pd.read_csv(os.path.join(prefix,'Cars.csv'), delimiter= ',')
rated_dist_dict, charge_time_dict = {}, {}
for i in range(len(cars.Battery)):
	rated_dist_dict[cars.Battery[i]] = cars.dist[i]
	charge_time_dict[cars.Battery[i]] = cars.Charge_time[i]
battery = np.random.choice(cars.Battery, size = Sample, p = cars.prob)
battery_commute = np.copy(battery)

dist_one_kWh = np.array([rated_dist_dict[i] for i in battery])

complete_charging_time = np.array([charge_time_dict[i] for i in battery])

eff = 0.62 #roundtrip efficiency of Tesla S 
#reference: Shirazi, Sachs 2017
DoD = 0.90   #depth of discharge

charging_rate = 11.5		
# Considering AC Level2 Charging using SAE J1772 (240V/80A), standard on board chargers are 
# capable of delivering 48A with the high amperage version being able to deliver 72A - tesla.com
# Super charger gives 17.2 kWh/hr charging rate

#battery degradation cost
lifecycles = 5300		#for lithium ion battery from Gerad's article
energy_throughput = 2*lifecycles*battery*DoD 		#Ln.Es.DoD (kWh)

battery_cap_cost = 209	
# Nikolas Soulopoulos, Cost Projections - Battery, vehicles and TCO, BNEF, June 2018 Report

bat_degradation = battery * battery_cap_cost/energy_throughput			
# using Gerad's equation, cb is the cost of current storage which will be introduced later.

working_days = 250

#For optimal scenario we need a set selling price
SP = np.arange(0,0.3,0.01)		#$/kWh
x = len(SP)

np.set_printoptions(precision = 2, threshold = np.inf)

def state_pop(state):
	pop_files = {'Arizona': 'ss16paz.csv', 
				'California': 'ss16pca.csv', 
				'DC': 'ss16pdc.csv', 
				'Illinois': 'ss16pil.csv', 
				'Massachusetts': 'ss16pma.csv', 
				'New York': 'ss16pny.csv', 
				'Texas': 'ss16ptx.csv'}
	LBMP_file = {'Arizona': 'phoenix.csv', 
				'California': 'sfca.csv', 
				'DC': 'washingtondc.csv', 
				'Illinois': 'chicago.csv', 
				'Massachusetts': 'boston.csv', 
				'New York': 'nyc.csv', 
				'Texas': 'houston.csv'}
	return os.path.join('Population_data', pop_files[state]), os.path.join('LBMP', LBMP_file[state])

#Extracting and histograming commute distance data for NYS from NHTS 
# Sampling using the probability vectors for distance and time 
nhts_filepath = os.path.join(prefix, 'PERV2PUB.CSV')
#commute distance
dist = np.genfromtxt(nhts_filepath, delimiter = ',', dtype = None, usecols = (103), skip_header = 1, filling_values = 0)	
#commute time
time = np.genfromtxt(nhts_filepath, delimiter = ',', dtype = None, skip_header=1, usecols = (84), filling_values = 0)
#work_time = np.genfromtxt('PERV2PUB.csv', delimiter = ',', dtype = None, usecols = (98), skip_header = 1, filling_values = 0)
#state in column 49 of PERV2PUB.csv
state = np.genfromtxt(nhts_filepath, delimiter = ',', dtype = None, usecols = (48), skip_header= 1, filling_values = 0)

state_list = ['Arizona', 'California', 'DC', 'Illinois', 'Massachusetts', 'New York']

mask = {'Arizona': 'AZ', 
		'California': 'CA', 
		'DC': 'DC', 
		'Illinois': 'IL', 
		'Massachusetts': 'MA', 
		'New York': 'NY', 
		'Texas': 'TX'}
print(dt.datetime.today().date())
result_path = os.path.join(prefix, 'Results', str(dt.datetime.today().date()), 'var_bat')
if not os.path.exists(result_path):
    # print('making a new dir!')
    os.makedirs(result_path)

max_savings = {}
pricetaker_savings = {}

for s in state_list:
	
	print(s)
	mask_index = np.asarray([mask[s]]*len(state), dtype = 'S2')
	dist_sample = []
	time_sample = []
	for i in range(len(dist)):
		if(state[i] == mask_index[i]):
			if(dist[i] > 0 and dist[i] < 100 and time[i]> 0 ):
				dist_sample.append(dist[i])
				time_sample.append(time[i])
	
	commute_dist, commute_time = sampling(data = dist_sample, N = Sample, data2=time_sample, correlated=True)
	
	#Actual calculations of battery usage and selling
	battery_used_for_travel = 2 * commute_dist/dist_one_kWh  #kWh
	battery_left_to_sell = battery*DoD - battery_used_for_travel

	# Sampling departure times
	pop_file, lbmp_file = state_pop(s)
	Time_depart_for_work = np.genfromtxt(os.path.join(prefix, pop_file), delimiter = ',', filling_values = 0, skip_header = 1, usecols = (91), dtype = None)
	depart_values, depart_keys = create_dict(Time_depart_for_work, bins = np.arange(1,151))
	depart_time_decode = pd.read_csv(os.path.join(prefix, 'input.csv'))
	depart_time_start = depart_time_decode.Time_start
	p_depart = depart_values/np.sum(depart_values)
	sample_time_depart = np.random.choice(depart_keys, size = Sample, p = p_depart ) 
	
	time_leave = pd.to_datetime(depart_time_start).dt.time
	time_leave_home = []
	for i in sample_time_depart:
		time_leave_home.append(time_leave[i-1])

	time_leave_home = np.asarray(time_leave_home)
	time_depart_from_home = []
	for i in range(len(time_leave_home)):
		time_depart_from_home.append(dt.datetime(2017,1,1,time_leave_home[i].hour, time_leave_home[i].minute, 0))

	time_arrival_work = addtime(np.asarray(time_depart_from_home), commute_time)

	#sampling work daily work hrs for different people
	work_hrs_per_week = np.genfromtxt(os.path.join(prefix, pop_file), delimiter = ',', filling_values = 0, skip_header = 1, usecols = (71), dtype = None)
	work_hrs_values, work_hrs_keys = create_dict(work_hrs_per_week, bins = np.arange(1,100))
	p_work_hrs = work_hrs_values/np.sum(work_hrs_values)
	weekly_work_hrs = np.random.choice(work_hrs_keys, size = Sample, p = p_work_hrs)
	daily_work_mins = weekly_work_hrs*60/5 
	plt.clf()

	#LBMP Data
	data = pd.read_csv(os.path.join(prefix, lbmp_file))
	dates = pd.to_datetime(data.Time_Stamp)
	price = np.asarray(data.LBMP)/1000		#LBMP in kWh

	time_arrival_work = roundupTime(time_arrival_work)
	final_discharge_cost = np.zeros((x,Sample))
	final_charge_cost = np.zeros((x,Sample))
	final_cdgdn = np.zeros((x, Sample))
	final_commute_cdgdn = np.zeros(Sample) 
	final_commute_cost = np.zeros(Sample)
	battery_cycles = np.zeros((x, Sample))
	q_deg = np.zeros((x, Sample))
	commute_cycles = np.zeros(Sample)
	k = 0
	day = dt.datetime(2017,1,1,0,0)
	count = 1

	#Using holiday calender
	holidays = USFederalHolidayCalendar().holidays(start = '2017-01-01', end = '2018-01-01').to_pydatetime()
	battery_charged = np.array([battery for i in range(x)])

	while(count <= working_days):
		#check if it is a holiday
		if day in holidays:
			print('{} is a holiday'.format(day.date()))
			k+=24
			day+= dt.timedelta(days=1)
			for i in range(Sample):
				time_arrival_work[i] += dt.timedelta(days = 1)
				
		#determining if it is a weekday	
		if(day.weekday()<5):
			#Total money earned for discharging the battery
			#print(k)
			#print(day.date())
			date_set = np.asarray([i for i in dates[k:k+24*3]])
			price_set = np.asarray([i for i in price[k:k+24*3]])
			battery_used = np.zeros((x,Sample))
			time_discharge_starts = roundupTime(time_arrival_work)

			for i in range(x):
				
				#Start with discharging, assuming battery has been used to commute one way
				cost_discharging, battery_sold = cost_calc(state = 'discharging', dates = date_set, price = price_set, battery = battery * (1 - q_deg[i]), time_start = time_discharge_starts,  N = battery_cycles[i], time_stop = None, daily_work_mins = daily_work_mins, set_SP = SP[i], battery_left = battery_charged[i] - battery_used_for_travel/2, timedelta = 60)
				final_discharge_cost[i] += cost_discharging
				battery_used[i] = battery_sold + battery_used_for_travel

				#Fast forward time to when charging should start
				time_leaving_work = addtime(time_arrival_work, daily_work_mins)
				time_reach_home = addtime(time_leaving_work, commute_time)
				time_charging_starts = roundupTime(time_reach_home)
				time_charging_stops = addtime(time_charging_starts, complete_charging_time*battery_used[i]/battery)
				time_charging_stops = roundupTime(time_charging_stops)

				#Charge the battery
				cost_charging, battery_charged[i], q_loss = cost_calc(state = 'charging', dates = date_set, price = price_set, battery = battery * (1 - q_deg[i]), time_start = time_charging_starts, N = battery_cycles[i], time_stop = time_charging_stops, battery_left = battery - battery_used[i], timedelta=60) 
				cost_charging += battery_used[i] * bat_degradation * eff
				final_cdgdn[i] += battery_used[i] * bat_degradation * eff
				final_charge_cost[i] += cost_charging
				q_deg[i] += q_loss/battery

			#Cost of commute without V2G
			charge_commute_stop = addtime(time_charging_starts, complete_charging_time*battery_used_for_travel/battery_commute)
			charge_commute_stop = rounddownTime(np.asarray(charge_commute_stop))
			cost_commute, battery_charged_commute, q_loss_commute = cost_calc(state = 'charging', dates= date_set, price = price_set, battery = battery_commute,  N = commute_cycles, time_start = time_charging_starts, time_stop = charge_commute_stop, battery_left = battery_commute - battery_used_for_travel) 
			commute_cycles += battery_charged_commute/battery_commute
			cost_commute += battery_used_for_travel * bat_degradation * eff
			final_commute_cdgdn += battery_used_for_travel * bat_degradation * eff
			final_commute_cost += cost_commute
			battery_commute = battery_commute * (1 - q_loss_commute)

			for i in range(Sample):
				time_arrival_work[i] += dt.timedelta(days = 1)
			k += 24
			day += dt.timedelta(days=1)
			battery_cycles += battery_used/battery
			count+=1
			
		else:
			k+=24
			day+= dt.timedelta(days=1)
			for i in range(Sample):
				time_arrival_work[i] += dt.timedelta(days = 1)


	#print(day.date(), 'Battery cycles =',battery_cycles)
	colors = ['#ffff00','#40E0D0','#ff0000', '#191970']
	alpha = [1, 0.7, 0.5, 0.3]

	Annual_savings = np.zeros((x,Sample))

	for i in range(x):
		Annual_savings[i] = final_discharge_cost[i] - final_charge_cost[i] - (0.0 - final_commute_cost)

	if not os.path.exists(os.path.join(result_path, s)):
		os.makedirs(os.path.join(result_path, s))
	output = os.path.join(result_path, s, 'savings.csv')
	results = {'Distance': commute_dist, 'Time':commute_time, 'Work hours': weekly_work_hrs, 'Work time': time_depart_from_home, 'Battery': battery, 'Commute_cost': final_commute_cost, 'Commute final Q': battery_commute}
	results.update({'Savings{}'.format(a):b for a,b in zip(SP, Annual_savings)})
	results.update({'Cycle{}'.format(a):b for a,b in zip(SP, battery_cycles)})
	results.update({'Q_loss{}'.format(a):b for a,b in zip(SP, q_deg)})
	results = pd.DataFrame.from_dict(results)
	results.to_csv(output)
