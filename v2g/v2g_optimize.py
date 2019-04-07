#!/usr/bin/env python3
import matplotlib as mlib
mlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar
from v2g.iutils import *
import fire, tqdm, math, os, scipy.optimize


def v2g_optimize(state_list=None, sell_prices=[], prefix='.', samples=2813, SF=0.3, degredation=True, efficiency=0.78, battery_cap_cost = 209, charging_rate = 11.5):
	#total EVs in NYC Sept2018 default samples
	if type(state_list) == str:
		state_list = [state_list]
	if type(sell_prices) == str or type(sell_prices) == float:
		sell_prices = [float(sell_prices)]
	print('Running with \n\t state_list = {}\n\t sell_prices = {}\n\t prefix = {}\n\t samples = {}\n\t SF = {}\n\t Degredation = {}\n\t efficiency = {}\n\t battery_cap_cost = {}\n\t charging_rate = {}'.format(state_list, sell_prices, prefix, samples, SF, degredation, efficiency, battery_cap_cost, charging_rate))

	#Source: https://www.nyserda.ny.gov/All-Programs/Programs/ChargeNY/Support-Electric/Map-of-EV-Registrations

	cars = pd.read_csv(os.path.join(prefix,'Cars.csv'), delimiter= ',')
	rated_dist_dict, charge_time_dict, range_dict = {}, {}, {}
	for i in range(len(cars.Battery)):
		rated_dist_dict[cars.Battery[i]] = cars.dist[i]
		charge_time_dict[cars.Battery[i]] = cars.Charge_time[i]
		range_dict[cars.Battery[i]] = cars.Range[i]
	battery = np.random.choice(cars.Battery, size = samples, p = cars.prob)
	battery_commute = np.copy(battery)

	ev_range = np.array([range_dict[i] for i in battery])
	dist_one_kWh = np.array([rated_dist_dict[i] for i in battery])
	complete_charging_time = np.array([charge_time_dict[i] for i in battery])

	eff = efficiency #roundtrip efficiency of Tesla S - 0.78
	#reference: Shirazi, Sachs 2017

#	charging_rate = 11.5
	# Considering AC Level2 Charging using SAE J1772 (240V/80A), standard on board chargers are
	# capable of delivering 48A with the high amperage version being able to deliver 72A - tesla.com
	# Super charger gives 17.2 kWh/hr charging rate

	#battery degradation cost
	# Nikolas Soulopoulos, Cost Projections - Battery, vehicles and TCO, BNEF, June 2018 Report
	# bat_degradation = battery * battery_cap_cost/energy_throughput 	#$/kWh
	bat_degradation = battery_cap_cost * battery * int(degredation)
	np.set_printoptions(precision = 2, threshold = np.inf)

	#Extracting commute distance and time data from NHTS
	nhts_filepath = os.path.join(prefix, 'PERV2PUB.CSV')
	#commute distance
	dist = np.genfromtxt(nhts_filepath, delimiter = ',', dtype = None, usecols = (103), skip_header = 1, filling_values = 0)
	#commute time
	time = np.genfromtxt(nhts_filepath, delimiter = ',', dtype = None, skip_header=1, usecols = (84), filling_values = 0)
	#state in column 49 of PERV2PUB.csv
	state = np.genfromtxt(nhts_filepath, delimiter = ',', dtype = None, usecols = (48), skip_header= 1, filling_values = 0)
	if state_list is None:
		state_list = ['Arizona', 'California', 'DC', 'Illinois', 'Massachusetts', 'New York']

	mask = {'Arizona': 'AZ',
			'California': 'CA',
			'DC': 'DC',
			'Illinois': 'IL',
			'Massachusetts': 'MA',
			'New York': 'NY',
			'Texas': 'TX'}
	print(dt.datetime.today().date())
	result_path = os.path.join(prefix, 'Realistic_model', 'Results', str(dt.datetime.today().date()))
	if not os.path.exists(result_path):
		# print('making a new dir!')
		os.makedirs(result_path)

	max_savings = {}
	pricetaker_savings = {}
	osp = {}

	for s in state_list:

		print(s)
		mask_index = np.asarray([mask[s]]*len(state), dtype = 'S2')
		dist_sample = []
		time_sample = []
		# masking the data before sampling the commute distance and time.
		for i in range(len(dist)):
			if(state[i] == mask_index[i]):
				if(dist[i] > 0 and dist[i] < 100 and time[i]> 0 and time[i] < 100):
					dist_sample.append(dist[i])
					time_sample.append(time[i])
		#samples commute distance and time for state
		commute_dist, commute_time = dist_time_battery_correlated_sampling(dist = dist_sample, time = time_sample, ev_range = ev_range, N = samples, SF = SF)

		#Actual calculations of battery usage and selling
		battery_used_for_travel = 2 * commute_dist/dist_one_kWh  #kWh

		# Sampling departure times
		pop_file, lbmp_file = state_pop(s)
		Time_depart_for_work = np.genfromtxt(os.path.join(prefix, pop_file), delimiter = ',', filling_values = 0, skip_header = 1, usecols = (91), dtype = None)
		depart_values, depart_keys = create_dict(Time_depart_for_work, bins = np.arange(1,151))
		# departure times in the datasets are encoded into numbers so use input.csv to get actual datetime values.
		depart_time_decode = pd.read_csv(os.path.join(prefix, 'input.csv'))
		depart_time_start = depart_time_decode.Time_start
		#calculate probabilities to pass into np.random.choice
		p_depart = depart_values/np.sum(depart_values)
		sample_time_depart = np.random.choice(depart_keys, size = samples, p = p_depart )

		time_leave = pd.to_datetime(depart_time_start).dt.time
		time_leave_home = []
		for i in sample_time_depart:
			time_leave_home.append(time_leave[i-1])

		time_leave_home = np.asarray(time_leave_home)
		time_depart_from_home = []
		for i in range(len(time_leave_home)):
			time_depart_from_home.append(dt.datetime(2017,1,1,time_leave_home[i].hour, time_leave_home[i].minute, 0))

		time_arrival_work = add_time(np.asarray(time_depart_from_home), commute_time)

		#sampling work daily work hrs for different people
		work_hrs_per_week = np.genfromtxt(os.path.join(prefix, pop_file), delimiter = ',', filling_values = 0, skip_header = 1, usecols = (71), dtype = None)
		work_hrs_values, work_hrs_keys = create_dict(work_hrs_per_week, bins = np.arange(1,100))
		p_work_hrs = work_hrs_values/np.sum(work_hrs_values)
		weekly_work_hrs = np.random.choice(work_hrs_keys, size = samples, p = p_work_hrs)
		daily_work_mins = weekly_work_hrs*60/5

		#LBMP Data
		data = pd.read_csv(os.path.join(prefix, lbmp_file))
		dates = convert_npdatetime64_topy(pd.to_datetime(data.Time_Stamp).values)
		price = np.asarray(data.LBMP)/1000		#LBMP in kWh

		data_file = os.path.join(result_path, '{}_eff{}_cap{}_rch{}.csv'.format(s, eff, battery_cap_cost, charging_rate))

		pricetaker_savings = []
		max_savings = []
		osp = []
		capacity_fade = []
		commute_cycles = []
		v2g_cycles, charging_cost, commute_cost, discharging_cost = [], [], [], []
		pt_cycles, pt_charging_cost, pt_discharging_cost, pt_cap_fade = [], [], [], []

		for i in tqdm.tqdm(range(samples)):

			pt_scenario = profit(x=-1000, battery = battery[i], battery_used_for_travel = battery_used_for_travel[i], commute_distance = commute_dist[i], commute_time = commute_time[i], complete_charging_time = complete_charging_time[i], time_arrival_work = time_arrival_work[i], daily_work_mins = daily_work_mins[i], dates = dates, price = price, bat_degradation = bat_degradation[i], charging_rate=charging_rate, eff = eff, SF=SF)
			pricetaker_savings.append(-pt_scenario[0])
			pt_cycles.append(pt_scenario[9])
			pt_charging_cost.append(pt_scenario[1])
			pt_discharging_cost.append(pt_scenario[2])
			pt_cap_fade.append(pt_scenario[6])

			if len(sell_prices) == 0:
				result = scipy.optimize.minimize_scalar(profit_wrapper, args=(battery[i], battery_used_for_travel[i], commute_dist[i], commute_time[i], complete_charging_time[i], time_arrival_work[i], daily_work_mins[i], dates, price,  bat_degradation[i], i, charging_rate, eff, SF), bounds=(0, 1.0), method='Golden', tol=1e-3, options=dict(maxiter = 50))
				o = max(0,result.x)
			elif len(sell_prices) == 1:
				o = sell_prices[0]
			else:
				assert len(sell_prices) == samples
				o = sell_prices[i]
			osp.append(o)

			result = profit(x= o, battery = battery[i], battery_used_for_travel = battery_used_for_travel[i], commute_distance = commute_dist[i], commute_time = commute_time[i], complete_charging_time = complete_charging_time[i], time_arrival_work = time_arrival_work[i], daily_work_mins = daily_work_mins[i], dates = dates, price = price, bat_degradation = bat_degradation[i], charging_rate=charging_rate, seed=i, SF=SF)
			max_savings.append(-result[0])
			capacity_fade.append(result[6])
			commute_cycles.append(result[8])
			v2g_cycles.append(result[9])
			commute_cost.append(result[4])
			charging_cost.append(result[1])
			discharging_cost.append(result[2])

		results = {'Distance': commute_dist, 'Time':commute_time, 'Work hours': weekly_work_hrs, 'Work time': time_depart_from_home, 'Battery': battery, 'OSP_Savings': max_savings, 'OSP': osp, 'Pricetaker_Savings': pricetaker_savings, 'capacity_fade': capacity_fade, 'Commute_cycles': commute_cycles, 'v2g_cycles': v2g_cycles, 'pt_cycles': pt_cycles, 'osp_charging': charging_cost, 'pt_charging': pt_charging_cost, 'osp_discharging': discharging_cost, 'commute_cost': commute_cost, 'pt_capacity_fade': pt_cap_fade }
		results = pd.DataFrame.from_dict(results)
		results.to_csv(data_file)


def main():
	fire.Fire(v2g_optimize)