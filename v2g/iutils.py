import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar
import scipy.stats as ss
import os
import math
import random
from .battery import Battery

def state_pop(state):
    pop_files = {'Arizona': 'ss16paz.csv',
                'California': 'ss16pca.csv',
                'DC': 'ss16pdc.csv',
                'Illinois': 'ss16pil.csv',
                'Massachusetts': 'ss16pma.csv',
                'NewYork': 'ss16pny.csv',
                'Texas': 'ss16ptx.csv'}
    LBMP_file = {'Arizona': 'phoenix_2019.csv',
                'California': 'sfca_2019.csv',
                'DC': 'washingtondc_2019.csv',
                'Illinois': 'chicago_2019.csv',
                'Massachusetts': 'boston_2019.csv',
                'NewYork': 'nyc_2019.csv',
                'Texas': 'houston.csv'}
    return os.path.join('Population_data', pop_files[state]), os.path.join('LBMP', LBMP_file[state])

def convert_npdatetime64_topy(a):
    return a.astype('M8[ms]').astype('O')

def add_time(array = None, minutes = [0]):
	''' This function adds the given time in minutes to the array elementwise
		Args: array of datetimes : The array to which time has to be added
		minutes : integer or array of integer for minutes to be added
		return: array: Array with time added
	'''
	if isinstance(array, (list, np.ndarray)):
		if not isinstance(minutes, (list, np.ndarray)):
			minutes = np.repeat(minutes, len(array))
		return np.asarray(array) + np.array([dt.timedelta(minutes = int(x)) for x in minutes])
	else:
		return array + dt.timedelta(minutes=int(minutes))

def round_dt_up(time, minutes=5, hours=1):
	'''Round up (ceil) to the nearest given time increment
	(the hours argument doesn't do what it is supposed to yet)
	Taken from: https://stackoverflow.com/a/10854034/9329318
	Args:
		time (datetime object): time to round up
		minutes (integer): nearest minute to round up to
		hours (integer): nearest hour to round up to
	Returns (datetime object): rounded time
	'''
	#rounds up to nearest y
	rounder = lambda x,y: (y - x % y) * (x % y != 0)
	return time + dt.timedelta(minutes=rounder(time.minute, minutes),
							   hours=rounder(time.hour, hours))


def round_dt_down(time, minutes=5, hours=1):
	'''Round down (floor) to the nearest given time increment
	(the hours argument doesn't do what it is supposed to yet)
	Taken from: https://stackoverflow.com/a/10854034/9329318
	Args:
		time (datetime object): time to round down
		minutes (integer): nearest minute to round down to
		hours (integer): nearest hour to round down to
	Returns (datetime object): rounded time
	'''
	return time - dt.timedelta(minutes=time.minute % minutes,
                             hours=time.hour % hours)

def find_price_for_timestamp(time, dates, price):
	''' Get nearest timestamp and return price corresponding to that timestamp'''
	dates = np.asarray(dates)
	idx = (np.abs(dates - time)).argmin()
	return dates[idx], price[idx]

def cost_calc(state, dates, price,
			  battery, time_start, charging_rate,
			  time_stop = None, daily_work_mins = None, set_SP = 0,
			  battery_left = 0, timedelta = 5):
	'''This function will calculate the cost of electricity for based on the state -discharging or charging
		Args: state = 'charging' or 'discharging'
			dates = The time stamps for the data used (otype- array)
			price = Cost of electricity at the given time stamps from data (otype - array)
			battery = The maximum battery capacity for users, length must be equal to sample size (otype - float)
			time_start = start of the time interval for which cost of electricity needs to be calculated (otype - datetime.datetime object)
			N = Number of cycles battery has experienced
			time_stop = end of interval which started, default = None (otype - datetime.datetime object)
			daily_work_mins = An array of working hours of users (otype - int)
			set_SP = The selling price set  by the user, default = 0
			battery_left = The battery left for user, default = None
			timedelta = The time interval considered for LBMP(hourly or five-minutes), in minutes, default = 60
			DoD = The depth of discharge for the battery, default = 0.90
		return: total_cost,
				If state = 'discharging'
					battery_sold = battery sold for V2G
				if state = 'charging'
					battery_charged = battery charged
					percent_deg  = percentage degradation
	'''
	# time_start = round_dt_down(time_start, minutes = 60)
	if(state == 'charging'):
		if time_stop is None:
			raise ValueError()
		total_cost = 0
		time = time_start
		stop = time_stop
		while battery.soc < 1.0 and time < stop:
			_, rate = find_price_for_timestamp(time, dates, price)
			# charge for either timedelta or until we stop
			# price paid depends on the efficiency
			total_cost += battery.charge(charging_rate, min(timedelta / 60, (stop - time).total_seconds() / 3600)) * rate / battery.eff
			time += dt.timedelta(minutes = timedelta)
		return total_cost

	elif(state == 'discharging'):
		if daily_work_mins is None:
			raise ValueError()
		time = time_start
		stop = round_dt_down(time + dt.timedelta(minutes = daily_work_mins), minutes = timedelta)
		money_earned = 0
		while battery.capacity * battery.soc > battery_left and time < stop:
			_, rate = find_price_for_timestamp(time, dates, price)
			if rate >= set_SP:
				money_earned += battery.discharge(charging_rate, min(timedelta / 60, (stop - time).total_seconds() / 3600)) * rate * battery.eff
			time += dt.timedelta(minutes = timedelta)
		return money_earned

	else:
		raise ValueError('Unknown state ' + state)

def create_dict(file_data, bins = 'auto'):
	''' This function will create a dictionary of values and occurrences by histograming given array
		Args: file_data : array of data
			bins = array of bin edges to be used for histogramming
		returns: values, keys: Arrays of values and keys from the dictionary created
	'''
	index = file_data > 0
	n, bins, patches = plt.hist(file_data[index], bins = bins)
	file_dict = {a:b for a,b in zip(bins, n)}
	values = np.asarray(list(file_dict.values()))
	keys = np.asarray(list(file_dict.keys()))
	return values, keys

def dist_time_battery_correlated_sampling(dist, time, ev_range, N, DoD = 0.9, SF = 0.3):
	'''Sample from data for a given sample size N
		Args:
			dist, time: The arrays to sample from (otype - array)
			ev_range: array of length N, This is used to check that range of battery size is capable of handling commute distance sampled.
			N: desired sample size
			DoD: depth of discharge
			SF: salvation factor for battery
		Returns:
			sampled_data: New array of randomly sampled elements. If correlated is True, then this returns a tuple of arrays.'''

	if len(ev_range) != N:
		raise ValueError('Size of ev_range must equal N')
	length = len(dist)
	p = np.asarray([1.0/length]*length)
	x = 1000.0	#start with a high commute distance
	sampled_dist = np.zeros(N)
	sampled_time = np.zeros(N)
	for i in range(N):
		while(2 * x/DoD >= ev_range[i]):
			ind = np.random.choice(np.arange(length), p = p)
			x = dist[ind]
		sampled_dist[i] = x
		sampled_time[i] = time[ind]
		x = 1000
	return sampled_dist, sampled_time

def profit(x, battery_size, battery_used_for_travel, commute_distance, commute_time, complete_charging_time, time_arrival_work, daily_work_mins, dates, price, bat_degradation,
vacation_days, charging_rate = 11.5, eff=0.837, SF = 0.3, DoD = 0.9):
	# time_arrival_work = round_dt_up(time_arrival_work)
	final_discharge_cost = 0
	final_charge_cost = 0
	final_commute_cost = 0
	k = 0
	day = dt.datetime(2019,1,1,0,0)
	count = 1
	#convert battery variable into battery object
	battery = Battery(battery_size, eff)
	commute_battery = Battery(battery_size, eff)

	#Using holiday calender
	holidays = USFederalHolidayCalendar().holidays(start = '2019-01-01', end = '2020-01-01').to_pydatetime()

	# 52*2 weekdays, 11 holidays, on average 14 vacation days
	# leaves 236 working days
	while(count <= 365): # calendar days
		#check if it is a holiday
		if day in holidays or count in vacation_days:
			# print('{} is a holiday'.format(day.date()))
			k+=288
			count+=1
			day+= dt.timedelta(days=1)
			time_arrival_work += dt.timedelta(days = 1)
			battery.age(24)
			commute_battery.age(24)
			continue

		#determining if it is a weekday
		if(day.weekday()<5):
			#Total money earned for discharging the battery
			date_set = dates[k:k+288*3]
			price_set = price[k:k+288*3]

			#go to work
			#driving discharge
			battery.discharge(battery_used_for_travel/2.0 / commute_time / 60.0, commute_time / 60.0, eff=1.0)
			commute_battery.discharge(battery_used_for_travel/2.0 / commute_time / 60.0, commute_time / 60.0, eff=1.0)

			time_discharge_starts = round_dt_up(time_arrival_work)
			#Start with discharging, assuming battery has been used to commute one way
			cost_discharging = cost_calc(state = 'discharging', dates = date_set, price = price_set, battery = battery, time_start = time_discharge_starts, time_stop = None, daily_work_mins = daily_work_mins, set_SP = x, battery_left = battery_used_for_travel/2, timedelta = 5, charging_rate=charging_rate)
			final_discharge_cost += cost_discharging
			#Fast forward time to when charging should start
			time_leaving_work = add_time(time_arrival_work, daily_work_mins)
			time_reach_home = add_time(time_leaving_work, commute_time)

			#driving discharge
			battery.discharge(battery_used_for_travel/2 / commute_time / 60, commute_time / 60, eff=1.0)
			commute_battery.discharge(battery_used_for_travel/2 / commute_time / 60, commute_time / 60, eff=1.0)

			time_arrival_work += dt.timedelta(days = 1)
			k += 288
			day += dt.timedelta(days=1)
			count+=1

			#Charge the battery up until time to leave for work
			time_charging_starts = round_dt_up(time_reach_home)
			cost_charging= cost_calc(state = 'charging', dates = date_set, price = price_set, battery = battery, time_start = time_charging_starts, time_stop = time_arrival_work - dt.timedelta(minutes=commute_time), timedelta=5, charging_rate=charging_rate)
			final_charge_cost += cost_charging

			#Cost of commute without V2G
			cost_commute = cost_calc(state = 'charging', dates = date_set, price = price_set, battery = commute_battery, time_start = time_charging_starts, time_stop = time_arrival_work - dt.timedelta(minutes=commute_time), timedelta=5, charging_rate=charging_rate)
			final_commute_cost += cost_commute

		else:
			k += 288
			count += 1
			day+= dt.timedelta(days=1)
			time_arrival_work += dt.timedelta(days = 1)
			#age battery
			battery.age(24)
			commute_battery.age(24)

	final_cdgdn = (1 - battery.capacity_fade) * bat_degradation / SF
	final_commute_cdgdn = (1 - commute_battery.capacity_fade) * bat_degradation / SF
	annual_savings = final_discharge_cost - (final_charge_cost + final_cdgdn) - (0.0 - (final_commute_cost + final_commute_cdgdn))

	return {'Annual_savings': -annual_savings, 'V2G_charge_cost':final_charge_cost, 'V2G_discharge_cost': final_discharge_cost, 'V2G_deg_cost': final_cdgdn, 'Commute_cost': final_commute_cost, 'Commute_deg_cost': final_commute_cdgdn, 'V2G_capacity_fade': 1.0 - battery.capacity_fade,  'Commute_capacity_fade': 1.0 - commute_battery.capacity_fade, 'Commute_cycles': commute_battery.cycles, 'V2G_cycles': battery.cycles}

def profit_wrapper(x, *args):
	output = profit(x, *args)
	return output['Annual_savings']
