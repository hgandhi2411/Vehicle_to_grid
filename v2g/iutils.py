import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar
import scipy.stats as ss
import os
import math

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


def add_time(array = None, minutes = [0]):
	''' This function adds the given time in minutes to the array elementwise
		Args: array of datetimes : The array to which time has to be added
		minutes : integer or array of integer for minutes to be added
		return: array: Array with time added
	'''
	if isinstance(array, (list, np.ndarray)):
		if not isinstance(minutes, (list, np.ndarray)):
			minutes = np.repeat(minutes, len(array))
		return np.asarray(array) + np.apply_along_axis(lambda x: dt.timedelta(minutes=int(x)), -1, minutes)
	else:
		return array + dt.timedelta(minutes=int(minutes))

def round_dt_up(time, minutes=5, hours=1):
	'''Round up (ceil) to the nearest given time increment
	Args:
		time (datetime object): time to round up
		minutes (integer): nearest minute to round up to
		hours (integer): nearest hour to round up to
	Returns (datetime object): rounded time
	'''
	#rounds up to nearest y
	rounder = lambda x,y: (y - x % y) * (x % y != 0)
	return time + dt.timedelta(minutes=rounder(time.minute,minutes),
							   hours=rounder(time.hour, hours))


def round_dt_down(time, minutes=5, hours=1):
	'''Round down (floor) to the nearest given time increment
	Args:
		time (datetime object): time to round down
		minutes (integer): nearest minute to round down to
		hours (integer): nearest hour to round down to
	Returns (datetime object): rounded time
	'''
	return time - dt.timedelta(minutes=time.minute % minutes,
                             hours=time.hour % hours)


def cost_calc(state, dates, price,
			  battery, time_start, N,
			  time_stop = None, daily_work_mins = None, set_SP = 0,
			  battery_left = None, timedelta = 60, charging_rate = 11.5,
			  eff = 0.78, DoD = 0.90):
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
	time_start = round_dt_down(time_start, minutes = 60)

	if(state == 'charging'):
		if time_stop is None:
			raise ValueError()
		total_cost = 0
		deg = 0
		if battery_left is None:
			battery_charged = battery*(1-DoD)
		else:
			battery_charged = battery_left
		time = time_start
		stop = time_stop
		for i in range(len(dates)):
			n = -1
			if(dates[i] == time):
				if((battery_charged <= battery) and (time < stop)):
					if((stop - time).total_seconds() / 60 < 60):
						total_cost += charging_rate * eff * price[i] * ((stop - time).seconds / 60) / 60
						battery_charged += charging_rate * ((stop - time).seconds / 60) / 60
						if(battery_charged > battery):
							battery_charged = battery
						soc = max(0.2, battery_charged/battery)
						if(n != -1):
							deg += (1 - real_battery_degradation(dt = 60 - n*60, SOC = soc, N = N))
						else:
							deg += (1 - real_battery_degradation(dt = 60, SOC = soc, N = N))
						break
					battery_charged += charging_rate	#hourly
					if (battery_charged > battery):
						battery_charged = battery
					total_cost += charging_rate*eff*price[i]	#hourly
					soc = max(0.2, battery_charged/battery)
					if(n != -1):
						deg += (1 - real_battery_degradation(dt = 60 - n*60, SOC = soc, N = N))
					else:
						deg += (1 - real_battery_degradation(dt = 60, SOC = soc, N = N))
					# -n*60 to factor for the fact that time - start is cumulative
					time += dt.timedelta(hours = 1)
					n += 1
					# print(cost)
		# print(percent_deg)
		return total_cost, battery_charged, deg

	elif(state == 'discharging'):
		if daily_work_mins is None:
			raise ValueError()
		total_cost = 0
		battery_sold = 0
		if battery_left is None:
			battery_left = battery
		if(battery_left == 0):
			total_cost = 0
			battery_sold = 0
		time = time_start
		stop = round_dt_down(time + dt.timedelta(minutes = daily_work_mins), minutes = timedelta)
		money_earned = 0
		for i in range(len(dates)):
			if(dates[i] == time):
				if((price[i] >= set_SP) and (battery_sold <= battery_left) and (time < stop)):
					if((stop - time).total_seconds() / 60 < 60):
						money_earned += charging_rate * eff * price[i] * ((stop - time).seconds / 60 ) / 60
						battery_sold += charging_rate * ((stop - time).seconds / 60 ) / 60
						break
					battery_sold += charging_rate
					money_earned += charging_rate*eff*price[i]
					time += dt.timedelta(minutes = timedelta)
					battery_left -= charging_rate
					if(battery_left < battery*(1 - DoD)):
						break
		total_cost = money_earned
		return total_cost, battery_sold
	else:
		raise ValueError('Unknown state ' + state)

def create_dict(file_data, bins = 'auto'):
	''' This function will create a dictionary of values and occurences by histograming given array
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

def dist_time_battery_correlated_sampling(dist, time, ev_range, N):
	'''Sample from data for a given sample size N
		Args:
			dist, time: The arrays to sample from (otype - array)
			ev_range: array of length N, This is used to check that range of battery size is capable of handling commute distance sampled.
			N: desired sample size
			state_mask: A string or number that should be used to filter data.
		Returns:
			sampled_data: New array of randomly sampled elements. If correlated is True, then this returns a tuple of arrays.'''

	if len(ev_range) != N:
		raise ValueError('Size of ev_range must equal N')
	p = np.asarray([1/len(dist)]*len(dist))
	length = len(dist)
	x = 1000
	sampled_dist = np.zeros(N)
	sampled_time = np.zeros(N)
	for i in range(N):
		while(2 * x >= ev_range[i]):
			ind = np.random.choice(np.arange(length), p = p)
			x = dist[ind]
		sampled_dist[i] = x
		sampled_time[i] = time[ind]
		x = 1000
	return sampled_dist, sampled_time

def calc_open_circuit_voltage(SOC):
    # Voc calc reference: Energies 2016, 9, 900
	# Parameters for LMNCO cathodes at 45 degreeC - couldn't find for NCA batteries
	if SOC == 0:
		raise ValueError()
	a = 3.535
	b = -0.0571
	c = -0.2847
	d = 0.9475
	m = 1.4
	n = 2
	Voc = a + b * (-math.log(SOC))**m + c * SOC + d * math.e**(n * (SOC - 1))
	return Voc

def real_battery_degradation(dt, N = 0., SOC = 1., i_rate = 11.5, T = 318, Dod_max = 0.9):
	''' This function calculates the percentage of battery degradation happening at every time step for NCA li-ion chemistry
	Args:
		dt : time in minutes (time_step of operation)
		N : cycle number, default = 0, no cycling
		DoD : depth of discharge, default = 0.9
		i_rate : charging/discharging rate of the battery, default = 11.5 kWh = 11500/360 = 31.78 Ah
		T : battery temperature in Kelvin, default = 318K
	Returns: float, total percentage loss in the battery capacity at given timestep timedelta
	'''
	dt = dt/24/60	#convert minutes to days
	V_nom = 360		#V Nominal voltage of EV batteries
	#reference quantities
	T_ref = 298.15 	# K
	V_ref = 3.6		#V
	Dod_ref = 1.0
	F = 96485 		# C/mol	Faraday constant
	Rug = 8.314	# J/K/mol

	# fitted parameters for NCA chemistries
	b0 = 1.04
	c0= 1
	b1_ref = 1.794 * 10**(-4)	# 1/day^(0.5)
	c2_ref = 1.074 * 10**(-4)	# 1/day
	Eab1 = 35000				# J/mol
	alpha_b1 = 0.051
	beta_b1 = 0.99
	Eac2 = 35000				# J/mol
	alpha_c2 =  0.023
	beta_c2 = 2.61
	t_life = 1					#day
	# Degradation rates are calculated in terms of representative duty cycle period - typically 1 day or 1 week
	dt_cyc = 1					#day

	Voc = calc_open_circuit_voltage(SOC)

	# integral over delta(tcyc) ignored because we are considering one time step
	b1 = (b1_ref / dt_cyc) * math.e**(-Eab1/Rug * (1/T - 1/T_ref)) \
		* math.e**(alpha_b1 * F / Rug *(Voc/T - V_ref/T_ref)) \
		* (((1 + Dod_max)/Dod_ref)**beta_b1) * dt

	# b1 = max(0, b1)
	Qli = b0 - b1*t_life**(0.5)

	c2 = (c2_ref / dt_cyc) *  math.e**(-Eac2/Rug * (1/T - 1/T_ref)) \
		* math.e**(alpha_c2 * F / Rug *(Voc/T - V_ref/T_ref)) \
		* (N* (Dod_max/Dod_ref)**beta_c2) * dt

	Qsites = c0 - c2 * t_life

	Q_loss = min(Qli, Qsites)		# Q_loss is relative capacity after degradation

	# Do not allow for negative degradation
	return min(max(Q_loss, 0), 1)



def profit(x, battery, battery_used_for_travel, commute_distance, commute_time, complete_charging_time, time_arrival_work, daily_work_mins, dates, price, bat_degradation, charging_rate = 11.5, eff=0.78):
	time_arrival_work = round_dt_up(time_arrival_work)
	final_discharge_cost = 0
	final_charge_cost = 0
	final_cdgdn = 0
	final_commute_cdgdn = 0
	final_commute_cost = 0
	battery_cycles = 0
	q_deg = 0
	q_loss_commute = 0
	q_deg_commute = 0
	commute_cycles = 0
	k = 0
	day = dt.datetime(2017,1,1,0,0)
	count = 1

	#Using holiday calender
	holidays = USFederalHolidayCalendar().holidays(start = '2017-01-01', end = '2018-01-01').to_pydatetime()
	battery_commute = battery
	battery_charged = battery

	while(count <= 250): #working_days
		#check if it is a holiday
		if day in holidays:
			# print('{} is a holiday'.format(day.date()))
			k+=24
			day+= dt.timedelta(days=1)
			time_arrival_work += dt.timedelta(days = 1)

		#determining if it is a weekday
		if(day.weekday()<5):
			#Total money earned for discharging the battery
			#print(k)
			#print(day.date())
			date_set = np.asarray([i for i in dates[k:k+24*3]])
			price_set = np.asarray([i for i in price[k:k+24*3]])
			battery_used = 0
			time_discharge_starts = round_dt_up(time_arrival_work)

			#Start with discharging, assuming battery has been used to commute one way
			cost_discharging, battery_sold = cost_calc(state = 'discharging', dates = date_set, price = price_set, battery = battery, time_start = time_discharge_starts,  N = battery_cycles, time_stop = None, daily_work_mins = daily_work_mins, set_SP = x, battery_left = battery_charged - battery_used_for_travel/2, timedelta = 60, charging_rate=charging_rate, eff = eff)
			final_discharge_cost += cost_discharging
			battery_used = battery_sold + battery_used_for_travel

			#Fast forward time to when charging should start
			time_leaving_work = add_time(time_arrival_work, daily_work_mins)
			time_reach_home = add_time(time_leaving_work, commute_time)
			time_charging_starts = round_dt_up(time_reach_home)
			time_charging_stops = add_time(time_charging_starts, complete_charging_time*battery_used/battery)
			time_charging_stops = round_dt_up(time_charging_stops)

			#Charge the battery
			cost_charging, battery_charged, q_loss = cost_calc(state = 'charging', dates = date_set, price = price_set, battery = battery, time_start = time_charging_starts, N = battery_cycles, time_stop = time_charging_stops, battery_left = battery - battery_used, timedelta=60, charging_rate=charging_rate, eff = eff)
			#q_loss is the relative capacity
			q_deg += q_loss
			# cost_charging += q_loss/100 *battery * bat_degradation * eff
			final_cdgdn += battery * q_loss * bat_degradation
			final_charge_cost += cost_charging
			battery_cycles += battery_used/battery

			#Cost of commute without V2G
			charge_commute_stop = add_time(time_charging_starts, complete_charging_time*battery_used_for_travel/battery_commute)
			charge_commute_stop = round_dt_up(charge_commute_stop)
			cost_commute, battery_charged_commute, q_loss_commute = cost_calc(state = 'charging', dates= date_set, price = price_set, battery = battery_commute,  N = commute_cycles, time_start = time_charging_starts, time_stop = charge_commute_stop, battery_left = battery_commute - battery_used_for_travel, charging_rate=charging_rate, eff = eff)
			commute_cycles += battery_used_for_travel/battery_commute
			q_deg_commute += q_loss_commute
			# cost_commute += battery * (1 - q_loss_commute) * bat_degradation * eff
			final_commute_cdgdn += battery_commute *  q_loss_commute * bat_degradation
			final_commute_cost += cost_commute

			time_arrival_work += dt.timedelta(days = 1)
			k += 24
			day += dt.timedelta(days=1)
			count+=1

		else:
			k+=24
			day+= dt.timedelta(days=1)
			time_arrival_work += dt.timedelta(days = 1)

	annual_savings = final_discharge_cost - final_charge_cost - final_cdgdn - (0.0 - final_commute_cost - final_commute_cdgdn)
	return -annual_savings, final_charge_cost, final_discharge_cost, final_cdgdn, final_commute_cost, final_commute_cdgdn, q_deg, q_deg_commute

def profit_wrapper(x, battery, battery_used_for_travel, commute_distance, commute_time, complete_charging_time, time_arrival_work, daily_work_mins, dates, price, bat_degradation, charging_rate, eff):
	output = profit(x, battery, battery_used_for_travel, commute_distance, commute_time, complete_charging_time, time_arrival_work, daily_work_mins, dates, price, bat_degradation, charging_rate, eff)
	return output[0]

#TODO: Check why degradation values are bizarre, exceeding battery capacity