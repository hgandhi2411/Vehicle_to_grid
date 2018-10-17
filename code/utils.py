import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar
import scipy.stats as ss
import os
import math

def addtime(array = None, minutes = [0]):
	''' This function adds the given time in minutes to the array elementwise
		Args: array : The array to which time has to be added 
			minutes : integer or array of integer for minutes to be added
		return: array: Array with time added
	'''
	new_array = []
	for i in range(len(array)):
		new_array.append(array[i] + dt.timedelta(minutes = int(minutes[i])))
	return np.asarray(new_array)


#www.stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object-python
def roundupTime(dtarray=None, roundTo=5*60):
	'''Round a datetime object to any time laps in seconds
	Args: dtarray : datetime.datetime object.
		roundTo : Closest number of seconds to round to, default 5 minutes.
	return: dtarray : Rounded time to next 5 minutes for each element of the array
	'''
	dtarray = np.array(dtarray)
	for i in range(len(dtarray)):
		seconds = (dtarray[i] - dtarray[i].min).seconds
		rounding = (seconds+roundTo) // roundTo * roundTo
		dtarray[i] = dtarray[i] + dt.timedelta(0, rounding-seconds, -dtarray[i].microsecond)
	return dtarray

def rounddownTime(dtarray=None, roundTo=5*60):
	'''Round a datetime object to any time laps in seconds
	Args: dtarray : datetime.datetime object.
		roundTo : Closest number of seconds to round to, default 5 minutes.
	return: dtarray : Rounded time to 5 minutes before for each element of the array
	'''
	dtarray = np.array(dtarray)
	for i in range(len(dtarray)):
		seconds = (dtarray[i] - dtarray[i].min).seconds
		rounding = (seconds) // roundTo * roundTo
		dtarray[i] = dtarray[i] + dt.timedelta(0, rounding-seconds, -dtarray[i].microsecond)
	return dtarray

# def makeDir(prefix, target=None):
# 	if not os.path.exists(os.path.join(prefix, target)):
# 		os.makedirs(os.path.join(prefix, target))
# 	return 


def cost_calc(state, dates, price, battery, time_start, N, time_stop = None, daily_work_mins = None, set_SP = 0, battery_left = None, timedelta = 60):
	'''This function will calculate the cost of electricity for based on the state -discharging or charging
		Args: state = 'charging' or 'discharging'
			dates = The time stamps for the data used (otype- array)
			price = Cost of electricity at the given time stamps from data (otype - array)
			battery = The maximum battery capacity for users, length must be equal to sample size (otype - array)
			time_start = start of the time interval for which cost of electricity needs to be calculated (otype - array)
			time_stop = end of interval which started, default = None (otype - array)
			daily_work_mins = An array of working hours of users
			set_SP = The selling price set  by the user, default = 0
			battery_left = The battery left for every user, default = None
			timedelta = The time interval considered for LBMP(hourly or five-minutes), in minutes, default = 60
		return: total_cost = An array of costs calculated, 
				If state = 'discharging'
					battery_sold = battery sold for V2G
				if state = 'charging'
					battery_charged = battery charged
					percent_deg  = percentage degradation
	'''
	Sample = 2559
	eff = 0.62 #roundtrip efficiency of Tesla S 
	#reference: Shirazi, Sachs 2017
	DoD = 0.90   #depth of discharge
	charging_rate = 11.5
	a = len(time_start)

	if type(battery) == int:
		battery = np.array([battery] * a)
	
	if(state == 'charging'):
		total_cost = np.zeros(a)
		percent_deg = np.zeros(a)
		if battery_left is None:
			battery_charged = np.zeros(a)
		else:
			battery_charged = battery_left
		for j in range(a):
			start = time_start[j] - dt.timedelta(seconds = 60)
			time = time_start[j]
			stop = time_stop[j]
			cost = 0
			# print((stop - time).seconds/60)
			for i in range(len(dates)):
				#print(time)
				if(dates[i] == time):
					if((battery_charged[j] <= battery[j]*DoD) and (time <= stop)):
						if((stop - time).seconds / 60 < 60):
							# 5 minute window for the owner to come and start his vehicle
							cost += charging_rate * eff * price[i] * ((stop - time).seconds / 60 - 5) / 60 
							battery_charged[j] += charging_rate * ((stop - time).seconds / 60 - 5) / 60
							soc = max(0.2, battery_charged[j]/battery[j])
							percent_deg[j] += real_battery_degradation(t = (time - start).seconds/60, SOC = soc, N = N[j])
							break
						battery_charged[j] += charging_rate	#hourly
						cost += charging_rate*eff*price[i]	#hourly
						soc = max(0.2, battery_charged[j]/battery[j])
						percent_deg[j] += real_battery_degradation(t = (time - start).seconds/60, SOC = soc, N = N[j])
						time += dt.timedelta(hours = 1)
			total_cost[j] = cost
		return total_cost, battery_charged, percent_deg
	
	elif(state == 'discharging'):
		total_cost = np.zeros(a)
		battery_sold = np.zeros(a)
		if battery_left is None:
			battery_left = battery
		for j in range(a):
			if(battery_left[j] == 0):
				total_cost[j] = 0
				battery_sold[j] = 0
				break
			time = time_start[j]
			stop = rounddownTime([time + dt.timedelta(minutes = daily_work_mins[j])], roundTo = 60 * timedelta)[0]
			money_earned = 0
			for i in range(len(dates)):
				if(dates[i] == time):
					if((price[i] >= set_SP) and (battery_sold[j] <= battery_left[j]) and (time <= stop)):
						if((stop - time).seconds / 60 < 60):
						# 5 minute window for the owner to come and start his vehicle
							money_earned += charging_rate * eff * price[i] * ((stop - time).seconds / 60 - 5) / 60 
							battery_sold[j] += charging_rate * ((stop - time).seconds / 60 - 5) / 60
							break
						battery_sold[j] += charging_rate
						money_earned += charging_rate*eff*price[i]
						time += dt.timedelta(minutes = timedelta)
						battery_left[j] -= charging_rate
						if(battery_left[j] < battery[j]*(1 - DoD)):
							break
			total_cost[j] = money_earned
		return total_cost, battery_sold

	else:
		return None

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

def sampling(data, N, data2 = None, correlated = False, p = None, mask = None, index = None):
	'''Sample from data for a given sample size N
		Args: 
			data: The array to sample from (otype - array)
			N: desired sample size
			data2: This array is required if correlated is True
			p: probability of elements in data, length must be equal to data (otype - array)
			mask: A string or number that should be used to filter data. 
			index: An array of indices used to filter data instead of mask.Indices will over-ride mask.
		Returns:
			sampled_data: New array of randomly sampled elements. If correlated is True, then this returns a tuple of arrays.'''
	data = np.array(data)
	if correlated:
		assert(data2 is not None), 'Data is correlated, please input two arrays'
		data2 = np.array(data2)
		if mask is None and index is None:
			if p is None:
				p = np.asarray([1.0/len(data)]*len(data))
			sampled_indices = np.random.choice(np.arange(len(data)), size = N, p = p)
		else:
			if index is not None:
				index = index
			elif mask is not None:		
				index = data == ([mask]*len(data))
			new_data = data[index]
			p = np.asarray([1/len(new_data)]*len(new_data))
			sampled_indices = np.random.choice(np.arange(len(new_data)), size = N, p = p)
		sampled_data = data[sampled_indices]
		sampled_data2 = data2[sampled_indices]
		return (sampled_data, sampled_data2)
	else:	
		if mask is None and index is None:
			if p is None:
				p = np.asarray([1/len(data)]*len(data)) 
			sampled_data = np.random.choice(data, size = N, p = p)
		else:
			if index is not None:
				index = index
			elif mask is not None:		
				index = data == [mask]*len(data)
			new_data = data[index]
			p = np.asarray([1/len(new_data)]*len(new_data))
			sampled_data = np.random.choice(new_data, size = N, p = p)
		return sampled_data
		

def plot_histogram(data, bins = 10, xlabel = None, ylabel = None, yticks = None, xticks = None, title = None, save_to = None, ci95 = False, text = None, xtext = None, ytext = None):
	''' Make a histogram of the data in mpl'''
	plt.figure(figsize=(10, 8))
	plt.hist(data, bins = bins)
	plt.xlabel('{}'.format(xlabel))
	plt.ylabel('{}'.format(ylabel))
	plt.title('{}'.format(title))
	if ci95:
		y = np.percentile(data, (2.5, 95), axis = 0)
		plt.axvline(y[0], lw = 1, color = 'orange')
		plt.axvline(y[1], lw = 1, color = 'orange')
	if text is not None:
		assert(xtext is not None and ytext is not None), 'Please specify text location'
		plt.text(xtext, ytext, '{}'.format(text) )
	plt.tight_layout()
	if save_to is not None:
		plt.savefig('{}'.format(save_to))
	else:
		plt.show()

def real_battery_degradation(t, N = 0, SOC = 1., i_rate = 11.5, T = 318, Dod_max = 0.8):
	''' This function calculates the percentage of battery degradation happening at every time step
	Args: 
		t : time in minutes
		N : cycle number, default = 0, no cycling
		battery_capacity : maximum battery capacity of the EV in kW, default = 60 kW
		DoD : depth of discharge, default = 0.9
		i_rate : charging/discharging rate of the battery, default = 11.5 kWh = 11500/360 = 31.78 Ah
		T : battery temperature in Kelvin, default = 318K
	Returns: float, total percentage loss in the battery capacity at given timestep timedelta
	'''

	V_nom = 323		#V Nominal voltage of EV batteries
	#reference quantities
	T_ref = 298.15 	# K
	V_ref = 3.7		#V
	U_ref = 0.08	#V
	F = 96485 		# A s/mol	Faraday constant
	R_ug = 8.314	# J/K/mol

	#beginning of life increase in capacity
	d3 = 0.43 		#Ah
	d0_ref = 75.1 	#Ah
	E_ad01 = 34300	# J/mol
	E_ad02 = 74860	# J/mol

	Ah_dis = i_rate * 1000/360 #cumulative Ah discharged from the cell, Ah -> kWh * 1000 / V
	d0 = d0_ref* math.e**(-(E_ad01/R_ug * (1/T - 1/T_ref)) - ((E_ad02/R_ug)**2 * (1/T - 1/T_ref)**2))

	Qpos = d0 + d3 * (1 - math.e**(-Ah_dis/228))

	# Calendar aging
	t = t/24*60		#days
	b1_ref = 3.503 * 10**(-3)	#1/ day^0.5
	E_ab1 = 35392 	#J/mol
	alpha_b1 = 1
	gamma_b1 = 2.472
	beta_b1 = 2.157
	b0 = 1.07
	b3_ref = 2.805 * 10**(-2)
	E_ab3 = 42800 	#J/mol
	alpha_b3 = 0.0066
	tau_b3 = 5
	theta = 0.135
	b2_ref = 1.541 * 10**(-5)
	E_ab2 = -42800 	#J/mol

	#For Li-C anode
	xa_0 = 8.5 * 10**(-3)
	xa_100 = 7.8 * 10**(-1)
	xa = xa_0 + SOC * (xa_100 - xa_0)
	Ua = 0.6379 + 0.5416 * math.e**(-305.5309 * xa) + 0.044 * math.tanh(- (xa - 0.1958)/0.1088) - 0.1978 * math.tanh((xa- 1.057)/0.0854) - 0.6875 * math.tanh((xa + 0.0117)/0.0529) - 0.0175 * math.tanh((xa - 0.5692)/0.0875)
	#For LFP cathode
	xc_0 = 9.16 * 10**(-1)
	xc_100 = 4.5 * 10**(-2)
	xc = xc_0 + SOC * (xc_100 - xc_0)
	Uc = 3.4323 - 0.8428 * math.e**(-80.2493 * (1 - xc)**1.3198) - 3.2474 * 10**(-6) * math.e**(20.2645 * (1 - xc)**3.8003) + 3.2482 * 10**(-6) * math.e**(20.2646 * (1 - xc)**3.7995)

	Ut = Ua + Uc
	# Voc calc reference: Energies 2016, 9, 900 
	# Parameters for LMNCO cathodes at 45 degreeC
	a = 3.535
	b = -0.0571
	c = -0.2847
	d = 0.9475
	m = 1.4
	n = 2
	Voc = a + b * (-math.log(SOC))**m + c * SOC + d * math.e**(n * (SOC - 1))


	b1 = b1_ref * math.e**(-(E_ab1/R_ug * (1/T - 1/T_ref))) *  math.e**(gamma_b1 * Dod_max**beta_b1) * math.e**(alpha_b1 * F / R_ug * (0.0800056/T - U_ref/T_ref)) 

	b2 = b2_ref * math.e**(-(E_ab2/R_ug * (1/T - 1/T_ref)))
	b3 = b3_ref * math.e**(-(E_ab3/R_ug * (1/T - 1/T_ref))) * math.e**(alpha_b3 * F / R_ug * (Voc/T - V_ref/T_ref)) * (1 + theta * Dod_max)
	Qli = d0*(b0 - b1 * t**0.5 - b2 * N)# - b3 * (1 - math.e**(-t/ tau_b3)))

	# Cycle aging
	c2_ref = 3.9193*10**(-3)	#Ah/cycle
	beta_c2 = 4.54
	E_ac2 = -48260 	#J/mol
	c0_ref = 75.64
	E_ac0 = 2224	#J/mol

	c0 = c0_ref * math.e**(-E_ac0/R_ug * (1/T - 1/T_ref))
	c2 = c2_ref * math.e**(-E_ac2/R_ug * (1/T - 1/T_ref)) * Dod_max**beta_c2
	Qneg = (c0**2 - 2*c2*c0* N)**0.5

	Q_loss = min(Qli, Qpos, Qneg)
	print(Qli, Qpos, Qneg)
	return Q_loss

def temperature_model():
    # Most EVs have a thermal management system(TMS) so this is not required for now.
	pass
