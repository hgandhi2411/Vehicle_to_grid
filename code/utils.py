import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import datetime as dt
from pandas.tseries.holiday import USFederalHolidayCalendar
import scipy.stats as ss
import os

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

def makeDir(prefix, target):
	if not os.path.exists(prefix + target):
		os.makedirs(prefix + target)
	return


def cost_calc(state, dates, price, battery, time_start, time_stop = None, daily_work_mins = None, set_SP = 0, battery_left = None, timedelta = 60):
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
		return: total_cost = An array of costs calculated, battery_sold = battery sold for V2G(only for state = 'discharging')
				battery_sold/battery_charged 
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
		if battery_left is None:
			battery_charged = np.zeros(a)
		else:
			battery_charged = battery_left
		for j in range(a):
			time = time_start[j]
			stop = time_stop[j]
			cost = 0
			for i in range(len(dates)):
				#print(time)
				if(dates[i] == time):
					if((battery_charged[j] <= battery[j]*DoD) and (time <= stop)):
						if((stop - time).seconds / 60 < 60):
							# 5 minute window for the owner to come and start his vehicle
							cost += charging_rate * eff * price[i] * ((stop - time).seconds / 60 - 5) / 60 
							battery_charged[j] += charging_rate * ((stop - time).seconds / 60 - 5) / 60
							break
						battery_charged[j] += charging_rate	#hourly
						cost += charging_rate*eff*price[i]	#hourly
						time += dt.timedelta(hours = 1)
			total_cost[j] = cost
		return total_cost, battery_charged
	
	elif(state == 'discharging'):
		total_cost = np.zeros(a)
		battery_sold = np.zeros(a)
		for j in range(a):
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
				p = np.asarray([1/len(data)]*len(data))
			sampled_indices = np.random.choice(np.arange(len(data)), size = N, p = p)
		else:
			if index is not None:
				index = index
			elif mask is not None:		
				index = data == [mask]*len(data)
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