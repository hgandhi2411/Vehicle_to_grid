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
import scipy.stats as ss

# Required numbers
Sample = 2559    #total EVs in NYC 2015
prefix = 'C:/Users/hetag/Desktop/Vehicle_to_grid/'
cars = pd.read_csv(prefix + 'Cars.csv', delimiter= ',')
rated_dist_dict, charge_time_dict = {}, {}
for i in range(len(cars.Battery)):
	rated_dist_dict[cars.Battery[i]] = cars.dist[i]
	charge_time_dict[cars.Battery[i]] = cars.Charge_time[i]
battery = np.random.choice(cars.Battery, size = Sample, p = cars.prob)
print(battery)
dist_one_kWh = np.array([rated_dist_dict[i] for i in battery])
complete_charging_time = np.array([charge_time_dict[i] for i in battery])

eff = 0.62 #roundtrip efficiency of Tesla S 
#reference: Shirazi, Sachs 2017
DoD = 0.90   #depth of discharge
charging_rate = 11.5		
# Considering AC Level2 Charging using SAE J1772 (240V/80A), standard on board chargers are capable of delivering 48A with the high amperage version being able to deliver 72A - tesla.com
# Super charger gives 17.2 kWh/hr charging rate

#battery degradation cost
lifecycles = 5300		#for lithium ion battery from Gerad's article
energy_throughput = 2*lifecycles*battery*DoD 		#Ln.Es.DoD (kWh)
battery_cap_cost = 410			# $/kWh Nykvist and nilsson 2015, for 2014
# Alternately from https://electrek.co/2017/02/18/tesla-battery-cost-gigafactory-model-3/
bat_degradation = battery * battery_cap_cost/energy_throughput			# using Gerad's equation, cb is the cost of current storage which will be introduced later.
lcos = 1142/1000			# $/kWh, Lazards LCOS V2.0 2016, Commercial lithium ion category average

working_days = 250

#For optimal scenario we need a set selling price
SP = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]		#$/kWh
x = len(SP)

np.set_printoptions(precision = 2, threshold = np.inf)

def state_pop(state):
	pop_files = {'Arizona': 'ss16paz.csv', 'California': 'ss16pca.csv', 'DC': 'ss16pdc.csv', 'Illinois': 'ss16pil.csv', 'Massachusetts': 'ss16pma.csv', 'New York': 'ss16pny.csv', 'Texas': 'ss16ptx.csv'}
	LBMP_file = {'Arizona': 'phoenix.csv', 'California': 'sfca.csv', 'DC': 'washingtondc.csv', 'Illinois': 'chicago.csv', 'Massachusetts': 'boston.csv', 'New York': 'nyc.csv', 'Texas': 'houston.csv'}
	return '/Population_data/' + pop_files[state], '/LBMP/' + LBMP_file[state]

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
def roundupTime(dtarray=None, roundTo=60*60):
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

def rounddownTime(dtarray=None, roundTo=60*60):
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

def cost_calc(state, dates, price, time_start, time_stop = None, daily_work_mins = None, set_SP = 0, battery_left = None):
	'''This function will calculate the cost of electricity for based on the state -discharging or charging
		Args: state = 'charging' or 'discharging'
			set_SP = The selling price set  by the user, default = 0
			daily_work_mins = An array of working hours of users
			dates = The time stamps for the data used (otype- array)
			price = Cost of electricity at the given time stamps from data (otype - array)
			time_start = start of the time interval for which cost of electricity needs to be calculated (otype - array)
			time_stop = end of interval which started, default = None (otype - array)
		return: total_cost = An array of costs calculated, battery_sold = battery sold for V2G(only for state = 'discharging')
	'''
	a = len(time_start)
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
			stop = rounddownTime([time + dt.timedelta(minutes = daily_work_mins[j])], roundTo = 60 * 5)[0]
			money_earned = 0
			for i in range(len(dates)):
				if(dates[i] == time):
					if((price[i] >= set_SP) and (battery_sold[j] <= battery_left_to_sell[j]) and (time <= stop)):
						if((stop - time).seconds / 60 < 60):
    						# 5 minute window for the owner to come and start his vehicle
							money_earned += charging_rate * eff * price[i] * ((stop - time).seconds / 60 - 5) / 60 
							battery_sold[j] += charging_rate * ((stop - time).seconds / 60 - 5) / 60
							break
						battery_sold[j] += charging_rate
						money_earned += charging_rate*eff*price[i]
						time += dt.timedelta(hours = 1)
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

#Extracting and histograming commute distance data for NYS from NHTS 
# Sampling using the probability vectors for distance and time 
dist = np.genfromtxt(prefix + 'PERV2PUB.CSV', delimiter = ',', dtype = None, usecols = (103), skip_header = 1, filling_values = 0)	#commute distance
time = np.genfromtxt(prefix + 'PERV2PUB.CSV', delimiter = ',', dtype = None, skip_header=1, usecols = (84), filling_values = 0)    #commute time
#work_time = np.genfromtxt('PERV2PUB.csv', delimiter = ',', dtype = None, usecols = (98), skip_header = 1, filling_values = 0)
state = np.genfromtxt(prefix + 'PERV2PUB.CSV', delimiter = ',', dtype = None, usecols = (48), skip_header= 1, filling_values = 0)		#state in column 49 of PERV2PUB.csv
state_list = ['Arizona', 'California', 'DC', 'Illinois', 'Massachusetts', 'New York']
mask = {'Arizona': 'AZ', 'California': 'CA', 'DC': 'DC', 'Illinois': 'IL', 'Massachusetts': 'MA', 'New York': 'NY', 'Texas': 'TX'}

result_path = prefix + 'Results/{}/'.format(dt.datetime.today().date()) + 'var_bat/'
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
					

	p_dist = np.asarray([1.0/len(dist_sample)]*len(dist_sample))

	commute_index = np.random.choice(np.arange(len(dist_sample)), size = Sample, p = p_dist)
	commute_dist = np.zeros(Sample)
	commute_time = np.zeros(Sample)

	for i in range(Sample):
		commute_dist[i] = dist_sample[commute_index[i]]
		commute_time[i] = time_sample[commute_index[i]]

	#print('Commute distance = ',commute_dist)
	#print('Commute time = ', commute_time)

	#Actual calculations of battery usage and selling
	battery_used_for_travel = 2 * commute_dist/dist_one_kWh  #kWh
	battery_left_to_sell = battery*DoD - battery_used_for_travel

	# Sampling departure times
	pop_file, lbmp_file = state_pop(s)
	Time_depart_for_work = np.genfromtxt( prefix + pop_file, delimiter = ',', filling_values = 0, skip_header = 1, usecols = (91), dtype = None)
	depart_values, depart_keys = create_dict(Time_depart_for_work, bins = np.arange(1,151))
	depart_time_decode = pd.read_csv(prefix + 'input.csv')
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
	work_hrs_per_week = np.genfromtxt(prefix + pop_file, delimiter = ',', filling_values = 0, skip_header = 1, usecols = (71), dtype = None)
	work_hrs_values, work_hrs_keys = create_dict(work_hrs_per_week, bins = np.arange(1,100))
	p_work_hrs = work_hrs_values/np.sum(work_hrs_values)
	weekly_work_hrs = np.random.choice(work_hrs_keys, size = Sample, p = p_work_hrs)
	daily_work_mins = weekly_work_hrs*60/5 
	plt.clf()

	#LBMP Data
	data = pd.read_csv(prefix + lbmp_file)
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
	day = dt.datetime(2017,1,1,0,0)
	count = 1

	#Using holiday calender
	holidays = USFederalHolidayCalendar().holidays(start = '2017-01-01', end = '2018-01-01').to_pydatetime()
	battery_charged = np.zeros((x, Sample))

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

				if battery_charged[i].all() != 0:
					cost_discharging, battery_sold = cost_calc('discharging', date_set, price_set, time_discharge_starts, time_stop = None, daily_work_mins = daily_work_mins, set_SP = SP[i], battery_left = battery_charged[i] + battery)
				else:
					cost_discharging, battery_sold = cost_calc('discharging', date_set, price_set, time_discharge_starts, time_stop = None, daily_work_mins = daily_work_mins, set_SP = SP[i], battery_left = battery_charged[i])
				final_discharge_cost[i] += cost_discharging
				battery_used[i] = battery_sold + battery_used_for_travel
				time_leaving_work = addtime(time_arrival_work, daily_work_mins)
				time_reach_home = addtime(time_leaving_work, commute_time)
				#print('time_reach_home', time_reach_home)
				time_charging_starts = roundupTime(time_reach_home)
				#print('time_charging_starts', time_charging_starts)
				time_charging_stops = addtime(time_charging_starts, complete_charging_time*battery_used[i]/battery)
				time_charging_stops = roundupTime(time_charging_stops)
				cost_charging, battery_charged = cost_calc('charging', date_set, price_set, time_charging_starts, time_stop = time_charging_stops, battery_left = battery - battery_used[i]) 
				cost_charging += battery_used[i] * bat_degradation * eff
				final_cdgdn[i] += battery_used[i] * bat_degradation * eff
				final_charge_cost[i] += cost_charging

			#Cost of commute without V2G
			charge_commute_stop = addtime(time_charging_starts, complete_charging_time*battery_used_for_travel/battery)
			charge_commute_stop = rounddownTime(np.asarray(charge_commute_stop))
			cost_commute, battery_charged_commute = cost_calc('charging', date_set, price_set, time_charging_starts, time_stop = charge_commute_stop, battery_left = battery - battery_used_for_travel) 
			cost_commute += battery_used_for_travel * bat_degradation * eff
			final_commute_cdgdn += battery_used_for_travel * bat_degradation * eff
			final_commute_cost += cost_commute
			#print(final_commute_cost)

			for i in range(Sample):
				time_arrival_work[i] += dt.timedelta(days = 1)
			k += 24
			day+=dt.timedelta(days=1)
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

	sns.set(context = 'talk', style = 'darkgrid', font= 'Times New Roman', palette= 'hls')
	mlib.rcParams['figure.figsize'] = (10, 8)
	cdgdn_commute = np.mean(final_commute_cdgdn)
	cch_commute = np.mean(final_commute_cost - final_commute_cdgdn)
	Annual_savings = np.zeros((x,Sample))
	mean, cdgdn, cch, rt, y = [], [], [], [], []
	ci = 0.95
	z = ss.norm.ppf((1+ci)/2)

	for i in range(x):
		Annual_savings[i] = final_discharge_cost[i] - final_charge_cost[i] - (0.0 - final_commute_cost)
		#Calculating statistics and CIs
		mean.append(np.mean(Annual_savings[i]))
		y.append(np.percentile(Annual_savings[i], [2.5, 97.5]))
		cdgdn.append(np.mean(final_cdgdn[i]))
		cch.append(np.mean(final_charge_cost[i] - final_cdgdn[i]))
		rt.append(np.mean(final_discharge_cost[i])) #revenue from V2G

	argmax = np.argmax(mean)
	key = '{}\n{}'.format(s,SP[argmax])
	max_savings[key] = Annual_savings[argmax]
	pricetaker_savings[s] = Annual_savings[0]


	if not os.path.exists(result_path + s):
		os.makedirs(result_path + s)
	output = result_path + '{}/data.csv'.format(s)
	results = {'Distance': commute_dist, 'Time':commute_time, 'Work hours': weekly_work_hrs, 'Work time': time_depart_from_home, 'Battery': battery}
	results.update({'Savings{}'.format(a):b for a,b in zip(SP, Annual_savings)})
	results.update({'Cycle{}'.format(a):b for a,b in zip(SP, battery_cycles)})
	results = pd.DataFrame.from_dict(results)
	results.to_csv(output)

	#Writing values to a file
	output = open(result_path + s + '/data.txt', 'w+')
	output.write('efficiency = {} \nSP \t\tSavings mean\t\tCI95 \t\t\t Mean cycles\n'.format(eff))
	for i in range(x):
		output.write('{} \t\t{} \t\t{} \t\t{}\n'.format(SP[i], mean[i], y[i], np.mean(battery_cycles[i])))
	output.write('cdgdn = {} \ncch = {} \nrt = {} \ncdgdn_commute = {} \ncch_commute = {} \n'.format(cdgdn, cch, rt, cdgdn_commute, cch_commute))
	output.write('\n')
	output.write('\ndistance = {}, mean = {:.2f} \ntime = {}, mean = {:.2f} \nwork hours = {}, mean = {:.2f} \n'.format(commute_dist, np.mean(commute_dist), commute_time, np.mean(commute_time), daily_work_mins/60.0, np.mean(daily_work_mins/60.0)))
	output.close()

	#creating filenames for storing results
	files, files1, files2, files3 = [], [], [], []
	for i in range(x):
		files1.append(result_path + s + '/Savings{}.svg'.format(SP[i]))
		files.append(result_path + s + '/Battery{}.svg'.format(SP[i]))
		files2.append(result_path + s + '/Time{}.svg'.format(SP[i]))
		files3.append(result_path + s + '/Distance{}.svg'.format(SP[i]))

	#Histogramming data
	low = math.floor(np.amin(Annual_savings))
	high = math.ceil(np.amax(Annual_savings))
	bins = np.linspace(low, high, 20)

	for i in range(x):
		plt.hist(Annual_savings[i], bins = bins, alpha = 0.8, label = 'SP = {} \nMean cycles = {:.2f}'.format(SP[i], np.mean(battery_cycles[i])), fc = None, ec = None)
		#plt.text(-100,150,'Mean cycles = {}'.format(np.mean(battery_cycles[i])))
		#plt.title(r'$ Optimal\ scenario\ - Comparison\ of\ different\ selling\ prices\ $')
		plt.xlabel(r'$Savings\ from\ V2G\ over\ normal\ commute (\$) $')
		plt.axvline(y[i][0], lw = 1, color = 'orange')
		plt.axvline(y[i][1], lw = 1, color = 'orange')
		plt.ylabel(r'$Number\ of\ EV\ Users\ $')
		plt.legend(loc = 'best', fontsize = 11)
		plt.savefig(files1[i])
		plt.clf() #clear existing figure
		#print(Annual_savings[i])

	plt.close() #close existing figure

	def square_plot(ax):
		#make the aspect ratio of data match that of plot
		xl, xh, yl, yh = ax.axis()
		ax.set_aspect((xh - xl) / (yh - yl), adjustable='box')

	time_depart = np.zeros(Sample)
	for i in range(Sample):
		time_depart[i] = time_depart_from_home[i].hour + time_depart_from_home[i].minute/60
	
	for i in range(x):

		# distance jointplot
		g = (sns.jointplot(battery, Annual_savings[i], kind = 'hex', gridsize = 20, marginal_kws={'bins': 20})).set_axis_labels("Battery size (kWh)", "Annual Savings")
		# g.ax_joint.add_patch(ellipse)
		# yticks = np.round(np.arange(np.min(Annual_savings[i]), np.max(Annual_savings[i]), step= int((np.max(Annual_savings[i]) - np.min(Annual_savings[i]))/10))/500.0)*500
		# g.ax_joint.set_yticks(yticks)
		plt.title('SP = {}'.format(SP[i]))
		plt.savefig(files[i])
		plt.close() #close the figure

		cmap = mlib.cm.hot
		#cmap = sns.light_palette('#4682b4', as_cmap = True)
		m = mlib.cm.ScalarMappable(cmap=cmap)
		m.set_array(Annual_savings[i]) 
		

		# a weighted joint plot showing relation between time, time of departure and savings 
		fig = plt.figure(figsize=(8, 7))
		grid = plt.GridSpec(2, 3, width_ratios = [3,1, 0.1], height_ratios=[1,3], hspace=0.05, wspace= 0.07) # hspace = 0.1, wspace = 0.2
		main_ax = fig.add_subplot(grid[1,0])
		y_hist = fig.add_subplot(grid[1,1], yticklabels=[]) #sharey=main_ax)# , xticklabels=[], yticklabels=[])#
		x_hist = fig.add_subplot(grid[0,0], xticklabels=[]) #sharex=main_ax)# , yticklabels=[], xticklabels=[])# 
		cax = fig.add_subplot(grid[1,2])

		# scatter points on the main axes
		a = main_ax.hexbin(commute_time, time_depart, C = Annual_savings[i] , cmap = cmap, gridsize = 15, vmin = low, vmax = high, edgecolors = None)
		main_ax.set_xlabel('Commute time (minutes)')
		main_ax.set_ylabel('time of departure (hour of the day) ')

		# histogram on the attached axes
		x_hist.hist(commute_time, 15, histtype='bar',
					orientation='vertical', color = '#ffa07a')
		#x_hist.invert_yaxis()

		y_hist.hist(time_depart, 15, histtype='bar',
					orientation='horizontal', color = '#ffa07a')
		y_hist.invert_xaxis()

		#fig.subplots_adjust(right = 0.9)
		cbar1 = plt.colorbar(a, cax = cax)
		cbar1.ax.set_ylabel('Annual savings from V2G')
		plt.suptitle('SP = {}'.format(SP[i]))
		plt.savefig(files2[i])
		plt.close()

		cmap1 = mlib.cm.winter
		#cmap = sns.light_palette('#4682b4', as_cmap = True)
		m = mlib.cm.ScalarMappable(cmap=cmap1)
		m.set_array(Annual_savings[i]) 
		
		# a weighted joint plot showing relation between battery size, commute distance and savings 
		fig = plt.figure(figsize=(8, 7))
		grid = plt.GridSpec(2, 3, width_ratios = [3,1, 0.1], height_ratios=[1,3], hspace=0.05, wspace= 0.07) # hspace = 0.1, wspace = 0.2
		main_ax = fig.add_subplot(grid[1,0])
		y_hist = fig.add_subplot(grid[1,1], yticklabels=[]) #sharey=main_ax)# , xticklabels=[], yticklabels=[])#
		x_hist = fig.add_subplot(grid[0,0], xticklabels=[]) #sharex=main_ax)# , yticklabels=[], xticklabels=[])# 
		cax = fig.add_subplot(grid[1,2])

		print('Starting the distance plot!')
		# scatter points on the main axes
		a = main_ax.hexbin(battery, commute_dist, C = Annual_savings[i] , cmap = cmap1, gridsize = 15, vmin = low, vmax = high, edgecolors = None)
		main_ax.set_xlabel('Battery size (kWh)')
		main_ax.set_ylabel('Commute distance (miles)')

		# histogram on the attached axes
		x_hist.hist(battery, 15, histtype='bar',
					orientation='vertical', color = '#40E0D0')
		#x_hist.invert_yaxis()

		y_hist.hist(commute_dist, 15, histtype='bar',
					orientation='horizontal', color = '#40E0D0')
		y_hist.invert_xaxis()

		#fig.subplots_adjust(right = 0.9)
		cbar1 = plt.colorbar(a, cax = cax)
		cbar1.ax.set_ylabel('Annual savings from V2G')
		plt.suptitle('SP = {}'.format(SP[i]))
		plt.savefig(files3[i])
		plt.close()

# end of state for loop
max_savings = pd.DataFrame.from_dict(max_savings)
sns.violinplot(data = max_savings)
plt.ylabel('Annual Savings from V2G($)')
plt.savefig('Results/violin.svg')
plt.close()

sns.violinplot(data = pd.DataFrame.from_dict(pricetaker_savings))
plt.savefig('Results/pricetaker.svg')
plt.ylabel('Annual Savings from V2G')
plt.title('Price taker scenario')
plt.close()