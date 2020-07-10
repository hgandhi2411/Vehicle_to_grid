import v2g.iutils as utils
from v2g import Battery
import datetime as dt
import numpy as np
import pytest

def test_add_time_arrayint():
    array = [dt.datetime(2000, 1, 1, 2), dt.datetime(2005, 1, 3, 5)]
    out = utils.add_time(array, 60)
    delta = out[0] - array[0]
    print(delta)
    assert delta.seconds == 60 * 60
    delta = out[1] - array[1]
    print(delta)
    assert delta.seconds == 60 * 60

def test_add_time_intint():
    array = dt.datetime(2000, 1, 1, 2)
    out = utils.add_time(array, 60)
    delta = out - array
    print(delta)
    assert delta.seconds == 60 * 60

def test_round_dt_up():
    #4:12 am
    time = dt.datetime(2000,1,1,4,12)
    rtime = utils.round_dt_up(time, minutes=5)
    assert rtime.minute == 15
    assert rtime.hour == 4
    #8:30am
    time = dt.datetime(2000,1,1,8,30)
    rtime = utils.round_dt_up(time, hours=12, minutes=15)
    assert rtime.minute == 30
    assert rtime.hour == 12
    rtime = utils.round_dt_up(time, minutes=60)
    assert rtime.minute == 0
    assert rtime.hour == 9

def test_round_dt_down():
    #4:12 am
    time = dt.datetime(2000,1,1,4,12)
    rtime = utils.round_dt_down(time, minutes=5)
    assert rtime.minute == 10
    assert rtime.hour == 4
    #8:30am
    time = dt.datetime(2000,1,1,8,30)
    rtime = utils.round_dt_down(time, hours=12, minutes=15)
    assert rtime.minute == 30
    assert rtime.hour == 0
    rtime = utils.round_dt_down(time, minutes=60)
    assert rtime.minute == 0
    assert rtime.hour == 8

def test_find_price_for_timestamp():
    time = dt.datetime(2020, 5, 1, 6, 22)
    dates, prices = [], []
    for j in range(1,9):
        for i in range(0, 60, 5):
            dates.append(dt.datetime(2020, 5, 1, j, i))
            prices.append(i * 0.1)
    dates = np.asarray(dates)
    nearest, rate = utils.find_price_for_timestamp(time, dates, prices)
    assert nearest == dt.datetime(2020, 5, 1, 6, 20)
    assert rate == 2

def test_cost_calc_discharging():
    #all day discharge
    state = 'discharging'
    #price is always 1
    dates = [dt.datetime(1,1,1,x) for x in range(24)]
    prices = [1 for _ in dates]
    #huge battery
    battery = Battery(1000, 1.0)
    #start right away
    time_start = dt.datetime(1,1,1,0)
    #working hours
    daily_work_mins = 24 * 60
    #charging rate in Energy / seconds?
    charging_rate = 1
    result = utils.cost_calc(state, dates, prices, battery,
                             time_start,
                             daily_work_mins=daily_work_mins,
                             charging_rate=charging_rate)
    #total money
    money = sum(prices) * charging_rate
    assert result == pytest.approx(money, 0.01)

def test_cost_calc_empty_battery():
    #price is always 1
    dates = [dt.datetime(1,1,1,x,y*5) for x,y in zip(range(24), range(12))]
    prices = [1 for _ in dates]
    #start right away
    time_start = dt.datetime(1,1,1,0,0)
    #working hours
    daily_work_mins = 24 * 60
    #charging rate in Energy / hour
    charging_rate = 0.1
    result = utils.cost_calc('charging', dates, prices, Battery(1, 1.0),
                             time_start,
                             daily_work_mins=daily_work_mins,
                             charging_rate=charging_rate,
                             time_stop = dates[-1] + dt.timedelta(minutes=5), 
                             battery_left=1)
    #total cost
    assert result < 10e-9

    battery = Battery(0.1, 1.0)
    battery.discharge(1, 1)
    result = utils.cost_calc('discharging', dates, prices, battery,
                             time_start,
                             daily_work_mins=daily_work_mins,
                             charging_rate=charging_rate,
                             timedelta=5,
                             time_stop = None, battery_left=0.001)
    #total money gained
    assert result < 10e-9

def test_cost_calc_charging():
    #all day discharge
    state = 'charging'
    #price is always 1
    dates = [dt.datetime(1,1,1,x, 0) for x in range(24)]
    prices = [1 for _ in dates]
    #huge battery
    battery = Battery(10000, 1.0)
    battery.soc = 0
    #start right away
    time_start = dt.datetime(1,1,1,0,0)
    #working hours
    daily_work_mins = 24 * 60
    #charging rate in Energy / seconds?
    charging_rate = 1
    result = utils.cost_calc(state, dates, prices, battery,
                             time_start = time_start,
                             daily_work_mins=daily_work_mins,
                             charging_rate=charging_rate,
                             time_stop = dates[-1] + dt.timedelta(minutes=60))
    #total money
    money = sum(prices) * charging_rate
    assert result == pytest.approx(money, 0.01)

def test_create_dict():
    #test data
    data = np.array([i//2 for i in range(10)])
    values, keys = utils.create_dict(data, bins = range(6))
    np.testing.assert_equal(keys, range(5))
    #zeros are not accounted for in create_dict
    np.testing.assert_equal(values[1:], 2*np.ones(4))

def test_dist_time_battery_sampling():
    # distance
    dist = [i*3 for i in range(1, 6)]
    # commute time
    time = dist*2
    #sample size
    N = 10
    # range of EVs - fixed at 30
    ev_range = np.ones(N) * 30
    result = utils.dist_time_battery_correlated_sampling(dist, time, ev_range, N = N)
    #Make sure the sample size is right
    assert len(result[0]) == N
    assert len(result[1]) == N
    #Make sure distance is less than range
    assert (result[0] * 2 < ev_range).all()
    #Make sure value error is raised if size ev_range is not N
    pytest.raises(ValueError, utils.dist_time_battery_correlated_sampling, *(dist, time, ev_range[:10], 20))

def test_profit():
    base = dt.datetime(2017, 1, 1, 0)
    dates = [base + dt.timedelta(minutes = 5) for x in range(72288)]
    #price is always 1
    prices = [1 for _ in dates]
    #battery specifications
    battery = 100 # kWh
    ev_range = 295 # miles
    charging_time = 510 #minutes
    # commute variables
    distance = 10
    time = distance * 2.5
    #work pattern
    time_arrival_work = dt.datetime(2017,1,1,8, 30)
    daily_work_mins = 200

    result = utils.profit(x = 2, battery = battery,
                        battery_used_for_travel = distance * 2 * battery/ev_range,
                        commute_distance = distance, commute_time = time,
                        complete_charging_time = charging_time,
                        time_arrival_work = time_arrival_work,
                        daily_work_mins = daily_work_mins,
                        dates = dates, price = prices, bat_degradation = 10)

    #Assert profit is zero
    assert result['Annual_savings'] == 0
    #assert discharge cost is zero
    assert result['V2G_discharge_cost'] == 0
    #assert charging cost and commute cost are the same
    assert result['V2G_charge_cost'] == result['Commute_cost']
    # assert q_deg and q_deg_commute are equal
    assert result['V2G_capacity_fade'] == result['Commute_capacity_fade']

    result = utils.profit(x = 0, battery = battery,
                        battery_used_for_travel = distance * 2 * battery/ev_range,
                        commute_distance = distance, commute_time = time,
                        complete_charging_time = charging_time,
                        time_arrival_work = time_arrival_work,
                        daily_work_mins = daily_work_mins,
                        dates = dates, price = prices, bat_degradation = 10)

    #assert discharge cost is greater than zero
    assert result['V2G_discharge_cost'] > 0
    #assert charging cost is expected to be more than commute cost due to additional degradation
    assert result['V2G_charge_cost'] > result['Commute_cost']
    # degradation is more when discharging everyday
    assert result['V2G_capacity_fade'] > result['Commute_capacity_fade']
    # total degradation must be less than the battery capacity
    assert result['V2G_capacity_fade'] < 1
