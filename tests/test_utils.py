import v2g.iutils as utils
import datetime as dt

def test_add_time_arrayint():
    array = [dt.datetime(2000, 1, 1, 2)]
    out = utils.add_time(array, 60)
    delta = out[0] - array[0]
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

def test_cost_calc_discharging():
    #all day discharge
    state = 'discharging'
    #price is always 1
    dates = [dt.datetime(1,1,1,x) for x in range(24)]
    prices = [1 for _ in dates]
    #huge battery
    battery = 10000
    #start right away
    time_start = dt.datetime(1,1,1,0)
    #no cycles
    N = 0
    #working hours
    daily_work_mins = 24 * 60
    #charging rate in Energy / seconds?
    charging_rate = 1
    result = utils.cost_calc(state, dates, prices, battery,
                             time_start, N,
                             daily_work_mins=daily_work_mins,
                             charging_rate=charging_rate,
                             eff = 1)
    #total money
    money = sum(prices) * charging_rate * 60
    assert result[0] == money
