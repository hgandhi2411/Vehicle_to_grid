from v2g import Battery

def test_charge():
    # efficiency test
    b = Battery(100, 0.8)
    assert abs(b.soc - 1) < 0.001
    # Almost full battery plugged in
    b.soc = 0.99
    b.charge(100, 10)
    assert abs(b.soc - 1) < 0.001
    # start with an empty battery, charge for a long period
    b.soc = 0
    b.charge(100,10)
    assert abs(b.soc - 1) < 0.001
    # Start with an empty battery, charge for short period
    b.soc = 0
    b.charge(100, 1)
    assert abs(b.soc - 0.8) < 0.001

def test_discharge():
    b = Battery(100, 0.8)
    b.discharge(100, 10, stop_soc=0.1)
    assert abs(b.soc - 0.1) < 0.001
