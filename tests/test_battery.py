from v2g import Battery

def test_charge():
    b = Battery(100, 0.8)
    assert b.soc == 1
    b.charge(100, 10)
    assert b.soc == 1
    b.soc = 0
    b.charge(100,10)
    assert b.soc == 1
    b.soc = 0
    b.charge(100, 1)
    assert b.soc == 0.8

def test_discharge():
    b = Battery(100, 0.8)
    b.discharge(100, 10, stop_soc=0.1)
    assert abs(b.soc - 0.1) < 0.001
