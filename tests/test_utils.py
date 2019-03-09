import v2g.iutils as utils
import datetime as dt

def test_addtime_arrayint():
    array = [dt.datetime(2000, 1, 1, 2)]
    out = utils.addtime(array, 60)
    delta = out[0] - array[0]
    print(delta)
    assert delta.seconds == 60 * 60

def test_addtime_intint():
    array = dt.datetime(2000, 1, 1, 2)
    out = utils.addtime(array, 60)
    delta = out - array
    print(delta)
    assert delta.seconds == 60 * 60