try:
    test_index()
except Exception:
    print('no')


def test_index():
    a = [1, 2, 4, 5]
    a.index(3)
