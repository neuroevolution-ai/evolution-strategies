import timeit

def main():
    setup = '''import numpy as np
def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)

def batched_weighted_sum(weights, vecs, batch_size):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

def old(proc, _noise):
    g, count = batched_weighted_sum(
        proc,
        _noise,
        batch_size=500
    )
    return g, count

def new(proc, noise):
    count_new = len(proc)
    g_new = np.dot(proc, noise)
    return g_new, count_new
noise = np.random.randn(2000, 75272)
proc = np.random.randn(2000)
_noise = (n for n in noise)
'''
    code_old = '''
old(proc, _noise)
    '''

    code_new = '''
new(proc, noise)
    '''

    print(timeit.timeit(stmt=code_old, setup=setup, number=20))
    print(timeit.timeit(stmt=code_new, setup=setup, number=20))


if __name__ == '__main__':
    main()

