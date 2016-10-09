import pymp

n = 0
ex_array = pymp.shared.array((100,), dtype='uint8')
with pymp.Parallel(8) as p:
    for index in p.range(0, 100):
        with p.lock:
            n += 1
            ex_array[0] += 1
        # The parallel print function takes care of asynchronous output.
        print('Yay! {} done!'.format(index))
print "ex_array[0] is now ", ex_array[0]
print "n is now ", n
