def parallel_overlap_count(data1, data2, threshold = 0):
      n_data1 = len(data1)
  n_data2 = len(data2)
  n_total = n_data1 * n_data2
  n_overlap = 0
  n_compare = 0
  n_process = 0
  max_threads = 14
  i_start = 0  
  while (n_process < n_data1):
    i_end = i_start + max_threads - 1        
    # ex_data1 = pymp.shared.array(data1[i_start:i_end,:,:]) 
    ex_counters = pymp.shared.array((2,), dtype='uint32') 
    with pymp.Parallel(max_threads) as p:
      for i in p.range(i_start, max_threads):
        for j in xrange(n_data2):
          if (is_similar(data1[i], data2[j])):
            #with p.lock:
            n_overlap += 1
              #print ("found ", ex_counters[0], " overlaps")
          with p.lock:
            n_compare += 1
          #if (ex_counters[1] % 1000 == 0):
          #    print("Made ", ex_counters[1], " processes out of ", n_data1)
    i_start = i_end + 1
  return n_overlap
