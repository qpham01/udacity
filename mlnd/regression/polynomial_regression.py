#
#	Polynomial Regression
#
#	In this exercise we will examine more complex models of test grades as a function of 
#	sleep using numpy.polyfit to determine a good relationship and incorporating more data.
#
#
#   at the end, store the coefficients of the polynomial you found in coeffs
#
import matplotlib.pyplot as plt
import numpy as np

sleep = [5,6,7,8,10,12,16]
scores = [65,51,75,75,86,80,0]

coeffs = np.polyfit(sleep, scores, 2)
p = np.poly1d(coeffs)
plt.scatter(sleep, scores)
plt.plot(sleep, p(sleep), color='blue', linewidth=3)
plt.xlabel("sleep")
plt.ylabel("scores")
plt.show()

