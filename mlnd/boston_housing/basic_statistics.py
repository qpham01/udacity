from housing_prices import *
from methods import *

# TODO: Minimum price of the data
minimum_price = prices.min()

# TODO: Maximum price of the data
maximum_price = prices.max()

# TODO: Mean price of the data
mean_price = prices.mean()

# TODO: Median price of the data
median_price = prices.median()

# TODO: Standard deviation of prices of the data
std_price = prices.std()

# Show the calculated statistics
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)

# TODO: Minimum rm of the data
minimum_rm = features['RM'].min()

# TODO: Maximum rm of the data
maximum_rm = features['RM'].max()

# TODO: Mean rm of the data
mean_rm = features['RM'].mean()

# TODO: Median rm of the data
median_rm = features['RM'].median()

# TODO: Standard deviation of features['RM'] of the data
std_rm = features['RM'].std()

print ""
print "Minimum rm: {:,.2f}".format(minimum_rm)
print "Maximum rm: {:,.2f}".format(maximum_rm)
print "Mean rm: {:,.2f}".format(mean_rm)
print "Median rm {:,.2f}".format(median_rm)
print "Standard deviation of rms: {:,.2f}".format(std_rm)

# TODO: Minimum ptratio of the data
minimum_lstat = features['LSTAT'].min()

# TODO: Maximum ptratio of the data
maximum_lstat = features['LSTAT'].max()

# TODO: Mean ptratio of the data
mean_lstat = features['LSTAT'].mean()

# TODO: Median ptratio of the data
median_lstat = features['LSTAT'].median()

# TODO: Standard deviation of features['LSTAT'] of the data
std_lstat = features['LSTAT'].std()

print ""
print "Minimum lstat: {:,.2f}".format(minimum_lstat)
print "Maximum lstat: {:,.2f}".format(maximum_lstat)
print "Mean lstat: {:,.2f}".format(mean_lstat)
print "Median lstat {:,.2f}".format(median_lstat)
print "Standard deviation of lstats: {:,.2f}".format(std_lstat)

# TODO: Minimum ptratio of the data
minimum_ptratio = features['PTRATIO'].min()

# TODO: Maximum ptratio of the data
maximum_ptratio = features['PTRATIO'].max()

# TODO: Mean ptratio of the data
mean_ptratio = features['PTRATIO'].mean()

# TODO: Median ptratio of the data
median_ptratio = features['PTRATIO'].median()

# TODO: Standard deviation of features['PTRATIO'] of the data
std_ptratio = features['PTRATIO'].std()

print ""
print "Minimum ptratio: {:,.2f}".format(minimum_ptratio)
print "Maximum ptratio: {:,.2f}".format(maximum_ptratio)
print "Mean ptratio: {:,.2f}".format(mean_ptratio)
print "Median ptratio {:,.2f}".format(median_ptratio)
print "Standard deviation of ptratios: {:,.2f}".format(std_ptratio)

y_true = [3.0, -0.5, 2.0, 7.0, 4.2]
y_pred = [2.5, 0.0, 2.1, 7.8, 5.3]
print "R-squared {:,.3f}".format(performance_metric(y_true, y_pred))