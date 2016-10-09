#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
from __future__ import division
import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

people_count = len(enron_data);
feature_count = len(enron_data.values()[0])
print "people count:  ", people_count
print "feature count: ", feature_count
#print "features:      ", enron_data.values()[0]
#print "names:         ", enron_data.keys()

# count poi
poi_count = 0
for i in xrange(people_count):
    if enron_data.values()[i]["poi"] == 1:
        poi_count += 1

print "poi count:      ", poi_count
print ""
print "Jame Prentice:      ", enron_data["PRENTICE JAMES"]["total_stock_value"]
print "Wesley Colwell:     ", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
print "Jeffrey K Skilling: ", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
print "Jeffrey K Skilling: ", enron_data["SKILLING JEFFREY K"]["total_payments"]
print "Kenneth Lay:        ", enron_data["LAY KENNETH L"]["total_payments"]
print "Andrew Fastow:      ", enron_data["FASTOW ANDREW S"]["total_payments"]

people_salary_count = 0
people_email_count = 0
people_no_payment = 0
poi_no_payment = 0
for i in xrange(people_count):
    if enron_data.values()[i]["salary"] != "NaN":
        people_salary_count += 1
    if enron_data.values()[i]["email_address"] != "NaN":
        people_email_count += 1
    if enron_data.values()[i]["total_payments"] == "NaN":
        people_no_payment += 1
        if enron_data.values()[i]["poi"] == 1:
            poi_no_payment += 1

print "valid salary count: ", people_salary_count
print "valid email count:  ", people_email_count
print "no payment count:   ", people_no_payment
print "no payment percent: ", round(people_no_payment / people_count, 3)
print "poi no payment: ", poi_no_payment
