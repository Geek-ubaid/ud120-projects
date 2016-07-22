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

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
#print "Dictionary Keys = ", enron_data.keys()
#print "Number of people = ", len(enron_data)
#print "Number of features per person = ", len(enron_data["SKILLING JEFFREY K"])
#print "Number of POIs = ", len([tuple for tuple in enron_data.keys() if enron_data[tuple]["poi"] == 1])
#print "Value of James Prentice's stock = ", enron_data["PRENTICE JAMES"]["total_stock_value"]
#print "Number of emails from Wesley Colwell to persons of interest = ", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
#print "Value of stock options exercised by Jeffrey Skilling = ", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
#print "Most money taken home by ", max([enron_data[name]["total_payments"] for name in enron_data.keys() if name.find("SKILLING") != -1 or name.find("LAY") != -1])
#print enron_data["SKILLING JEFFREY K"]
#print "Number of people with a quantified salary = ", len([name for name in enron_data.keys() if enron_data[name]["salary"] != 'NaN'])
#print "Number of people with a quantified email id = ", len([name for name in enron_data.keys() if enron_data[name]["email_address"] != 'NaN'])
print "Number of people with total payments not defined = ", len([name for name in enron_data.keys() if enron_data[name]["total_payments"] == 'NaN'])
print "^ As a % of the whole dataset = ", len([name for name in enron_data.keys() if enron_data[name]["total_payments"] == 'NaN'])/float(len(enron_data))
print "Number of POIs who have total_payments as NaN = ", len([name for name in enron_data.keys() if enron_data[name]["total_payments"] == 'NaN' and enron_data[name]["poi"] == 1])/float(len(enron_data))