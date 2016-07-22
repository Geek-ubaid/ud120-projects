#!/usr/bin/python

def outlierCleaner(predictions, ages, net_worths):
    """
    Clean away the 10% of points that have the largest residual errors 
    (difference between the prediction and the actual net worth). 
    Return a list of tuples named cleaned_data where each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []
    ### your code goes here
    limit = int(0.9 * len(ages))
    errors = map(lambda x,y: abs(x[0] - y[0]), predictions, net_worths)    
    #Zip, sort and unzip. To only sort, use zip().sort()
    errors, ages, net_worths = zip(*sorted(zip(errors, ages, net_worths)))    
    cleaned_data = [(age, net_worth, error) for age, net_worth, error in zip(ages, net_worths, errors)]    
    return cleaned_data[0:limit]