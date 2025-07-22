from math import acos

def search_for_function (y, function, tolerance = 0.001):
    greater_than_function = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance))
    equals_function = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r+tolerance)-abs(x-r)+tolerance)*((-1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance)+1))
    average_function = lambda a, b: (a+b)/2
    
    lower_value1 = float("-inf")
    upper_value1 = float("inf")
    result_with_zero_parameter = function(0)
    average1 = greater_than_function(y,result_with_zero_parameter,tolerance) -greater_than_function(result_with_zero_parameter,y,tolerance)
    result1 = function(average1)
    is_less = False
    
    if greater_than_function(y, result_with_zero_parameter, tolerance):
        while greater_than_function(abs(y), abs(result1), tolerance) and greater_than_function(abs(result_with_zero_parameter), abs(result1), tolerance): # and result1 > 0:
            average1 *= 2
            result1 = function(average1)
        
        upper_value1 = average1
        
        if not greater_than_function(result1,0,tolerance):
            is_less = True
            
        average2 = greater_than_function(result1,0,tolerance) -greater_than_function(0,result1,tolerance)
        result2 = function(average2)
        
        while greater_than_function(abs(result2), abs(y), tolerance) and greater_than_function(abs(result2), abs(result_with_zero_parameter), tolerance):
            average2 /= 2
            result2 = function(average2)
        lower_value1 = average2
    else:
        #is_less = True
        while greater_than_function(abs(result1), abs(y), tolerance) and greater_than_function(abs(result1), abs(result_with_zero_parameter), tolerance):
            average1 /= 2
            result1 = function(average1)
        lower_value1 = average1
        
        if not greater_than_function(result1,0,tolerance):
            is_less = True
        
        average2 = greater_than_function(result1,0,tolerance) -greater_than_function(0,result1,tolerance)
        result2 = function(average2)
        
        while greater_than_function(abs(y), abs(result1), tolerance) and greater_than_function(abs(result_with_zero_parameter), abs(result2), tolerance): # and result1 > 0:
            average2 *= 2
            result2 = function(average2)
        upper_value1 = average2
        
    continue_execution = greater_than_function(upper_value1, lower_value1, tolerance)
    value_not_found = greater_than_function (abs(y-average1), tolerance, tolerance)
    average1 = average_function(lower_value1, upper_value1) +average1*equals_function(y, result1, tolerance)
    result1 = function(average1)
    while value_not_found and continue_execution:
        if not is_less:
            lower_value2 = average1 * greater_than_function(y,result1, tolerance) + lower_value1*greater_than_function(result1, y, tolerance)
            upper_value2 = upper_value1 * greater_than_function(y,result1, tolerance) + average1*greater_than_function(result1, y, tolerance)
        else:
            lower_value2 =  lower_value1 * greater_than_function(y,result1, tolerance) + average1*greater_than_function(result1, y, tolerance)
            upper_value2 = average1 * greater_than_function(y,result1, tolerance) + upper_value1*greater_than_function(result1, y, tolerance)
        average2 = average_function(lower_value2, upper_value2) +average1*equals_function(y, result1, tolerance)
        result2 = function(average2)
        lower_value1 = lower_value2
        upper_value1 = upper_value2
        average1 = average2
        result1 = result2
        continue_execution = greater_than_function(upper_value2, lower_value2, tolerance)
        value_not_found = greater_than_function (abs(y-result1), tolerance, tolerance)
    return average2

print(search_for_function (1, lambda x: acos(0) -x))