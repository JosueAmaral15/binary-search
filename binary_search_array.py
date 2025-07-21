def search_for_array(expected_result, array, tolerance = 0.001):
    greater_than_function = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance))
    equals_function = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r+tolerance)-abs(x-r)+tolerance)*((-1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance)+1))
    average_function = lambda a, b: (a+b)/2
    lowest_function = lambda average2, initial_lower, result2, expected_result, tolerance: equals_function(average2, initial_lower, tolerance) * greater_than_function(result2, expected_result, tolerance)
    greatest_function = lambda average2, initial_upper, result2, expected_result, tolerance: equals_function(average2, initial_upper, tolerance) * greater_than_function(expected_result, result2, tolerance)
    array_length = len(array)
    initial_upper = array_length -1
    upper_value1 = initial_upper
    initial_lower = 1
    lower_value1 = initial_lower
    average1 = initial_upper #average_function(lower_value1, upper_value1)
    result1 = array[average1]
    is_global_maximum = 0
    is_global_minimum = 0
    
    continue_execution = greater_than_function(upper_value1, lower_value1, tolerance)
    
    while continue_execution and not is_global_maximum and not is_global_minimum:
        lower_value2 = average1 * greater_than_function(expected_result,result1, tolerance) + lower_value1*greater_than_function(result1, expected_result, tolerance)
        upper_value2 = upper_value1 * greater_than_function(expected_result,result1, tolerance) + average1*greater_than_function(result1, expected_result, tolerance)
        average2 = average_function(lower_value2, upper_value2) +average1*equals_function(expected_result, result1, tolerance)
        result2 = array[int(average2)]
        lower_value1 = lower_value2
        upper_value1 = upper_value2
        average1 = average2
        result1 = result2
        continue_execution = greater_than_function(upper_value2, lower_value2, tolerance)
        is_global_maximum = greatest_function (average2, initial_upper, result2, expected_result, tolerance)
        is_global_minimum = lowest_function (average2, initial_lower, result2, expected_result, tolerance)

    return int(average1), is_global_maximum, is_global_minimum

print(search_for_array(1, [1, 2, 3, 4, 5]))
