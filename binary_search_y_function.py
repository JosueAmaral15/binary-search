def binary_search(y, function, tolerance):
    greater_than = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance))
    equals = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r+tolerance)-abs(x-r)+tolerance)*((-1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance)+1))
    average = lambda a, b: (a+b)/2
    
    lower_value1 = float("-inf")
    upper_value1 = float("inf")
    average1 = greater_than(y,0,tolerance) -greater_than(0,y,tolerance)
    result1 = function(average1)

    if greater_than(y, result1, tolerance):
        while greater_than(y, result1, tolerance):
            average1 *= 2
            result1 = function(average1)
        upper_value1 = average1
        lower_value1 = average1 /2
    else:
        while greater_than(result1, y, tolerance):
            average1 /= 2
            result1 = function(average1)
        lower_value1 = average1
        upper_value1 = average1 * 2
    
    continue_execution = greater_than(upper_value1, lower_value1, tolerance)
    value_not_found = greater_than (abs(y-average1), tolerance, tolerance)
    
    while value_not_found and continue_execution:
        lower_value2 = average1 * greater_than(y,result1, tolerance) + lower_value1*greater_than(result1, y, tolerance)
        upper_value2 = upper_value1 * greater_than(y,result1, tolerance) + average1*greater_than(result1, y, tolerance)
        average2 = average(lower_value2, upper_value2) +average1*equals(y, result1, tolerance)
        result2 = function(average2)
        lower_value1 = lower_value2
        upper_value1 = upper_value2
        average1 = average2
        result1 = result2
        continue_execution = greater_than(upper_value2, lower_value2, tolerance)
        value_not_found = greater_than (abs(y-average1), tolerance, tolerance)

    return average2

#print(binary_search(5, lambda x: x**2, 0.0000001))

number = int(input("enter the desired number that would be the expected result of a function: "))
function = input("enter the function that you want to use to find the number (write it in lambda format, e.g. lambda x: x**2): ")
tolerance = float(input("enter the desired tolerance which defines how accurate the result should be (e.g. 0.001): "))
print(f"The result is: {binary_search(number, eval(function), tolerance)}.")
