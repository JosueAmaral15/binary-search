class BinarySearch:
    def __init__(self, binary_search_priority_for_smallest_values = True, previous_value_should_be_the_basis_of_binary_search_calculations = False, previous_value_is_the_target = False, change_behavior_mid_step = False, number_of_attempts = 20):
        
        #functions
        self.greater_than_function = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance))
        self.equals_function = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r+tolerance)-abs(x-r)+tolerance)*((-1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance)+1))
        self.average_function = lambda a, b: (a+b)/2
        self.lowest_function = lambda average2, initial_lower, result2, expected_result, tolerance: self.equals_function(average2, initial_lower, tolerance) * self.greater_than_function(result2, expected_result, tolerance)
        self.greatest_function = lambda average2, initial_upper, result2, expected_result, tolerance: self.equals_function(average2, initial_upper, tolerance) * self.greater_than_function(expected_result, result2, tolerance)
        self.rational_function = lambda a : a/(a+1)
        self.arithmetic_progression_function = lambda a1, n, r: a1+ n*r
        #self.selector_without_graduation_and_without_inclusion = lambda x, a, b, d, m: (a -m)*(ceil((1/2)*(abs(x-b)-abs(x-b-1)+1))) +m +d
        self.selector_without_graduation_and_with_inclusion = lambda x, a, b, d, m: (a -m)*(floor((1/2)*(abs(x-b+1)-abs(x-b)+1))) +m +d
        self.equals_with_no_graduation = lambda x, b, d: d*(-ceil((1/2)*(abs(x-b)-abs(x-b-1)+1))+1)*floor((1/2)*(abs(x-b+1)-abs(x-b)+1))
        
        #values
        self.binary_search_priority_for_smallest_values = binary_search_priority_for_smallest_values
        self.previous_value_should_be_the_basis_of_binary_search_calculations = previous_value_should_be_the_basis_of_binary_search_calculations
        self.previous_value_is_the_target = previous_value_is_the_target
        self.number_of_attempts = number_of_attempts
        self.binary_search_priority_modified = False
        self.change_behavior_mid_step = change_behavior_mid_step
        
    
    def search_for_array(self, expected_result, array, tolerance = 0.001):
        array_length = len(array)
        initial_upper = array_length -1
        upper_value1 = initial_upper
        initial_lower = 0
        lower_value1 = initial_lower
        average1 = initial_upper #self.average_function(lower_value1, upper_value1)
        result1 = array[average1]
        is_global_maximum = 0
        is_global_minimum = 0
        
        continue_execution = self.greater_than_function(upper_value1, lower_value1+1, tolerance)
        
        while continue_execution and not is_global_maximum and not is_global_minimum:
            lower_value2 = average1 * self.greater_than_function(expected_result,result1, tolerance) + lower_value1*self.greater_than_function(result1, expected_result, tolerance)
            upper_value2 = upper_value1 * self.greater_than_function(expected_result,result1, tolerance) + average1*self.greater_than_function(result1, expected_result, tolerance)
            average2 = self.average_function(lower_value2, upper_value2) +average1*self.equals_function(expected_result, result1, tolerance)
            lower_value1 = lower_value2
            upper_value1 = upper_value2
            average1 = average2
            average2 = int(average2)
            result2 = array[average2]
            result1 = result2
            continue_execution = self.greater_than_function(upper_value2, lower_value2 +1, tolerance)
            is_global_maximum = self.greatest_function (average2, initial_upper, result2, expected_result,tolerance)
            is_global_minimum = self.lowest_function (average2, initial_lower, result2, expected_result, tolerance)

        return int(average1), is_global_maximum, is_global_minimum
    
    def search_for_function (self, y, function, tolerance = 0.001):
        
        lower_value1 = float("-inf")
        upper_value1 = float("inf")
        average1 = self.greater_than_function(y,0,tolerance) -self.greater_than_function(0,y,tolerance)
        result1 = function(average1)

        if self.greater_than_function(y, result1, tolerance):
            while self.greater_than_function(y, result1, tolerance):
                average1 *= 2
                result1 = function(average1)
            upper_value1 = average1
            lower_value1 = average1 /2
        else:
            while self.greater_than_function(result1, y, tolerance):
                average1 /= 2
                result1 = function(average1)
            lower_value1 = average1
            upper_value1 = average1 * 2
        
        continue_execution = self.greater_than_function(upper_value1, lower_value1, tolerance)
        value_not_found = self.greater_than_function (abs(y-average1), tolerance, tolerance)
        
        while value_not_found and continue_execution:
            lower_value2 = average1 * self.greater_than_function(y,result1, tolerance) + lower_value1*self.greater_than_function(result1, y, tolerance)
            upper_value2 = upper_value1 * self.greater_than_function(y,result1, tolerance) + average1*self.greater_than_function(result1, y, tolerance)
            average2 = self.average_function(lower_value2, upper_value2) +average1*self.equals_function(y, result1, tolerance)
            result2 = function(average2)
            lower_value1 = lower_value2
            upper_value1 = upper_value2
            average1 = average2
            result1 = result2
            continue_execution = self.greater_than_function(upper_value2, lower_value2, tolerance)
            value_not_found = self.greater_than_function (abs(y-average1), tolerance, tolerance)

        return average2

    def search_for_function_step (self, y, function, lower_value, upper_value, average, result):
        average2 = self.average_function(lower_value, upper_value) +average*self.equals_with_no_graduation(y,result,1)
        result2 = function(average2)
        lower_value2 = average * self.selector_without_graduation_and_with_inclusion(y,1,result,0,0) + lower_value * self.selector_without_graduation_and_with_inclusion(result, 1, y, 0, 0)
        upper_value2 = upper_value * self.selector_without_graduation_and_with_inclusion(y,1,result,0,0) + average*self.selector_without_graduation_and_with_inclusion(result, 1, y, 0, 0)
        return lower_value2, upper_value2, average2, result2
    
    def binary_search_by_step (self, steps, minimum_limit, maximum_limit, previous_value = 0):
        if not self.previous_value_should_be_the_basis_of_binary_search_calculations:
            if self.binary_search_priority_for_smallest_values:
                return minimum_limit +(maximum_limit -minimum_limit) * self.rational_function(steps)
            else:
                return maximum_limit -(maximum_limit -minimum_limit) * self.rational_function(steps)
        else:
            if self.change_behavior_mid_step and not self.binary_search_priority_modified and steps > self.number_of_attempts//2:
                    self.binary_search_priority_for_smallest_values = not self.binary_search_priority_for_smallest_values
                    self.binary_search_priority_modified = True
                    
            if self.previous_value_is_the_target:
                if not self.change_behavior_mid_step:
                    if self.binary_search_priority_for_smallest_values: 
                        return minimum_limit +(previous_value-minimum_limit) * self.rational_function(steps)
                    else:
                        # Vai ter que começar do maior valor para o valor alvo, que é previous values
                        return maximum_limit -(maximum_limit -previous_value) * self.rational_function(steps)
                else:
                    if self.binary_search_priority_for_smallest_values: 
                        return minimum_limit +(previous_value-minimum_limit) * self.rational_function(steps % (self.number_of_attempts//2 +1))
                    else:
                        # Vai ter que começar do maior valor para o valor alvo, que é previous values
                        return maximum_limit -(maximum_limit -previous_value) * self.rational_function(steps % (self.number_of_attempts//2 +1))
            else:
                if not self.change_behavior_mid_step:
                    if self.binary_search_priority_for_smallest_values:
                        return previous_value -(previous_value -minimum_limit) * self.rational_function(steps)
                    else:
                        # vai do previous_value para o maior valor definido pelo maximum_limit
                        return previous_value +(maximum_limit -previous_value) * self.rational_function(steps)
                else:
                    if self.binary_search_priority_for_smallest_values:
                        return previous_value -(previous_value -minimum_limit) * self.rational_function(steps % (self.number_of_attempts//2 +1))
                    else:
                        # vai do previous_value para o maior valor definido pelo maximum_limit
                        return previous_value +(maximum_limit -previous_value) * self.rational_function(steps % (self.number_of_attempts//2 +1))
    
    def linear_search_step(self, steps, minimum_limit, maximum_limit, previous_value = 0):
        if not self.previous_value_should_be_the_basis_of_binary_search_calculations:
            if self.binary_search_priority_for_smallest_values:
                return self.arithmetic_progression_function(minimum_limit, steps, (maximum_limit-minimum_limit)/self.number_of_attempts)
            else:
                return self.arithmetic_progression_function(minimum_limit, self.number_of_attempts -steps, (maximum_limit-minimum_limit)/self.number_of_attempts)
        else:
            if self.change_behavior_mid_step and steps > self.number_of_attempts//2 and not self.binary_search_priority_modified:
                self.binary_search_priority_for_smallest_values = not self.binary_search_priority_for_smallest_values
                self.binary_search_priority_modified = True
            
            if self.previous_value_is_the_target:
                if self.binary_search_priority_for_smallest_values: 
                    return self.arithmetic_progression_function(minimum_limit,steps if not self.change_behavior_mid_step else 2*(steps % (self.number_of_attempts//2 +1)), (previous_value-minimum_limit)/self.number_of_attempts)
                else:
                    # Vai ter que começar do maior valor para o valor alvo, que é previous values
                    return self.arithmetic_progression_function(maximum_limit, steps if not self.change_behavior_mid_step else 2*(steps % (self.number_of_attempts//2 +1)), -(maximum_limit-previous_value)/self.number_of_attempts)
            else:
                if self.binary_search_priority_for_smallest_values:
                    return self.arithmetic_progression_function(previous_value, steps if not self.change_behavior_mid_step else 2*(steps % (self.number_of_attempts//2 +1)), (minimum_limit-previous_value)/self.number_of_attempts)
                else:
                    # vai do previous_value para o maior valor definido pelo maximum_limit
                    return self.arithmetic_progression_function(previous_value, steps if not self.change_behavior_mid_step else 2*(steps % (self.number_of_attempts//2 +1)), (maximum_limit-previous_value)/self.number_of_attempts)
    
    def reset(self):
        if self.binary_search_priority_modified:
            self.binary_search_priority_modified = False
            self.binary_search_priority_for_smallest_values = not self.binary_search_priority_for_smallest_values
