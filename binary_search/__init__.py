from math import isfinite

class BinarySearch:
    def __init__(self, binary_search_priority_for_smallest_values = True, previous_value_should_be_the_basis_of_binary_search_calculations = False, previous_value_is_the_target = False, change_behavior_mid_step = False, number_of_attempts = 20):
        
        #functions
        self.greater_than_function = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance))
        self.equals_function = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r+tolerance)-abs(x-r)+tolerance)*((-1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance)+1))
        self.average_function = lambda a, b: (a+b)/2
        self.lowest_function = lambda average, initial_lower, result, expected_result, tolerance: self.equals_function(average, initial_lower, tolerance) * self.greater_than_function(result, expected_result, tolerance)
        self.greatest_function = lambda average, initial_upper, result, expected_result, tolerance: self.equals_function(average, initial_upper, tolerance) * self.greater_than_function(expected_result, result, tolerance)
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
            average1 = average2
            average2 = int(average2)
            result2 = array[average2]
            lower_value1 = lower_value2
            upper_value1 = upper_value2
            result1 = result2
            continue_execution = self.greater_than_function(upper_value2, lower_value2 +1, tolerance)
            is_global_maximum = self.greatest_function (average2, initial_upper, result2, expected_result,tolerance)
            is_global_minimum = self.lowest_function (average2, initial_lower, result2, expected_result, tolerance)

        return int(average1), is_global_maximum, is_global_minimum
    
    #def search_for_function (y, function, tolerance = 0.001):
    #    greater_than_function = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance))
    #    equals_function = lambda x, r, tolerance: round((1/(2*tolerance))*(abs(x-r+tolerance)-abs(x-r)+tolerance)*((-1/(2*tolerance))*(abs(x-r)-abs(x-r-tolerance)+tolerance)+1))
    #    average_function = lambda a, b: (a+b)/2
    #    
    #    lower_value1 = float("-inf")
    #    upper_value1 = float("inf")
    #    result_with_zero_parameter = function(0)
    #    average1 = greater_than_function(y,result_with_zero_parameter,tolerance) -greater_than_function(result_with_zero_parameter,y,tolerance)
    #    result1 = function(average1)
    #    is_less = False
    #    
    #    if greater_than_function(y, result_with_zero_parameter, tolerance):
    #        while greater_than_function(abs(y), abs(result1), tolerance) and greater_than_function(abs(result_with_zero_parameter), abs(result1), tolerance): # and result1 > 0:
    #            average1 *= 2
    #            result1 = function(average1)
    #        
    #        upper_value1 = average1
    #        
    #        if not greater_than_function(result1,0,tolerance):
    #            is_less = True
    #            
    #        average2 = greater_than_function(result1,0,tolerance) -greater_than_function(0,result1,tolerance)
    #        result2 = function(average2)
    #        
    #        while greater_than_function(abs(result2), abs(y), tolerance) and greater_than_function(abs(result2), abs(result_with_zero_parameter), tolerance):
    #            average2 /= 2
    #            result2 = function(average2)
    #        lower_value1 = average2
    #    else:
    #        #is_less = True
    #        while greater_than_function(abs(result1), abs(y), tolerance) and greater_than_function(abs(result1), abs(result_with_zero_parameter), tolerance):
    #            average1 /= 2
    #            result1 = function(average1)
    #        lower_value1 = average1
    #        
    #        if not greater_than_function(result1,0,tolerance):
    #            is_less = True
    #        
    #        average2 = greater_than_function(result1,0,tolerance) -greater_than_function(0,result1,tolerance)
    #        result2 = function(average2)
    #        
    #        while greater_than_function(abs(y), abs(result1), tolerance) and greater_than_function(abs(result_with_zero_parameter), abs(result2), tolerance): # and result1 > 0:
    #            average2 *= 2
    #            result2 = function(average2)
    #        upper_value1 = average2
    #        
    #    continue_execution = greater_than_function(upper_value1, lower_value1, tolerance)
    #    value_not_found = greater_than_function (abs(y-average1), tolerance, tolerance)
    #    average1 = average_function(lower_value1, upper_value1) +average1*equals_function(y, result1, tolerance)
    #    result1 = function(average1)
    #    while value_not_found and continue_execution:
    #        if not is_less:
    #            lower_value2 = average1 * greater_than_function(y,result1, tolerance) + lower_value1*greater_than_function(result1, y, tolerance)
    #            upper_value2 = upper_value1 * greater_than_function(y,result1, tolerance) + average1*greater_than_function(result1, y, tolerance)
    #        else:
    #            lower_value2 =  lower_value1 * greater_than_function(y,result1, tolerance) + average1*greater_than_function(result1, y, tolerance)
    #            upper_value2 = average1 * greater_than_function(y,result1, tolerance) + upper_value1*greater_than_function(result1, y, tolerance)
    #        average2 = average_function(lower_value2, upper_value2) +average1*equals_function(y, result1, tolerance)
    #        result2 = function(average2)
    #        lower_value1 = lower_value2
    #        upper_value1 = upper_value2
    #        average1 = average2
    #        result1 = result2
    #        continue_execution = greater_than_function(upper_value2, lower_value2, tolerance)
    #        value_not_found = greater_than_function (abs(y-result1), tolerance, tolerance)
    #    return average2
    
    def search_for_function(y, function, tolerance=1e-6, max_iter=1000):
        """
        Busca binária otimizada para encontrar x tal que function(x) ≈ y.

        - Detecta se a função é crescente ou decrescente.
        - Expande o intervalo na direção correta.
        - Minimiza |f(x)-y| sem depender de mudança de sinal.

        Parâmetros:
            y (float): valor alvo
            function (callable): função f(x)
            tolerance (float): precisão desejada
            max_iter (int): máximo de iterações

        Retorna:
            float: valor de x tal que f(x) ≈ y
        """
        # --- 1. Ponto inicial ---
        x0 = 0.0
        try:
            f0 = function(x0)
        except ValueError:
            raise ValueError("A função não está definida em x=0")

        if abs(f0 - y) < tolerance:
            return x0

        # --- 2. Detectar monotonicidade ---
        test_step = 1e-3
        try:
            f_test = function(x0 + test_step)
            is_increasing = f_test > f0
        except ValueError:
            # Se não dá para calcular, assume decrescente
            is_increasing = False

        # --- 3. Expandir intervalo na direção correta ---
        step = 1.0
        x_best, f_best = x0, f0

        for _ in range(50):  # limite de expansões
            x1 = x0 + step if (y > f0) == is_increasing else x0 - step
            try:
                f1 = function(x1)
            except ValueError:
                break  # atingiu limite do domínio
            if not isfinite(f1):
                break
            # Se encontrou valor mais próximo, atualiza melhor estimativa
            if abs(f1 - y) < abs(f_best - y):
                x_best, f_best = x1, f1
            # Se já está suficientemente próximo, encerra
            if abs(f1 - y) < tolerance:
                return x1
            # Expande
            x0, f0 = x1, f1
            step *= 2

        # Define intervalo [a,b] em torno do melhor valor encontrado
        a, b = x_best - abs(step), x_best + abs(step)

        # --- 4. Busca binária refinada ---
        for _ in range(max_iter):
            mid = (a + b) / 2
            try:
                fmid = function(mid)
            except ValueError:
                # Se a função não é definida, encolhe intervalo
                b = mid if mid > a else a
                continue

            if not isfinite(fmid):
                b = mid
                continue

            # Atualiza melhor estimativa
            if abs(fmid - y) < abs(f_best - y):
                x_best, f_best = mid, fmid

            if abs(fmid - y) < tolerance:
                return mid

            # Direção da busca binária
            if (fmid < y) == is_increasing:
                a = mid
            else:
                b = mid

        return x_best

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
    
    def binary_search_to_find_miniterm_from_dict (self, wanted_number, array_dict):
        determine_the_most_approximate_value = False # booleano que indica se deve acontecer o deslocamento em um determinado index
        array_dict_length = len(array_dict)-1 # Seria o tamanho ou o comprimento da lista, ou seja, quantos elementos tem a lista.
        factor_binary_search = (array_dict_length)//2 if array_dict_length % 2 != 0 else (array_dict_length+1)//2
        factor_is_zero = False # Verifica se a variável factor_binary_search tem valor zero
        first_iteration = True # Indica a primeira iteração do algoritmo.
        index_middle = array_dict_length # Seria uma variável que aponta para o meio entre o intervalo dos limites superior e inferior da lista através do valor da média entre o limite superior com o inferior. Seria a mesma coisa que mk.
        index_middle_result = index_middle # Guarda o valor da média (que aponta para o índice entre o limite inferior do intervalo com o limite superior do intervalo) com um número natural.
        lower_limit = 0 # Trata-se do limite inferior do intervalo. Seria equivalente a lk.
        average = lambda a, b: (a+b)/2 # função que calcula a média entre dois valores. Seria equivalente a mk+1
        upper_limit = array_dict_length # Limite superior do intervalo. Seria equivalente a uk.
        array_list = list(array_dict.values()) # array_list contém os resultados das operações da tabela verdade.
        
        while not determine_the_most_approximate_value:

            if wanted_number == array_list[index_middle]:
                determine_the_most_approximate_value = True
                                                            
            elif wanted_number > array_list[index_middle]: # Se o número à esquerda for maior que o número da busca binária:
                if index_middle < array_dict_length: # Se o número verificado à direita não tiver o mesmo índice referente ao maior índice do vetor aberto (overflow):
                    
                    lower_limit = index_middle_result
                    index_middle_result = average(lower_limit, upper_limit)
                    index_middle = int(index_middle_result)                
                    
                else: # Do contrário, se os índices forem iguais, então devemos fazer a inserção à direita do número encontrado
                    determine_the_most_approximate_value = True                
            else:
                if not first_iteration:
                    upper_limit = index_middle_result
                else:
                    first_iteration = False
                    
                index_middle_result = average(lower_limit, upper_limit)
                index_middle = int(index_middle_result)
            
            if factor_is_zero:
                determine_the_most_approximate_value = True
            
            if not determine_the_most_approximate_value and not factor_is_zero:
                factor_binary_search = factor_binary_search / 2
                if int(factor_binary_search) == 0:
                    factor_is_zero = True
        
        if index_middle > 0: # Se a distância entre o valor do índice encontrado for maior que o valor do índice anterior, então desejamos aquele valor com a menor distância ou com a menor diferença com o valor desejado.
            if abs(wanted_number -array_list[index_middle]) > abs(wanted_number - array_list[index_middle-1]):
                index_middle -= 1
                
        groups = dict()
        for key, value in array_dict.items():
            groups[value] = key # Valor é o resultado das operações, e key seria o índice desses resultados.
        
        #print(f"groups: {groups}")
        
        return array_list[index_middle], groups[array_list[index_middle]]
    
    def reset(self):
        if self.binary_search_priority_modified:
            self.binary_search_priority_modified = False
            self.binary_search_priority_for_smallest_values = not self.binary_search_priority_for_smallest_values
