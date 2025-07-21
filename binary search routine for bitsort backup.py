factor_binary_search = length_open_sub_vector//2
                            if factor_binary_search:
                                factor_is_zero = False # Verifica se a variável factor_binary_search tem valor zero
                                determine_the_most_approximate_value = False # booleano que indica se deve acontecer o deslocamento em um determinado index
                                #greater_than_open_vector = False # indica que o índice de inserção está mais à direita do vetor aberto
                                #lower_than_open_vector = False # O valor cuja posição deve ser mudada no vetor deve ser inserido à esquerda do vetor aberto
                                insert_left = False # O número deve ser inserido à esquerda ou à direita do valor encontrado.
                                while not determine_the_most_approximate_value:
                                    
                                    if left_number > array_modified[index_middle]: # Se o número à esquerda for maior que o número da busca binária:
                                        if not factor_is_zero:
                                            if index_middle < array_length -length_closed_sub_vector -1: # Se o número verificado à direita não tiver o mesmo índice referente ao maior índice do vetor aberto (overflow):
                                                index_middle += factor_binary_search
                                            else: # Do contrário, se os índices forem iguais, então devemos fazer a inserção à direita do número encontrado
                                                determine_the_most_approximate_value = True
                                                greater_than_open_vector = True
                                                perform_element_exchange_process = True
                                        else:
                                            determine_the_most_approximate_value = True
                                            perform_element_exchange_process = True
                                    else:
                                        if index_middle: # Esta condicional é apenas uma precaução para que possamos utilizar a variável index_middle sem que a mesma variável tenha o valor zero, e isso devido ao fato de que estamos trabalhando com index de vetores. Teoricamente pode dar sempre True (Tautologia), porque a condição if not factor_is_zero tem que dar False para que index_middle possa ser zero. Esta condição só existe por precaução, e por isso deve continuar existindo.
                                            if not factor_is_zero:
                                                    index_middle -= factor_binary_search
                                            else:
                                                determine_the_most_approximate_value = True
                                                perform_element_exchange_process = True
                                                insert_left = True
                                                #first_index_open_sub_vector = calculate_first_index_of_open_subarray (array_length, length_open_sub_vector, length_closed_sub_vector)
                                                
                                                if index_middle == first_index_open_sub_vector: # Queremos que o index_middle sempre aponte para o index cujo valor esteja dentro do intervalo do sub vetor aberto.
                                                    lower_than_open_vector = True
                                                    
                                                if length_open_sub_vector % 2 != 0 and index_middle == first_index_open_sub_vector +1 and left_number < array_modified[index_middle -1]: # Se dentro do intervalo do sub vetor aberto o valor de left_number for menor que o valor do primeiro número do sub vetor aberto, então apontaremos o índex para esse primeiro número.
                                                    lower_than_open_vector = True
                                                    index_middle = first_index_open_sub_vector # Apontaremos para o primeiro valor do sub vetor aberto.
                                                        
                                                    
                                    if not determine_the_most_approximate_value and not factor_is_zero:
                                        factor_binary_search //= 2
                                        if factor_binary_search == 0:
                                            factor_is_zero = True
