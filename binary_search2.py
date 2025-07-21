from random import randrange

def binary_search (wanted_number = 75, length = 100, range_number = 100, show_list = False):
    array_modified = []  # Cria uma nova lista para guardar os valores modificados. Seria a mesma coisa que o vetor v
    for i in range(length): #Acessa a lista e cria iterações para guardar valores na lista
        array_modified.append(randrange(range_number)) # Valores sendo guardados na lista
    array_modified.sort() # Faz a ordenação dos valores
    if show_list:
        print(array_modified)
    determine_the_most_approximate_value = False # booleano que indica se deve acontecer o deslocamento em um determinado index
    length_open_sub_vector = len(array_modified) # Seria o tamanho ou o comprimento da lista, ou seja, quantos elementos tem a lista.
    factor_binary_search = (length_open_sub_vector-1)//2 if length_open_sub_vector % 2 != 0 else length_open_sub_vector//2
    factor_is_zero = False # Verifica se a variável factor_binary_search tem valor zero
    first_index_open_sub_vector = 0 # Aponta sempre para o primeiro elemento do vetor.
    first_iteration = True # Indica a primeira iteração do algoritmo.
    greater_than_open_vector = False # indica que o índice de inserção está mais à direita do vetor aberto
    last_index_open_sub_vector = length_open_sub_vector -1 # Aponta para o último índice do vetor
    index_middle = last_index_open_sub_vector # Seria uma variável que aponta para o meio entre o intervalo dos limites superior e inferior da lista através do valor da média entre o limite superior com o inferior. Seria a mesma coisa que mk.
    index_middle_result = index_middle # Guarda o valor da média (que aponta para o índice entre o limite inferior do intervalo com o limite superior do intervalo) com um número natural.
    insert_left = False # O número deve ser inserido à esquerda ou à direita do valor encontrado.
    lower_than_open_vector = False # O valor cuja posição deve ser mudada no vetor deve ser inserido à esquerda do vetor aberto
    lower_limit = first_index_open_sub_vector # Trata-se do limite inferior do intervalo. Seria equivalente a lk.
    average = lambda a, b: (a+b)/2 # função que calcula a média entre dois valores. Seria equivalente a mk+1
    upper_limit = last_index_open_sub_vector # Limite superior do intervalo. Seria equivalente a uk.
    
    while not determine_the_most_approximate_value:

        if wanted_number == array_modified[index_middle]:
            determine_the_most_approximate_value = True
                                                        
        elif wanted_number > array_modified[index_middle]: # Se o número à esquerda for maior que o número da busca binária:
            if index_middle < last_index_open_sub_vector: # Se o número verificado à direita não tiver o mesmo índice referente ao maior índice do vetor aberto (overflow):
                
                lower_limit = index_middle_result
                index_middle_result = average(lower_limit, upper_limit)
                index_middle = int(index_middle_result)                
                
            else: # Do contrário, se os índices forem iguais, então devemos fazer a inserção à direita do número encontrado
                determine_the_most_approximate_value = True
                greater_than_open_vector = True
                
        else:
            if not first_iteration:
                upper_limit = index_middle_result
            else:
                first_iteration = False
                
            index_middle_result = average(lower_limit, upper_limit)
            index_middle = int(index_middle_result)
                    
            if index_middle == first_index_open_sub_vector:
                lower_than_open_vector = True
        
        if factor_is_zero:
            determine_the_most_approximate_value = True
            if wanted_number <= array_modified[index_middle]:
                insert_left = True
        
        if not determine_the_most_approximate_value and not factor_is_zero:
            factor_binary_search = factor_binary_search / 2
            if int(factor_binary_search) == 0:
                factor_is_zero = True
        else:
            perform_element_exchange_process = True
            
    return index_middle
    
if __name__ == "__main__":
    list_length = 100
    number = 24
    range_number = 80
    show_list = True
    index = binary_search(number, list_length, range_number, show_list)
    print(f"the value {number} was found (or is very close to) at index {index}")
