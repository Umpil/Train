'''
  Строки
  Строки имеют тип данных str, объявляются они с помощью "" или ''
  Есть свои нюансы межджу '' и "", но о них позже
'''

print(type(""))

string_var = "Я строка "
string_var_from_int = str(5)
print(type(string_var))
print(string_var)
print(type(string_var_from_int))
print(string_var_from_int, "\n")


'''
  Строки поддерживают операции:
  Сложения
  Умножения на целое число, строка повторится n-раз 
  Принадлежности, возвращает булевую переменную (True, False), - True, если символ есть в строке и False в ином случае
'''

print("Привет," + " Мир")
print(string_var * 2)
print("Я" in string_var, "\n")


'''
  В Python существуют встроенные функции, работающие со строками, так
  len() - возвращает длину строки
  chr() - преобразует целое число в символ строки
  ord() - преобразует символ строки в целое число
  str() - преобразует тип данных переданного объекта к str
'''
print(len("Строка длины 15"))
print(chr(97), chr(98))
print(ord("c"), ord("d"))
print(type(str(1.231)), "\n")  # Изначально тип данных float

'''
  Строки поддерживают индексацию с помощью [], индексация начинается с нуля, где ноль - первый символ строки (слева)
  Строки поддерживают слайсинг с помощью [n:m], n - индекс начала среза, m - индекс конца среза, не включая его
  Индексирование также может быть отрицетальным, -1 - последний символ строки, -2 - предпоследний и т.д.
'''
print("012345678"[5])  # Будет 5
print("012345678"[2:7])  # Будет со 2 по 6, так как 7 не включается
print("012345678"[-1])
print("012345678"[-7: -1], "\n")  # -1 > -7

'''
  Также возможно указывать шаг для среза с помощью [n:m:k], n - начало среза, m -  конец среза, не включая его, k - шаг  
'''
print("012345678"[0 : len("012345678") : 2])  # Будет взят каждый второй символ, т.е. все чётные
print("012345678"[0 : len("012345678") - 1 : 2], "\n")  # Будет взят каждый второй символ, т.е. все чётные, кроме 8


'''
  Форматирование строк
  Легче на примере ебануть
'''
some_var = "pimer"
print("2 =", 2, ", some_var=", some_var)  # долго и сложно

print(f"2 = {2}, some_var = {some_var}", "\n")  # быстро, просто и удобно

'''
  Изменение строк не поддерживается питоном, поэтому есть несколько способов изменить строку
  1. Взять срез строки
  2. метод replace(old, new, необязательно: n), old - старый паттерн (что изменяем), new - новый паттерн (на что изменяем),
    n - количество замен (сколько раз заменяем), если не передан n, то заменятся все символы в строке по паттерну old
  3. С помощью списка 
'''
string_var_need_refactor = "Меня бы поменять"
new_string = string_var_need_refactor[:5] + "ТУТ" + string_var_need_refactor[6:] # Так мы заменили символ б
print(new_string)

new_string = string_var_need_refactor.replace("б", "ТУТ")
print(new_string)

new_string = list(string_var_need_refactor)
new_string[5] = "ТУТ"
new_string = "".join(new_string)
print(new_string, "\n")

#  Ох, бля, если это строки, то что будет в списках и словарях..., но надо, моя хорошая, надо

'''
  Тут можно было бы написать про методы строк, но их так много, что проще при встрече с конкретной задачей найти их,
  Но самые основные вот:
  lower() - преобразует строку в нижний регистр
  capitalize() - Делает строку с большой буквы
  isalpha() - возрващает True, если все символы в строке есть алфавитные, False в ином случае
  isnumeric() - возрващает True, если все символы в строке есть числа (без пробелов), False в ином случае
'''
print("БоЛьШиЕ БуКаВы".lower())
print("пишем маленькие букавы на первой позиции".capitalize())
print("asd".isalpha())
print("55".isnumeric())
