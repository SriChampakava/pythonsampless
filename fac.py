def factorial(n):
    return 1 if n in (0,1) else n*(factorial(n-1))
number=int(input("Enter a number to find the factorial :"))
print("the factorial value is ",factorial(number))
