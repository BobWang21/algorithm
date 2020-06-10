import math


# 快速幂
def power(x, n):
    if n == 0:  # base
        return 1
    if n == 1:  # base
        return x
    if n < 0:
        return 1 / power(x, -n)
    b = power(x, n // 2)
    return x ** (n % 2) * b * b


if __name__ == '__main__':
    print('\n快速幂')
    print(power(3, 4))
