def divide(dividend, divisor):
    sign = 1
    if (dividend > 0) ^ (divisor > 0):
        sign = -1
    dividend, divisor = abs(dividend), abs(divisor)
    om = (1 << 31) - 1 if sign > 0 else (1 << 31)
    if divisor == 1:
        return dividend * sign if dividend <= om else (1 << 31) - 1
    res = 0
    divisor_old = divisor
    while dividend >= divisor_old:
        divisor = divisor_old
        k = 1
        while dividend >= divisor * k:
            dividend -= divisor * k
            res += k
            k *= divisor

    return res * sign if res <= om else (1 << 31) - 1


print(divide(-2147483648, -1))
