import math


# 快速幂
def power(x, n):
    if n < 0:
        return 1 / power(x, -n)
    if n == 0:  # base
        return 1
    if n == 1:  # base
        return x
    b = power(x, n // 2)
    if n % 2 == 0:
        return b * b
    else:
        return x * b * b


def find_max(a, b):
    return max(a, b), min(a, b)


# 第二大的数
def second_largest(nums):
    size = len(nums)
    if size < 2:
        raise Exception('array length must be more than 2')
    first = second = -math.inf
    for i in range(size):
        if nums[i] > first:
            second = first
            first = nums[i]
        elif nums[i] > second:
            second = nums[i]
    return second


# 二维数组查找 减而直至
# matrix = [
#        [1, 3, 5, 7],
#        [10, 11, 16, 20],
#        [23, 30, 34, 50]
#    ]
def search_matrix(matrix, tar):
    if not matrix or not matrix[0]:
        return False
    row = 0
    col = len(matrix[0]) - 1
    while row < len(matrix) and col >= 0:
        num = matrix[row][col]
        if num == tar:
            return True
        elif num < tar:
            row += 1
        else:
            col += 1
    return False


if __name__ == '__main__':
    print('数组中第2大的数')
    print(second_largest([2, 3, 4, 10, 100]))

    matrix = [
        [1, 3, 5, 7],
        [10, 11, 16, 20],
        [23, 30, 34, 50]
    ]
    print(search_matrix(matrix, 13))
