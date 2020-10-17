def binary_search(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1


def first_bigger_pos(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] < target:
            l = mid + 1
        else:
            r = mid
    return l if nums[l] >= target else -1


def first_large(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] <= target:
            l = mid + 1
        else:
            r = mid
    return l if nums[l] > target else -1



def search_last_pos(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l + 1) // 2
        if nums[mid] <= target:
            l = mid
        else:
            r = mid - 1
    return l if nums[l] == target else -1


def find_duplicate(nums):
    n = len(nums)

    def count(target):
        return sum([1 if v <= target else 0 for v in nums])

    l, r = 0, n - 1
    while l < r:
        mid = l + (r - l) // 2
        if count(mid) <= mid:
            l = mid + 1
        else:
            r = mid
    return l




if __name__ == '__main__':
    print(binary_search([1, 3, 5], 1))
    print(search_last_pos([1, 3, 3, 5], 3))
    print(find_duplicate([3,1,3,4,2]))
