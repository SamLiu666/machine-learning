
print("######################## 1.1 将序列分解为单独的变量")
data = [ 'ACME', 50, 91.1, (2012, 12, 21) ]
name, shares, price, date = data
print("1.1 赋值: ", name, shares, price, date)
_, shares, price, _ = data
print(shares, price)

print("######################## 1.2 解压可迭代对象赋值给多个变量")
record = ('Dave', 'dave@example.com', '773-555-1212', '847-555-1212')
name, email, *phone_numbers = record
print("1.2: ", phone_numbers)
*trailing, current = [10, 8, 7, 1, 9, 5, 10, 3]
print("sum(trailing)/current=", sum(trailing)/current)

print("######################## 1.3 保留最后 N 个元素-队列:deque 实现")
from collections import deque
q = deque(maxlen=4)
q.append(1)  # append in right side
print("Deque -- Queue: ",q)
q.append(2)  # append in right side
print("append the right: ",q)
q.appendleft(0)
print("append the left: ",q)
q.pop()  # pop out the right side
print("pop the right: ",q)
q.popleft()  # pop out the left side
print("pop the left: ",q)

print("######################## 1.4 查找最大或最小的 N 个元素:heapq 实现")
import heapq
nums = [1, 8, 2, 23, 7, -4, 18, 23, 42, 37, 2]
print(nums)
print("前3个最大值",heapq.nlargest(3, nums)) # Prints [42, 37, 23]
print("前3个最小值",heapq.nsmallest(3, nums)) # Prints [-4, 1, 2]

portfolio = [
    {'name': 'IBM', 'shares': 100, 'price': 91.1},
    {'name': 'AAPL', 'shares': 50, 'price': 543.22},
    {'name': 'FB', 'shares': 200, 'price': 21.09},
    {'name': 'HPQ', 'shares': 35, 'price': 31.75},
    {'name': 'YHOO', 'shares': 45, 'price': 16.35},
    {'name': 'ACME', 'shares': 75, 'price': 115.65}
]
cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
print("Sort in Price: ", cheap)
expensive = heapq.nlargest(3, portfolio, key=lambda s: s['shares'])
print("Sort in Shares: ", expensive)

print("\n  对集合数据进行堆排序然后放入列表中，heap[0]永远是最小的元素")
heap = list(nums)
heapq.heapify(heap)  # heapify list
print("original nums: ", nums)
print("堆 nums: ", heap)
print("heap pop out the smallest: ",heapq.heappop(heap))
print("堆：", heap)

print("######################## 1.4 查找最大或最小的 N 个元素:heapq 实现")
