A2 = [5,2,2]
A3 = [-2,-3,-2,-2]

def choose(a1,a2):
    l = 0
    for i in range(len(a1)):
        for j in a1:
            if -j in a2:
                a1.remove(j)
                a2.remove(-j)
                l +=1
    return a1,a2,l
# print(choose(A2,A3))
def choose2(a1,a2):
    l = 0
    # for i in range(len(a1)):
    for j in a1:
        if -j in a2:
            a1.remove(j)
            a2.remove(-j)
            l +=1
    return a1,a2,l
print(choose(A2,A3))
# a = [1,1,1,2]
# a.remove(1)
# print(a)
# a.remove(1)
# print(a)