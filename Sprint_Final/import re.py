import re
from itertools import filterfalse

def getStr(tup):
    s = ""
    for i in range(6):
        temp = ""
        for j in range(6):
            temp = temp + tup[j][i]
        s = s + temp  
    #print(s)
    return s

def combinaisonsRow():
    array = []
    for v1 in "ABCD":
        for v2 in "ABCD":
            for v3 in "ABCD":
                for v4 in "ABCD":
                    for v5 in "ABCD":
                        for v6 in "ABCD":
                            array.append(v1+v2+v3+v4+v5+v6)
    return array

def combinaisonsMatrix(fr):
    matrix = []
    c = 0
    for v1 in fr:
        c = c+1
        print(str(c)+"/"+str(len(fr)))
        for v2 in fr:
            for v3 in fr:
                for v4 in fr:
                    for v5 in fr:
                        for v6 in fr:
                            matrix.append(v1)
                            matrix.append(v2)
                            matrix.append(v3)
                            matrix.append(v4)
                            matrix.append(v5)
                            matrix.append(v6)
    return matrix

print("creating list")
r = combinaisonsRow()
print("filtering")
fr = list(filterfalse(lambda x: re.search("AAA|BBB|CCC|DDD",x), r))
print("creating matrixes")
m = combinaisonsMatrix(fr)
print(m[0])
print("filtering")


