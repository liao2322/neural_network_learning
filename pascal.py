# Parscal帕斯卡三角形算法
# 2018-04-26

def pascal(n):
    if n==1:
        return [1]
    else:
        line=[1]
        previous_line=pascal(n-1)
        for i in range(len(previous_line)-1):
            line.append(previous_line[i]+previous_line[i+1])
        line.append(1)
    return line
n=int(input("Enter the number of lines:"))
for i in range(1,n+1):
    print(pascal(i))

