# Function 과 class의 차이를 알아보자

'''Function'''
result = 0

def add(num) : 
    global result
    result += num
    return result

print(add(3))   # 3
print(add(2))   # 5


# 독립적인 덧셈기를 만들고 싶다면 함수를 두 개 만들어야 한다.

result1 = 0
result2 = 0



def add1(num) : 
    global result1
    result1 += num
    return result1

    
def add2(num) : 
    global result2
    result2 += num
    return result2


print(add1(4))  # 4
print(add1(8))  # 12

print(add2(5))  # 5
print(add2(10)) # 15


'''Class'''
# 하나의 클래스로 독립적인 객체를 만들어 연산할 수 있다.

class Calculator:
    def __init__(self): # 초기화 함수, 생성자
        self.result = 0

    def add(self, num):
        self.result += num
        return self.result

# cal1 객체 만들기
cal1 = Calculator()

# cal2 객체 만들기
cal2 = Calculator()

print(cal1.add(3))  # 3
print(cal1.add(6))  # 9

print(cal2.add(5))  # 5
print(cal2.add(10)) # 15