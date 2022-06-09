def test(number, statement):
    a = 4
    if(statement):
        a = number*10

    return a


if __name__ == '__main__':
    a = test(10, test=True)
    print(a)