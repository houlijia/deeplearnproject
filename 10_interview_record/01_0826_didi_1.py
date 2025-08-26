
str = ['flower', 'flow', 'flight']


def funct(strl):
    res = []
    if len(strl) > 0:
        aa = zip(*strl)
        for i in aa:
            a011 = set(i)
            a012 = len(a011)
            if len(a011) == 1:
                res.append(i[0])
            else:
                break
        return ''.join(res)
    else:
        print("请确认输入是否合理")


a = funct(str)
print(a)
