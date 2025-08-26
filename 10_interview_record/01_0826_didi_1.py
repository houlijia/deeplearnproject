
str = ['flower', 'flow', 'flight']


def funct(strl):
    res = []
    if len(strl) > 0:
        for i in zip(*strl):
            if len(set(i)) == 1:
                res.append(i[0])
            else:
                break
        return ''.join(res)
    else:
        print("请确认输入是否合理")


a = funct(str)
print(a)
