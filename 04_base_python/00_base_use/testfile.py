import os


# 先删除文件尾的空行
def removefilenull(drectory, suffix):
    print('-------- Remove empty lines ----- start ---')
    for root, dirs, files in os.walk(drectory):
        for filename in files:
            if filename.endswith(suffix):
                with open(root + '/' + filename, 'r') as file:
                    lines = file.readlines()
                while lines and lines[-1].strip() == '':
                    lines.pop()
                with open(root + '/' + filename, 'w') as file:
                    file.writelines(lines)
                    print(os.path.join(root, filename))
    print('-------- Remove empty lines ----- end ----')
    print()


# 再向文件最后一行追加quit
def findendwithq(drectory, suffix):
    content = 'q'
    print('-------- add quit to .info file ----- start ---')
    for root, dirs, files in os.walk(drectory):
        for filename in files:
            if filename.endswith(suffix):
                with open(root + '/' + filename, 'r') as file:
                    filecontent = file.readlines()
                    endcontent = filecontent[-1].strip()
                    if endcontent.startswith(content) == False:
                        print(os.path.join(root, filename))
                        with open(root + '/' + filename, 'a') as file:
                            file.writelines('q,q,')
    print('-------- add quit to info file ------ end ----')
    print()


def dealwithfile():
    dirpath = 'file/basecase'
    suffix = '.info'
    removefilenull(dirpath, suffix)
    findendwithq(dirpath, suffix)
