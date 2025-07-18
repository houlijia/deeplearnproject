from pathlib import Path

import requests
import re
import os

# 创建文件夹
filename = os.path.join(os.path.dirname(__file__), "video")
if not os.path.exists(filename):
    os.mkdir(filename)

# 网页链接
url = 'https://music.163.com/discover/toplist?id=3778678'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36'
}

# 发送请求获取歌曲信息
response = requests.get(url=url, headers=headers)
result = re.findall('<li><a href="/song\?id=(\d+)">(.*?)</a></li>', response.text)

# 下载歌曲
for music, title in result:
    music_url = f'http://music.163.com/song/media/outer/url?id={music}.mp3'
    music_content = requests.get(url=music_url, headers=headers).content

    with open(filename + "//" + title + '.mp3', mode='wb') as f:
        f.write(music_content)
        print(title)
