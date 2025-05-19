# deeplearn 
1、查看 conda虚拟环境：conda env list

2、创建 conda虚拟环境：conda create -n pytorch-m2 python=3.10

3、激活虚拟环境：activate pytorch(conda环境)

4、查看安装源：conda config --show-sorces

5、安装清华源：conda config --add channels https://mirros.tuna.tsinghua.edu.cn/anconda/pkgs/free/

6、移除conda 默认安装源：conda config --remove channels defaults

7、Conda环境中安装OpenCV：conda install -c conda-forge opencv