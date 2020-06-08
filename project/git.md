```shell
git add .  # 全部添加
git commit -m "改动的地方"
git push -u origin master # 提交

# 激活虚拟环境
activate env_name
deactivate env_name

conda list：查看安装了哪些包。
conda install package_name(包名)：安装包
conda env list 或 conda info -e：查看当前存在哪些虚拟环境
conda update conda：检查更新当前conda

git config --global https.proxy http://127.0.0.1:1080

git config --global https.proxy https://127.0.0.1:1080

git config --global --unset http.proxy

git config --global --unset https.proxy
```



```git
$ git init
Initialized empty Git repository in D:/git-project/test/.git/

liu@LAPTOP-KNM96SDR MINGW64 /d/git-project/test (master)
$ touch test.py

liu@LAPTOP-KNM96SDR MINGW64 /d/git-project/test (master)
$ touch readme.md

liu@LAPTOP-KNM96SDR MINGW64 /d/git-project/test (master)
$ git status

liu@LAPTOP-KNM96SDR MINGW64 /d/git-project/test (master)
$ git add .


liu@LAPTOP-KNM96SDR MINGW64 /d/git-project/test (master)
$ git branch hello

liu@LAPTOP-KNM96SDR MINGW64 /d/git-project/test (master)
$ git checkout hello
Switched to branch 'hello'


```

$ git add .

$ git commit -m "change"

[git 操作](https://blog.csdn.net/jtracydy/article/details/70402663)

 **提交**

```shell
git remote add origin https://github.com/SamLiu666/java.git

git push -u origin master

```

**分支**

```shell
(base) lxp@lxp-virtual-machine:~/apps/AI/pytorch$ git branch ubuntu
(base) lxp@lxp-virtual-machine:~/apps/AI/pytorch$ git branch
* master
  ubuntu

(base) lxp@lxp-virtual-machine:~/apps/AI/pytorch$ git checkout ubuntu
Already on 'ubuntu'

```

**代理**

```shell
git config http.proxy http://127.0.0.1:2334
```

**配置仓库**

```shell
第三步：输入git add .     
这个是将项目上所有的文件添加到仓库中的意思，如果想添加某个特定的文件，只需把.换成这个特定的文件名即可。

第四步输入git commit -m "first commit"
表示你对这次提交的注释，双引号里面的内容可以根据个人的需要改。

第五步输入git remote add origin 自己的仓库url地址

```

https://blog.csdn.net/weixin_40096730/article/details/87872228?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase