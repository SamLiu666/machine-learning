import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
from git import Repo
update_str = input("Enter Your Record: ")

dirfile = os.path.abspath('')
print("File Name: ",dirfile)

repo = Repo(dirfile)

g = repo.git
print(g.add)
g.add("-A")
g.commit(update_str)
g.push
print("Successful Push!")