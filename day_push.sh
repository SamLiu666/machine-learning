#!/bin/bash
git status
git add -A
echo -n " Please Enter Your Push Record ->"
read record
echo -n
git commit -m $record
git push origin master

echo "done"
