#!/bin/bash
git status
git add .
echo -n "Please Enter Your Push Record ->"
read record
git commit -m $record
git push origin master

echo "done"