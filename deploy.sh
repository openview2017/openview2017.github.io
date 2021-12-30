#!/bin/bash

echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# Build the project.
hugo -t even # if using a theme, replace with `hugo -t <YOURTHEME>`

# Go To Public folder
cd public

git init -b source
git remote add origin https://github.com/openview2017/openview2017.github.io.git
# Add changes to git.
git add .

# Commit changes.
msg="rebuilding site `date`"
if [ $# -eq 1 ]
  then msg="$1"
fi
git commit -m "$msg"

# Push source and build repos.
git push origin source -f

# Come Back up to the Project Root
cd ..

rm -rf public