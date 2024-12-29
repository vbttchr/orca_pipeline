#!/bin/bash

## get all the dirs 

echo "This is very specific and shoudl only be used for the andresRequest dir"

dirs=$(find . -type d -iname "pipeline")
current_dir=$(pwd)
for dir in $dirs
do
if [[ "$dir" =~ [Nn][Oo]_[Ll][Ii] ]]; then
    echo "Yes No_LI $dir"
    cd $dir
    ssubpipeline -c -1 -m 1 -s thf
    cd $current_dir
else
    echo "No No_LI $dir"
    cd $dir
    ssubpipeline -c 0 -m 1 -s thf
    cd $current_dir
fi
done