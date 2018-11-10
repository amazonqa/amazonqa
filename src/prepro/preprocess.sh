#!/bin/bash

categories=("Automotive" "Baby" "Beauty" "Cell Phones and Accessories" "Clothing Shoes and Jewelry" "Electronics" "Grocery and Gourmet Food" "Health and Personal Care" "Home and Kitchen" "Musical Instruments" "Office Products" "Patio Lawn and Garden" "Pet Supplies" "Sports and Outdoors" "Tools and Home Improvement" "Toys and Games" "Video Games")
categories=("Baby")

data_dir="../../data"

mkdir -p $data_dir

for i in "${categories[@]}"; do
    echo "$i"
    python3 preprocess_data.py --download 0 --data_dir $data_dir --category $i &
done

wait

qar_all_categories="$data_dir/qar_all_categories.jsonl"

for i in "${categories[@]}"; do
    echo "$i"
    file="$data_dir/qar_$i.jsonl"
    cat $file > $qar_all_categories
done

gshuf $qar_all_categories -o $qar_all_categories

num_lines=($(wc -l $qar_all_categories))
parts=('train' 'val' 'test')

part=0
percentSum=0
currentLine=1
for percent in 80 10 10; do
        ((percentSum += percent))
        ((nextLine = num_lines * percentSum / 100))
        if ((nextLine < currentLine)); then
                printf "" # create empty file
        else
                sed -n "$currentLine,$nextLine"p "$qar_all_categories"
        fi > "$data_dir/${parts[$part]}-qar_all_categories.jsonl"
        ((currentLine = nextLine + 1))
        ((part++))
done

