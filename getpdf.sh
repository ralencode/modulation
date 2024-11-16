#!/bin/bash

for i in $(ls); do
    magick "$i/*.svg" "$i.pdf"
done
