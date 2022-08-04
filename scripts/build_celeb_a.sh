#! /bin/sh

gdown https://drive.google.com/uc?id=1tCj_CKUgjtThk1fiL0wP0WiuYxk4dvjT -O /tmp/celeb.zip
mkdir -p ~/tensorflow_datasets/celeb_a
unzip /tmp/celeb.zip -d ~/tensorflow_datasets/celeb_a
