#!/usr/bin/zsh
source ~/.virtualenvs/BA_Thesis/bin/activate
repeat 1 {python3 main_vgg.py "tiny_vgg_1epoch_full"}
repeat 1 {python3 main_vgg.py "tiny_vgg_1epoch_half"}
repeat 1 {python3 main_vgg.py "tiny_vgg_1epoch_quarter"}
repeat 1 {python3 main_vgg.py "tiny_vgg_6epoch_full"}
repeat 1 {python3 main_vgg.py "tiny_vgg_6epoch_half"}
repeat 1 {python3 main_vgg.py "tiny_vgg_6epoch_quarter"}
# tarantula