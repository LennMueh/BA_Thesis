#!/usr/bin/zsh
source ~/.virtualenvs/BA_Thesis/bin/activate
#repeat 1 {python3 main.py "cnn1_1epoch_full"}
repeat 1 {python3 main.py "cnn1_1epoch_half"}
repeat 1 {python3 main.py "cnn1_1epoch_quarter"}
repeat 1 {python3 main.py "cnn1_6epoch_full"}
repeat 1 {python3 main.py "cnn1_6epoch_half"}
repeat 1 {python3 main.py "cnn1_6epoch_quarter"}
#repeat 1 {python3 main.py "cnn2_1epoch_full"}
#repeat 1 {python3 main.py "cnn2_1epoch_half"}
#repeat 1 {python3 main.py "cnn2_1epoch_quarter"}
#repeat 1 {python3 main.py "cnn2_6epoch_full"}
#repeat 1 {python3 main.py "cnn2_6epoch_half"}
#repeat 1 {python3 main.py "cnn2_6epoch_quarter"}
#repeat 1 {python3 main.py "dnn1_1epoch_full"}
#repeat 1 {python3 main.py "dnn1_1epoch_half"}
#repeat 1 {python3 main.py "dnn1_1epoch_quarter"}
#repeat 1 {python3 main.py "dnn1_6epoch_full"}
#repeat 1 {python3 main.py "dnn1_6epoch_half"}
#repeat 1 {python3 main.py "dnn1_6epoch_quarter"}
