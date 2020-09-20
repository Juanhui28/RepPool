##NCI1
python main.py -data NCI1 -aggre_mode local -struct_mode local -sele_method 1  -lamda1 0.1  -num_layers 2 -gcn_layers 3  -percent 0.3 -lr 0.002 -fold 1

##NCI109
python main.py -data NCI109 -aggre_mode local -struct_mode local -sele_method 1  -lamda1 0.1  -num_layers 2 -gcn_layers 3  -percent 0.5 -lr 0.001 -fold 1

##PROTEINS
python main.py -data PROTEINS -aggre_mode local -struct_mode local -sele_method 1  -lamda1 0.1  -num_layers 2 -gcn_layers 3  -percent 0.3 -lr 0.005 -fold 1

##DD
python main.py -data DD -aggre_mode local -struct_mode local -sele_method 1  -lamda1 0.1  -num_layers 2 -gcn_layers 3  -percent 0.1 -lr 0.0005 -fold 1

##MUTAG
python main.py -data MUTAG -aggre_mode local -struct_mode local -sele_method 1  -lamda1 0.1  -num_layers 2 -gcn_layers 3  -percent 0.4 -lr 0.002 -fold 1

##Synthie
python main.py -data Synthie -aggre_mode local -struct_mode local -sele_method 1  -lamda1 0.1  -num_layers 2 -gcn_layers 3  -percent 0.3 -lr 0.01 -fold 1

##PTC
python main.py -data PTC -aggre_mode local -struct_mode local -sele_method 1  -lamda1 0.1  -num_layers 2 -gcn_layers 3  -percent 0.2 -lr 0.007 -fold 1

##IMDB-BINARY
python main.py -data IMDBBINARY -aggre_mode local -struct_mode local -sele_method 1  -lamda1 0.1  -num_layers 2 -gcn_layers 1  -percent 0.08 -lr 0.002 -fold 1
