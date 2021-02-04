 horovodrun -np 2 --cycle-time-ms 0.1 python train.py -net resnet152 -b 128 -lr 0.1 --compress --compressor efsignsgd --fusion-num 1 | tee logs2/efsignsgd_resnet152_y_1
 horovodrun -np 2 --cycle-time-ms 0.1 python train.py -net resnet152 -b 128 -lr 0.1 --compress --compressor efsignsgd --fusion-num 2 | tee logs2/efsignsgd_resnet152_y_2
 horovodrun -np 2 python train.py -net resnet152 -b 128 -lr 0.1 --compress --compressor efsignsgd --fusion-num 0 | tee logs2/efsignsgd_resnet152_lw
