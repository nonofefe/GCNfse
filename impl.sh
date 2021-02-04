a=0
while [ $a -lt 10 ]
do
    python run_node_cls.py --rate 0.8
done


# while [ $a -lt 5 ]
# do
#     a=`expr $a + 1`
#     b=`expr 4 \* $a`
#     python3 train.py --n_train $b --patience 10 --epoch 200 --alpha 0.6 --times 300 --dataset cora
# done
# echo "" >> log.txt