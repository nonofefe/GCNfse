declare -a array=()
declare -a array=("uniform" "bias" "struct")

for ((i = 0; i < ${#array[@]}; i++)) {
    echo "type = ${array[i]}" >> log.txt
    a=0
    while [ $a -lt 10 ]
    do
        b=`echo "scale=1; $a / 10 " | bc`
        echo $b >> log.txt
        python run_node_cls.py --dataset cora --rate $b --epoch 30 --patience 10 --type ${array[i]}
        a=`expr $a + 1`
    done
    t=`expr $t + 1`
}



# while [ $a -lt 5 ]
# do
#     a=`expr $a + 1`
#     b=`expr 4 \* $a`
#     python3 train.py --n_train $b --patience 10 --epoch 200 --alpha 0.6 --times 300 --dataset cora
# done
# echo "" >> log.txt