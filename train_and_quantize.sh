#!/bin/bash

modelname=$1

function msg {
    echo "-------------------------------"
    echo $1
    echo "-------------------------------"
}

if [ -z "$1" ]; then
    echo "empty modelname"
    exit 0
fi

if ! ([ -e "train_mnist.py" ] && [ -e "checkpoint2pb.py" ] && [ -e "pb2tflite.sh" ]); then
    echo "missing executables"
    exit 1
fi

msg "training ${modelname}"

./train_mnist.py -m "${modelname}"

msg "freezing trained model"

r=$(./checkpoint2pb.py "${modelname}" chkpt/checkpoints -g)
# figure out the input/output names
inp=$(echo $r | cut -d' ' -f2)
out=$(echo $r | cut -d' ' -f3)

msg "writing tflite model"

./pb2tflite.sh "${modelname}.pb" "${inp}" "${out}"
