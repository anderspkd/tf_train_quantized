#!/bin/bash

pb_model=$1

function print_usage_and_exit {
    echo "usage: $0 [frozen_model.pb] [input_arrays] [output_arrays]"
    exit 0
}

if ! [ -f "$pb_model" ]; then
    print_usage_and_exit
fi

input_arrays=$2
output_arrays=$3

if ([ -z "$input_arrays" ] || [ -z "$output_arrays" ]); then
    print_usage_and_exit
fi

tflite_model="$(basename $pb_model .pb).tflite"
echo "converting \"$pb_model\" to \"$tflite_model\""

tflite_convert --graph_def_file=$pb_model \
	       --output_file=$tflite_model \
	       --inference_type=QUANTIZED_UINT8 \
	       --input_arrays=$input_arrays \
	       --output_arrays=$output_arrays \
	       --mean_values=0 \
	       --std_dev_values=255

echo "done!"
