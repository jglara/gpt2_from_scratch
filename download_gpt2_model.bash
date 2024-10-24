#!/bin/bash

destination=./gpt2_data
model_size=124M
URL=https://openaipublic.blob.core.windows.net/gpt-2/models/${model_size}
files="checkpoint encoder.json hparams.json model.ckpt.data-00000-of-00001 model.ckpt.index model.ckpt.meta vocab.bpe"
mkdir ${destination}
for f in $files; do curl ${URL}/${f} --output ./${destination}/${f}; done
