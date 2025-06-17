echo $1 $2

export PATH="/trxkeipa/bin/:$PATH"

python predict.py -i $1 -o $2 -mdir model_res2net_202108 -cpu 30
