echo $1 $2

export PATH="/home/www/KGIPA/trxkeipa/bin/:$PATH"

python /home/www/KGIPA/webserver/tools/trRosettaX/predict.py -i $1 -o $2 -mdir /mnt/home/webserver/model_res2net_202108 -cpu 30
