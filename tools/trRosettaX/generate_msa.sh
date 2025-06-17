echo $1 $2

export PATH="/trxkeipa/bin/:$PATH"

python generate_msa.py -i $1 -o $2 -hhdb /uniclust30_2018_08/uniclust30_2018_08 -e_value 0.001 -cpu 30
