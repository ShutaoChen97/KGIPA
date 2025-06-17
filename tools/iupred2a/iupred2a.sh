echo $1 $2 $3

# python3 iupred2a.py $1 long
python3 /home/www/KGIPA/webserver/tools/iupred2a/iupred2a.py $1 $2 >> $3 2>&1 &
