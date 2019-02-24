x=$1
y=$2
for z in `seq $x $y`; do
echo Starting Job task $z 

mkdir $3/$z
$6 ./ner.py --verbose $3/$z $4 cnn 5 $5/$z

echo Finished running the CNN version starting LSTM version

$6 ./ner.py --verbose $3/$z $4 lstm 5 $5/$z
rm -r $5/$z

echo Finished Job task $z 
done