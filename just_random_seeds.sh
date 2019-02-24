mkdir $4
$5 ./ner.py --verbose $1 $2 cnn $3 $4

echo Finished running the CNN version starting LSTM version

$5 ./ner.py --verbose $1 $2 lstm $3 $4
rm -r $4
echo Finished