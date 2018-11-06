f=$1
if [ -z "$f" ]
then
	echo "SYNAPSIS: $0 prediction_out_FILENAME.csv.gz"
else
	sort -m -k1,1n -t, <(zcat exgal/$f) <(zcat gal/$f|grep -v "^object_id")|gzip > $f
	#sort -m -k1,1n -t, <(zcat exgal/$f) <(zcat gal-{dip,nodip}/$f|grep -v "^object_id")|gzip > $f
fi
