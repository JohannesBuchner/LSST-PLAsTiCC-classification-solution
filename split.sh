# split this file into parts based on object_id

export LC_ALL=C 

mkdir -p chunks

function splitfile {
f=$1
awk -F, '{print $0 > "chunks/'$f'_chunk"(int($1 / 4000000))".csv"}' "${f}.csv"

# insert header into each file
header=$(head -n1 ${f}.csv)
ls chunks/${f}_chunk*.csv|xargs --max-procs=30 --max-args=1 sed -i '1s/^/'"$header"'\n'/ $part
# now we have it twice in the first file -- remove it again
sed -i '1d' chunks/${f}_chunk0.csv
}

splitfile test_set &
splitfile test_set_metadata

for i in $(seq 0 100)
do
	[ -e chunks/test_set_metadata_chunk${i}.csv ] && mv chunks/test_set_metadata_chunk${i}.csv chunks/test_set_chunk${i}_metadata.csv
done

wait

