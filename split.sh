f=$1
# split this file into parts based on object_id

export LC_ALL=C 

mkdir -p chunks
awk -F, '{print $0 > "chunks/'$f'_chunk"(int($1 / 4000000))".csv"}' "${f}.csv"

# insert header into each file
header=$(head -n1 ${f}.csv)
ls chunks/${f}_chunk*.csv|xargs --max-procs=30 --max-args=1 sed -i '1s/^/'"$header"'\n'/ $part
# now we have it twice in the first file -- remove it again
sed -i '1d' chunks/${f}_chunk0.csv

