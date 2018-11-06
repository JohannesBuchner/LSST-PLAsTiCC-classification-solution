f=$1
# split this file into 100 parts based on 2nd and 3rd digit (of object_id)

#for i in $(seq 0 99)
#do 
#i=$(printf %02d $i)
#echo $i
#LC_ALL=C grep "^.$i" $f.csv > parts/${f}_part${i}.csv
#done

export LC_ALL=C 

#awk -F, '{print $0 > "parts/'$f'_part"($1 % 100)".csv"}' "${f}.csv"
awk -F, '{print $0 > "chunks/'$f'_chunk"(int($1 / 4000000))".csv"}' "${f}.csv"

# insert header into each file
header=$(head -n1 ${f}.csv)
ls chunks/${f}_chunk*.csv|xargs --max-procs=30 --max-args=1 sed -i '1s/^/'"$header"'\n'/ $part
# now we have it twice in the first file -- remove it again
sed -i '1d' chunks/${f}_chunk0.csv

