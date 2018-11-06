#{
#export LC_ALL=C
#cat ../test_set_all.csv.header
#paste <(cat test_set_part*metadata.csv) <(cat test_set_part*std_features.txt | sed 's/._fraclgf, ._fracrgf, *//g' | sed 's/^#//g'|sed 's/^/,/g') <(cat test_set_part*_slope_features.txt | 
#sed 's/#//g'|sed 's/, *$//g')|sed 's,[\t ][\t ]*,,g'|grep -v "^object_id" | sort -S 10M -T . --parallel=10 -t , -k1,1 -n; 
#} | gzip > ../test_set_all_sorted.csv.gz

#test -e ../test_set_all_sorted.csv.gz || {
{
export LC_ALL=C
cat ../test_set_all.csv.header
IDs=$(seq 0 32)
a=$(for i in $IDs; do echo test_set_chunk${i}_metadata.csv; done)
b=$(for i in $IDs; do echo test_set_chunk${i}_std_features.txt; done)
c=$(for i in $IDs; do echo test_set_chunk${i}_colorslope_features.txt; done)
c=$(for i in $IDs; do echo test_set_chunk${i}_SEDprob_features.txt; done)
paste -d, <(cat $a) <(cat $b) <(cat $c) | sed 's,#,,g'|grep -v "^object_id";
} | gzip > ../test_set_all_sorted.csv.gz


#test -e ../test_set_gal2.csv.gz   || < ../test_set_all2_sorted.csv.gz gunzip | LC_ALL=C awk -F, '$8==0 || NR==1' | gzip > ../test_set_gal2.csv.gz &
#test -e ../test_set_exgal2.csv.gz || < ../test_set_all2_sorted.csv.gz gunzip | LC_ALL=C awk -F, '$8!=0 || NR==1' | gzip > ../test_set_exgal2.csv.gz &


