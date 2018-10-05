{
export LC_ALL=C
cat ../test_set_all.csv.header
paste <(cat test_set_part*metadata.csv) <(cat test_set_part*std_features.txt | sed 's/._fraclgf, ._fracrgf, *//g' | sed 's/^#//g'|sed 's/^/,/g') <(cat test_set_part*_slope_features.txt|sed 's/#//g'|sed 's/, *$//g')|sed 's,[\t ]*,,g'|LC_ALL=C grep -v "^object_id" | LC_ALL=C sort -S 10M -T . --parallel=10 -t , -k1,1 -n; 
} | gzip > ../test_set_all_sorted.csv.gz

#{ cat test_set_all.csv.header; zcat test_set_all.csv.gz |LC_ALL=C grep -v "^object_id" | LC_ALL=C sort -S 10M -T . --parallel=10 -t , -k1,1 -n; } | gzip > test_set_all_sorted.csv.gz
