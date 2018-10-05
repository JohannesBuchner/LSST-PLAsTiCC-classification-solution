paste training_set_metadata.csv <(sed 's/._fraclgf, ._fracrgf, *//g' training_set_std_features.txt | sed 's/^#//g'|sed 's/^/,/g') <(sed 's/#//g' training_set_slope_features.txt|sed 's/, *$//g')|sed 's,[\t ]*,,g'|gzip > training_set_all.csv.gz
awk -F, '{print $12}' training_set_metadata.csv > training_set_target.csv
