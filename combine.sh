#paste training_set_metadata.csv <(sed 's/._fraclgf, ._fracrgf, *//g' training_set_std_features.txt | sed 's/^#//g'|sed 's/^/,/g') <(sed 's/#//g' training_set_slope_features.txt|sed 's/, *$//g')|sed 's,[\t ]*,,g'|gzip > training_set_all.csv.gz
paste -d, training_set_metadata.csv training_set_std_features.txt training_set_colorslope_features.txt|sed 's,#,,g'|gzip > training_set_all2.csv.gz
awk -F, '{print $12}' training_set_metadata.csv > training_set_target.csv

paste resampled_training_set_metadata.csv <(sed 's/._fraclgf, ._fracrgf, *//g' resampled_training_set_std_features.txt | sed 's/^#//g'|sed 's/^/,/g') <(sed 's/#//g' resampled_training_set_slope_features.txt|sed 's/, *$//g')|sed 's,[\t ]*,,g'|gzip > resampled_training_set_all.csv.gz
awk -F, '{print $12}' resampled_training_set_metadata.csv > resampled_training_set_target.csv

