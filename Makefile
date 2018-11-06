
help: 
	@echo "POSSIBLE TARGETS:"
	@echo "chunks/test_set_chunk{0"$$(for i in $$(seq 1 99); do [ -e "chunks/test_set_chunk$${i}.csv" ] && echo -n ,$${i}; done)"}_{std,colorslope,SEDprob}_features.txt"
	@echo "{gal,gal-nodip,gal-dip,exgal}/{training_set,test_set}.csv.gz"

%_std_features.txt: %.csv
	python2.7 make_std_features.py $*

%_colorslope_features.txt: %.csv
	python2.7 make_2dslope_features.py $*

%_SEDprob_features.txt: %.csv training_set_SED_features.txt 
	SEDTRANSFORMER=1 python2.7 make_SED_features.py $*

%_slope_features.txt: %.csv
	python2.7 make_slope_features.py $*

test_set_all.csv.header: training_set_all.csv.gz
	zcat $^|head -n1|sed s/,target//g > $@

test_set_all_sorted.csv.gz: test_set_all.csv.header
	pushd chunks && bash ../combine_test.sh 

%_all.csv.gz: %_metadata.csv %_std_features.txt %_colorslope_features.txt %_SEDprob_features.txt
	paste -d, $^|sed 's,#,,g'|gzip > $@
%_gal.csv.gz: %_metadata.csv %_std_features.txt %_colorslope_features.txt %_SEDprob_features.txt
	paste -d, $^|sed 's,#,,g'|awk -F, '$$8==0 || NR==1'|gzip > $@
%_exgal.csv.gz: %_metadata.csv %_std_features.txt %_colorslope_features.txt %_SEDprob_features.txt
	paste -d, $^|sed 's,#,,g'|awk -F, '$$8!=0 || NR==1'|gzip > $@

test_set_gal.csv.gz: test_set_all_sorted.csv.gz
	< $^ gunzip | LC_ALL=C awk -F, '$$8==0 || NR==1' | gzip > $@
test_set_exgal.csv.gz: test_set_all_sorted.csv.gz
	< $^ gunzip | LC_ALL=C awk -F, '$$8!=0 || NR==1' | gzip > $@

# we split based on the ndips column
# but that column is 1 later in the training set (136) than in the test set
gal-nodip/training_set.csv.gz: training_set_gal.csv.gz
	mkdir -p gal-nodip
	zcat $^|awk -F, 'NR==1 || !($$136 > 0)' | gzip > $@
gal-dip/training_set.csv.gz: training_set_gal.csv.gz
	mkdir -p gal-dip
	zcat $^|awk -F, 'NR==1 || $$136 > 0' | gzip > $@
gal-nodip/test_set.csv.gz: test_set_gal.csv.gz
	mkdir -p gal-nodip
	zcat $^|awk -F, 'NR==1 || !($$135 > 0)' | gzip > $@
gal-dip/test_set.csv.gz: test_set_gal.csv.gz
	mkdir -p gal-dip
	zcat $^|awk -F, 'NR==1 || $$135 > 0' | gzip > $@
exgal/%.csv.gz: %_exgal.csv.gz
	mkdir -p exgal
	ln -s ../$^ $@
gal/%.csv.gz: %_gal.csv.gz
	mkdir -p gal
	ln -s ../$^ $@

rexgal/training_set.csv.gz: resampled_training_set_exgal.csv.gz
	mkdir -p rexgal
	ln -s ../$^ $@
rgal/training_set.csv.gz: resampled_training_set_gal.csv.gz
	mkdir -p rgal
	ln -s ../$^ $@
rexgal/test_set.csv.gz: test_set_exgal.csv.gz
	mkdir -p rexgal
	ln -s ../$^ $@
rgal/test_set.csv.gz: test_set_gal.csv.gz
	mkdir -p rgal
	ln -s ../$^ $@


.PRECIOUS: training_set_exgal.csv.gz training_set_gal.csv.gz test_set_all_sorted.csv.gz

