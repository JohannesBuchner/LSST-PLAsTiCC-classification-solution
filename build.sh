#make -j30 $(for i in $(seq 0 99); do echo parts/test_set_part${i}_std_features.txt; done)
renice +20 $$
#make -k -j20 $(for i in $(seq 0 99); do echo parts/{test,training}_set_part${i}_{std,slope}_features.txt; done)
#make -k -j20 $(for i in $(seq 0 99); do echo parts/{test,training}_set_part${i}_{std,slope,colorslope}_features.txt; done)
#make -k -j8 $(for i in $(seq 0 99); do echo parts/test_set_part${i}_{std,slope,colorslope}_features.txt; done)
make -k -j20 $(for i in $(seq 0 99); do [ -e "chunks/test_set_chunk${i}.csv" ] && echo chunks/test_set_chunk${i}_{std,colorslope}_features.txt; done)
