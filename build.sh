#make -j30 $(for i in $(seq 0 99); do echo parts/test_set_part${i}_std_features.txt; done)
renice +20 $$
make -j20 $(for i in $(seq 50 99); do echo parts/{test,training}_set_part${i}_{std,slope}_features.txt; done)
