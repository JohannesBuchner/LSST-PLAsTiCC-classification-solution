
%_std_features.txt: %.csv
	python2.7 make_std_features.py $*

%_slope_features.txt: %.csv
	python2.7 make_slope_features.py $*
