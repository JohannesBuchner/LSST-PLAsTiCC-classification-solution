export TRAINING_FILE=training_set.csv.gz

FIND_FEATURE_SUBSET=1 nohup python ../train_randomforest.py > nohup.out.rf_features
sort -k2,2 important_columns* > important_columns.txt

export PREDICT_FILE=test_set.csv.gz 
nohup python ../train_randomforest.py > nohup.out.rf &

SIMPLIFY=1 TRANSFORM=MM nohup python ../train_knn.py > nohup.out.knn-mm &
SIMPLIFY=1 TRANSFORM=QTN nohup python ../train_knn.py > nohup.out.knn-qtn &

K=20 SIMPLIFY=1 NPCACOMP=30 PROB_FLATNESS=1 FLATPRIOR_STRENGTH=0.1 OUTLIERS_STRENGTH=1.0 TRANSFORM=MM nohup python ../train_kmeans.py > nohup.out.kmeans &
IMPUTE=0 SIMPLIFY=1 nohup python ../train_naive_bayes.py > nohup.out.nb &

exit 0
wait
nohup python ../hyperpredictor.py RandomForest400 SIMPLEMM-PCA40-SVC-default SIMPLEQTN-PCA40-MLP4 SIMPLEQTN-PCA10-MLP10 > nohup.out.hyper

