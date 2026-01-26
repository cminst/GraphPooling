#   --model MODEL         Model name: TopKPool, SAGPool, ASAP, EdgePool, MinCutPool, DiffPool, Graclus, GMT, Global-Attention, SortPool, Set2Set, GCN, GIN

python3 gnn_pooling_benchmark_cv.py --dataset DD --model Set2Set --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Set2Set --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Set2Set --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Set2Set --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Set2Set --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model GCN --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model GCN --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model GCN --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model GCN --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model GCN --tune --seed 46
