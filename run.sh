#   --model MODEL         Model name: TopKPool, SAGPool, ASAP, EdgePool, MinCutPool, DiffPool, Graclus, GMT, Global-Attention, SortPool, Set2Set, GCN, GIN

python3 gnn_pooling_benchmark_cv.py --dataset DD --model MinCutPool --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset DD --model MinCutPool --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset DD --model MinCutPool --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset DD --model MinCutPool --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset DD --model MinCutPool --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model MinCutPool --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model MinCutPool --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model MinCutPool --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model MinCutPool --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model MinCutPool --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model MinCutPool --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model MinCutPool --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model MinCutPool --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model MinCutPool --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model MinCutPool --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model MinCutPool --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model MinCutPool --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model MinCutPool --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model MinCutPool --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model MinCutPool --tune --seed 46
