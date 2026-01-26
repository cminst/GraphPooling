#   --model MODEL         Model name: TopKPool, SAGPool, ASAP, EdgePool, MinCutPool, DiffPool, Graclus, GMT, Global-Attention, SortPool, Set2Set, GCN, GIN

python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model GMT --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model GMT --tune --seed 46
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model GMT --tune --seed 47

python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model Global-Attention --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model Global-Attention --tune --seed 46
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model Global-Attention --tune --seed 47
