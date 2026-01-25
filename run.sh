#   --model MODEL         Model name: TopKPool, SAGPool, ASAP, EdgePool, MinCutPool, DiffPool, Graclus, GMT, Global-Attention, SortPool, Set2Set, GCN, GIN

python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model SAGPool --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model SAGPool --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model DiffPool --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model DiffPool --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model DiffPool --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model Graclus --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model Graclus --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset REDDIT-BINARY --model Graclus --tune --seed 44


