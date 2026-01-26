#   --model MODEL         Model name: TopKPool, SAGPool, ASAP, EdgePool, MinCutPool, DiffPool, Graclus, GMT, Global-Attention, SortPool, Set2Set, GCN, GIN

python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Set2Set --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Set2Set --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Set2Set --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Set2Set --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Set2Set --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Set2Set --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Set2Set --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Set2Set --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Set2Set --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Set2Set --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Set2Set --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Set2Set --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Set2Set --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Set2Set --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Set2Set --tune --seed 46

# -------------------------------------------------------------------------------

python3 gnn_pooling_benchmark_cv.py --dataset DD --model Global-Attention --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Global-Attention --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Global-Attention --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Global-Attention --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Global-Attention --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Global-Attention --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Global-Attention --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Global-Attention --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Global-Attention --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Global-Attention --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Global-Attention --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Global-Attention --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Global-Attention --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Global-Attention --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Global-Attention --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Global-Attention --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Global-Attention --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Global-Attention --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Global-Attention --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Global-Attention --tune --seed 46

# -------------------------------------------------------------------------------

python3 gnn_pooling_benchmark_cv.py --dataset DD --model Graclus --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Graclus --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Graclus --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Graclus --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset DD --model Graclus --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Graclus --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Graclus --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Graclus --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Graclus --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model Graclus --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Graclus --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Graclus --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Graclus --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Graclus --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model Graclus --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Graclus --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Graclus --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Graclus --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Graclus --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model Graclus --tune --seed 46

# -------------------------------------------------------------------------------

python3 gnn_pooling_benchmark_cv.py --dataset DD --model DiffPool --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset DD --model DiffPool --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset DD --model DiffPool --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset DD --model DiffPool --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset DD --model DiffPool --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model DiffPool --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model DiffPool --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model DiffPool --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model DiffPool --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset PROTEINS --model DiffPool --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model DiffPool --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model DiffPool --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model DiffPool --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model DiffPool --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI1 --model DiffPool --tune --seed 46

python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model DiffPool --tune --seed 42
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model DiffPool --tune --seed 43
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model DiffPool --tune --seed 44
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model DiffPool --tune --seed 45
python3 gnn_pooling_benchmark_cv.py --dataset NCI109 --model DiffPool --tune --seed 46
