for s in 1 2 3 4 5; do
    python3 ShortestPathExpBaseline.py --model_name SPO --grid_size 15 --seed $s
    python3 ShortestPathExpBaseline.py --model_name SCE --grid_size 15 --seed $s
    python3 ShortestPathExpBaseline.py --model_name PFL --grid_size 15 --seed $s
    python3 ShortestPathExpBaseline.py --model_name PFY --grid_size 15 --seed $s
    python3 ShortestPathExpBaseline.py --model_name CAVE --grid_size 15 --seed $s
    python3 ShortestPathExpBaseline.py --model_name CVX-Squared --grid_size 15 --seed $s
    python3 ShortestPathExpBaseline.py --model_name CVX-Regret --grid_size 15 --seed $s
    python3 ShortestPathExpBaseline.py --model_name CVX-SPO --grid_size 15 --seed $s
    python3 ShortestPathExpBaseline.py --model_name CVX-SCE --grid_size 15 --seed $s
done
for s in 1 2 3 4 5; do
    python3 KnapsackExpBaseline.py --model_name SPO  --num_items 400 --seed $s
    python3 KnapsackExpBaseline.py --model_name SCE  --num_items 400 --seed $s
    python3 KnapsackExpBaseline.py --model_name PFL  --num_items 400 --seed $s
    python3 KnapsackExpBaseline.py --model_name PFY  --num_items 400 --seed $s
    python3 KnapsackExpBaseline.py --model_name CAVE  --num_items 400 --seed $s
    python3 KnapsackExpBaseline.py --model_name CVX-Squared  --num_items 400 --seed $s
    python3 KnapsackExpBaseline.py --model_name CVX-Regret  --num_items 400 --seed $s
    python3 KnapsackExpBaseline.py --model_name CVX-SPO  --num_items 400 --seed $s
    python3 KnapsackExpBaseline.py --model_name CVX-SCE  --num_items 400 --seed $s
done
for s in  1 2 3 4 5; do
    python3 FacilityLocationExpBaseline.py --model_name SPO  --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpBaseline.py --model_name SCE  --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpBaseline.py --model_name PFL  --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpBaseline.py --model_name PFY  --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpBaseline.py --model_name CAVE  --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpBaseline.py --model_name CVX-Squared  --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpBaseline.py --model_name CVX-Regret  --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpBaseline.py --model_name CVX-SPO  --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpBaseline.py --model_name CVX-SCE  --num_customers 200 --num_facilities 10 --seed $s
done
for s in 1 2 3 4 5; do
    python3 ShortestPathExpDYS.py --model_name DYS-Squared --grid_size 15 --seed $s
    python3 ShortestPathExpDYS.py --model_name DYS-Regret --grid_size 15 --seed $s
    python3 ShortestPathExpDYS.py --model_name DYS-SPO --grid_size 15 --seed $s
    python3 ShortestPathExpDYS.py --model_name DYS-SCE --grid_size 15 --seed $s
done
for s in  1 2 3 4 5; do
    python3 FacilityLocationExpDYS.py --model_name DYS-Squared   --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpDYS.py --model_name DYS-Regret   --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpDYS.py --model_name DYS-SPO   --num_customers 200 --num_facilities 10 --seed $s
    python3 FacilityLocationExpDYS.py --model_name DYS-SCE   --num_customers 200 --num_facilities 10 --seed $s
done
for s in 1 2 3 4 5; do
    python3 KnapsackExpDYS.py --model_name DYS-Squared --num_items 400 --seed $s
    python3 KnapsackExpDYS.py --model_name DYS-Regret --num_items 400 --seed $s
    python3 KnapsackExpDYS.py --model_name DYS-SPO --num_items 400 --seed $s
    python3 KnapsackExpDYS.py --model_name DYS-SCE --num_items 400 --seed $s
done

