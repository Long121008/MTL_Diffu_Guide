python test.py --problem=CVRP --model_type=MTL --checkpoint="results/20251105_133802/epoch-100.pt"

python test.py --problem=OVRPTW --model_type=MOE --num_experts=4 --routing_level=node --routing_method=input_choice --checkpoint="results/20251106_073934/epoch-100.pt"

python test.py --problem=OVRPBLTW --model_type=MTL --checkpoint="results/20251109_120042/epoch-100.pt"

python OR-Tools_baseline.py --problem=VRPTW --datasets="c:/Users/HC COMPUTER/Prj_MTL/Routing-MVMoE/data_train3phase/VRPTW/vrptw50_uniform.pkl" -n=3334 --cpus=1 -timelimit=200

python LKH_baseline.py --problem=CVRP --datasets="c:/Users/HC COMPUTER/Prj_MTL/Routing-MVMoE/data_train3phase/CVRP/cvrp50_uniform.pkl" -n=3334 --cpus=1 -runs=1 -max_trials=10000

results/20251111_211024/epoch-60.pt

python test.py --problem=CVRP --model_type=MTL --checkpoint="results/20251105_133802/epoch-60.pt"