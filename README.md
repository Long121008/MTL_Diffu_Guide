<h1 align="center"> SOLID: Slot-Oriented Learning and Integrated Diffusion </h1>

## How to Run

<details>
    <summary><strong>Train</strong></summary>


```shell
# Default: --problem_size=50 --pomo_size=50 --gpu_id=0 --slot_num=32

# 0. SOLID-POMO-MTL
python train.py --problem=Train_ALL --model_type=SlotDiffModel --enable_slot_diffusion 

# 1. SOLID-MVMoE/4E 
python train.py --problem=Train_ALL --model_type=SlotDiffMOEModel --num_experts=4 --routing_level=node --routing_method=input_choice --enable_slot_diffusion

```

</details>

<details>
    <summary><strong>Evaluation</strong></summary>


```shell

# 0. SOLID-POMO-MTL
python test.py --problem=ALL --model_type=SlotDiffModel --enable_slot_diffusion --checkpoint={MODEL_PATH}

# 1. SOLID-MVMoE/4E
python test.py --problem=ALL --model_type=SlotDiffMOEModel --num_experts=4 --routing_level=node --routing_method=input_choice --enable_slot_diffusion --checkpoint={MODEL_PATH}

# 2. Evaluation on CVRPLIB
python test.py --problem=CVRP --model_type={MODEL_TYPE} --checkpoint={MODEL_PATH} --test_set_path=../data/CVRP-LIB

# 3. Evaluation on Set Solomon
python test.py --problem=VRPTW --model_type={MODEL_TYPE} --checkpoint={MODEL_PATH} --test_set_path=../data/Solomon
```

</details>

<details>
    <summary><strong>Baseline</strong></summary>


```shell
# 0. LKH3 - Support for ["CVRP", "OVRP", "VRPL", "VRPTW"]
python LKH_baseline.py --problem={PROBLEM} --datasets={DATASET_PATH} -n=1000 --cpus=32 -runs=1 -max_trials=10000

# 1. HGS - Support for ["CVRP", "VRPTW"]
python HGS_baseline.py --problem={PROBLEM} --datasets={DATASET_PATH} -n=1000 --cpus=32 -max_iteration=20000

# 2. OR-Tools - Support for all 16 VRP variants
python OR-Tools_baseline.py --problem={PROBLEM} --datasets={DATASET_PATH} -n=1000 --cpus=32 -timelimit=20

# 3. POMO
python train.py --problem={PROBLEM} --model_type=SINGLE

# 4. POMO-MTL
python train.py --problem=Train_ALL --model_type=MTL

# 5. MVMoE/4E 
python train.py --problem=Train_ALL --model_type=MOE --num_experts=4 --routing_level=node --routing_method=input_choice 

# 6. POMO-MTL-Mixed
python train.py --problem=Train_ALL --model_type=MTL_Mixed

# 7. MVMoE-Mixed
python train.py --problem=Train_ALL --model_type=MOE_Mixed

```

</details>

## Dependency
Python >= 3.9

Pytorch >= 2.0.0

Geoopt >= 0.4.0

CUDA >= 11.8
