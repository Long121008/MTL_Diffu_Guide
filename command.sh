nohup python train.py --problem=Train_ALL --model_type=SlotDiffModel \
    --enable_slot_diffusion --slot_num=32 --problem_size=50 \
    > slotdiffpomo_size50_slotnum32_e0.log 2>&1 &