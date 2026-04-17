import os
import torch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


class SlotConsistency:

    def __init__(self, args, env_params, device):
        self.args = args
        self.env_params = env_params
        self.device = device

    def _align_slots(self, attn1, attn2):
        sim = torch.matmul(attn1, attn2.T) 
        cost = -sim.cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
        return col_ind

    def run(self, model, env):

        print("\n>> Computing Slot Consistency (Multi-Variant)...")

        model.eval()

        variants = {
            "CVRP":  [1,0,0,0,0],
            "OVRP":  [1,1,0,0,0],
            "VRPB":  [1,0,1,0,0],
            "OVRPB":  [1,1,1,0,0],
            "VRPBL": [1,0,1,1,0],
        }

        trained_variants = {"CVRP", "OVRP", "VRPB"}
        unseen_variants  = {"OVRPB", "VRPBL"}

        results = []

        with torch.no_grad():

            for _ in range(100):

                data = env.get_random_problems(1, self.env_params["problem_size"])

                for v1_name, v1_emb in variants.items():
                    for v2_name, v2_emb in variants.items():

                        if v1_name >= v2_name:
                            continue

                        attn1 = self._encode(model, env, data, v1_emb)
                        attn2 = self._encode(model, env, data, v2_emb)

                        # ===== ALIGN =====
                        perm = self._align_slots(attn1, attn2)
                        attn2 = attn2[perm]

                        # ===== ASSIGN =====
                        assign1 = attn1.argmax(dim=0)
                        assign2 = attn2.argmax(dim=0)

                        overlap = (assign1 == assign2).float().mean().item()

                        if v1_name in trained_variants and v2_name in trained_variants:
                            pair_type = "trained-trained"
                        elif v1_name in unseen_variants and v2_name in unseen_variants:
                            pair_type = "unseen-unseen"
                        else:
                            pair_type = "trained-unseen"

                        results.append({
                            "pair": f"{v1_name}-{v2_name}",
                            "type": pair_type,
                            "overlap": overlap
                        })

        df = pd.DataFrame(results)

        summary = df.groupby("type")["overlap"].mean()

        print("\n>> Summary:")
        for k, v in summary.items():
            print(f"{k}: {v*100:.2f}%")

        os.makedirs(self.args.log_path, exist_ok=True)

        df.to_csv(os.path.join(self.args.log_path, "slot_consistency_detailed.csv"), index=False)
        summary.to_csv(os.path.join(self.args.log_path, "slot_consistency_summary.csv"))

        print(">> Saved detailed + summary CSV")

    def _encode(self, model, env, data, prob_emb):

        env.load_problems(1, problems=data)
        reset_state, _, _ = env.reset()

        reset_state.prob_emb = torch.FloatTensor([prob_emb]).to(self.device)

        model.pre_forward(reset_state)
        attn = model.encoder.slot_attention_module.last_attention[0][:, 1:]

        return attn