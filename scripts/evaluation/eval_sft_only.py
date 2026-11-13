import os, sys, json
sys.path.insert(0, os.path.expanduser('~/BindCraft'))
from colabdesign import mk_afdesign_model, clear_mem

params_dir = os.path.expanduser('~/BindCraft/params')

with open('benchmark/generated_sequences/sft_only_sequences.json') as f:
    data = json.load(f)

out_path = 'benchmark/sft_only_results.json'
results = {}
if os.path.exists(out_path):
    with open(out_path) as f:
        results = json.load(f)

for target, info in data.items():
    if target in results:
        print(f"{target}: already done, skipping", flush=True)
        continue
    pdb_path = info['pdb_path']
    target_chain = info['target_chain']
    seqs = info['sequences']
    
    iptms = []
    for seq in seqs:
        try:
            clear_mem()
            af_model = mk_afdesign_model(protocol="binder", data_dir=params_dir)
            af_model.prep_inputs(pdb_filename=pdb_path, chain=target_chain, binder_len=len(seq), hotspot=None)
            af_model.predict(seq=seq, models=["model_1_ptm"], num_recycles=1, verbose=False)
            iptm = float(af_model.aux["log"]['i_ptm'])
            iptms.append(iptm)
        except Exception as e:
            iptms.append(None)
    
    valid = [v for v in iptms if v is not None]
    mean_iptm = sum(valid)/len(valid) if valid else None
    results[target] = {'all': iptms, 'mean': mean_iptm, 'max': max(valid) if valid else None}
    print(f"{target}: mean={mean_iptm:.4f}" if mean_iptm else f"{target}: all failed", flush=True)
    
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

print("\nDone all targets", flush=True)
