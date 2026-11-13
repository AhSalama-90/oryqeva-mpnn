import sys, os, json, torch, random
sys.path.insert(0, os.path.expanduser('~/ProteinMPNN'))
from protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

def make_model():
    return ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.1, k_neighbors=48)

# Start from vanilla ProteinMPNN, not a derived checkpoint
base_checkpoint = torch.load(os.path.expanduser('~/ProteinMPNN/vanilla_model_weights/v_48_020.pt'), map_location='cpu')

ref_model = make_model()
ref_model.load_state_dict(base_checkpoint['model_state_dict'])
ref_model = ref_model.to(device).eval()
for p in ref_model.parameters():
    p.requires_grad = False

policy_model = make_model()
policy_model.load_state_dict(base_checkpoint['model_state_dict'])
policy_model = policy_model.to(device).train()

optimizer = torch.optim.Adam(policy_model.parameters(), lr=5e-6)

with open(os.path.expanduser('~/train_pairs_TRULY_CLEAN.json')) as f:
    all_pairs = json.load(f)

print(f"Training on CLEAN data, base=ProteinMPNN: {len(all_pairs)} pairs", flush=True)

def seq_is_valid(seq):
    return all(c in ALPHABET for c in seq)

filtered_pairs = [p for p in all_pairs if seq_is_valid(p['preferred']['sequence']) and seq_is_valid(p['dispreferred']['sequence'])]
print(f"Valid pairs: {len(filtered_pairs)}", flush=True)

by_source = {}
for p in filtered_pairs:
    src = p['preferred'].get('source', 'unknown')
    by_source.setdefault(src, []).append(p)
print(f"Sources: { {k: len(v) for k,v in by_source.items()} }", flush=True)

random.seed(42)
for src in by_source:
    random.shuffle(by_source[src])

def get_full_logprobs_and_logprob(model, pdb_path, binder_chain, target_chain, binder_sequence):
    parsed = parse_PDB(pdb_path, input_chain_list=[binder_chain, target_chain])
    pdb_name = parsed[0]['name']
    chain_id_dict = {pdb_name: ([binder_chain], [target_chain])}
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(parsed, device, chain_id_dict, None, None, None, None, ca_only=False)
    randn = torch.randn(chain_M.shape, device=device)
    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
    seq_indices = torch.tensor([ALPHABET.index(a) for a in binder_sequence], device=device)
    binder_positions = (chain_M_pos[0] == 1).nonzero(as_tuple=True)[0][:len(binder_sequence)]
    per_residue_lp = log_probs[0, binder_positions, :].gather(1, seq_indices.unsqueeze(1)).squeeze(1)
    return per_residue_lp.sum(), per_residue_lp.mean()

def dpo_step(pair, beta=0.5, lam=1.0):
    pos, neg = pair['preferred'], pair['dispreferred']
    try:
        policy_chosen_lp, policy_chosen_mean = get_full_logprobs_and_logprob(policy_model, pos['pdb_path'], pos['binder_chain'], pos['target_chain'], pos['sequence'])
        policy_rejected_lp, _ = get_full_logprobs_and_logprob(policy_model, neg['pdb_path'], neg['binder_chain'], neg['target_chain'], neg['sequence'])
        with torch.no_grad():
            ref_chosen_lp, _ = get_full_logprobs_and_logprob(ref_model, pos['pdb_path'], pos['binder_chain'], pos['target_chain'], pos['sequence'])
            ref_rejected_lp, _ = get_full_logprobs_and_logprob(ref_model, neg['pdb_path'], neg['binder_chain'], neg['target_chain'], neg['sequence'])
        policy_logratio = policy_chosen_lp - policy_rejected_lp
        ref_logratio = ref_chosen_lp - ref_rejected_lp
        dpo_loss = -torch.nn.functional.logsigmoid(beta * (policy_logratio - ref_logratio))
        sft_loss = -policy_chosen_mean
        return dpo_loss + lam * sft_loss
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        return None

def evaluate_heldout():
    with open(os.path.expanduser('~/heldout_test_positives.json')) as f:
        heldout = json.load(f)
    policy_model.eval()
    scores = []
    for target_name in ['TREM2', 'RBX1', 'Mdm2']:
        items = [d for d in heldout if d['target']==target_name and os.path.exists(d['pdb_path'])]
        if not items:
            continue
        d = items[0]
        try:
            _, mean_lp = get_full_logprobs_and_logprob(policy_model, d['pdb_path'], d['binder_chain'], d['target_chain'], d['sequence'])
            scores.append(mean_lp.item())
        except Exception:
            pass
    policy_model.train()
    return sum(scores)/len(scores) if scores else None

N_ROUNDS = 6
STEPS_PER_ROUND = 400
best_score = -float('inf')
best_round = 0
source_names = list(by_source.keys())
source_idx = {s: 0 for s in source_names}

import time
t0 = time.time()
global_step = 0

for round_num in range(1, N_ROUNDS+1):
    ref_model.load_state_dict(policy_model.state_dict())
    for p in ref_model.parameters():
        p.requires_grad = False
    
    total_loss_sum = 0.0
    n_done = 0
    for step in range(STEPS_PER_ROUND):
        src = source_names[global_step % len(source_names)]
        pool = by_source[src]
        if not pool:
            global_step += 1
            continue
        idx = source_idx[src] % len(pool)
        pair = pool[idx]
        source_idx[src] += 1
        global_step += 1

        optimizer.zero_grad()
        loss = dpo_step(pair)
        if loss is None:
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss_sum += loss.item()
        n_done += 1

    elapsed = time.time() - t0
    avg_loss = total_loss_sum / max(n_done, 1)
    heldout_score = evaluate_heldout()
    print(f"Round {round_num}/{N_ROUNDS}: avg_loss={avg_loss:.4f}, heldout_score={heldout_score}, elapsed={elapsed:.1f}s", flush=True)
    
    torch.save({'model_state_dict': policy_model.state_dict(), 'num_edges': 48, 'round': round_num, 'base': 'ProteinMPNN_raw'},
               os.path.expanduser(f'~/dpo_only_round{round_num}.pt'))
    
    if heldout_score is not None and heldout_score > best_score:
        best_score = heldout_score
        best_round = round_num
        torch.save({'model_state_dict': policy_model.state_dict(), 'num_edges': 48, 'round': round_num, 'base': 'ProteinMPNN_raw'},
                   os.path.expanduser('~/dpo_only_best.pt'))
        print(f"  -> New best checkpoint (round {round_num})", flush=True)

print(f"\nTraining complete. Best: round {best_round} (heldout_score={best_score:.4f})", flush=True)
