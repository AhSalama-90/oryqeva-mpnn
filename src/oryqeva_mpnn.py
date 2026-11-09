"""
OryqevaMPNN: A DPO-finetuned variant of ProteinMPNN trained on
experimentally-verified binder-design preference pairs.

Base model: vanilla ProteinMPNN (v_48_020.pt)
Training data: 3,680 preference pairs (anthropic + overath sources only,
               verified sequence/PDB exact match, 23 unique targets)
Training method: Direct Preference Optimization (DPO) with reference
                 reinitialization every 400 steps (6 rounds)
Result: mean i_pTM 0.488 vs ProteinMPNN's 0.424 on 21 fully held-out
        targets (paired t-test p=0.047)
"""

import sys, os, json, torch, random, numpy as np
sys.path.insert(0, os.path.expanduser('~/ProteinMPNN'))
from protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def build_model():
    return ProteinMPNN(
        num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.1, k_neighbors=48,
    )


def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = build_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# Training (DPO on preference pairs)
# ---------------------------------------------------------------------------

def seq_is_valid(seq):
    return all(c in ALPHABET for c in seq)


def get_full_logprobs_and_mean(model, pdb_path, binder_chain, target_chain, binder_sequence):
    parsed = parse_PDB(pdb_path, input_chain_list=[binder_chain, target_chain])
    pdb_name = parsed[0]['name']
    chain_id_dict = {pdb_name: ([binder_chain], [target_chain])}
    (X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list,
     visible_list_list, masked_list_list, masked_chain_length_list_list,
     chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask,
     tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all,
     bias_by_res_all, tied_beta) = tied_featurize(
        parsed, DEVICE, chain_id_dict, None, None, None, None, ca_only=False)
    randn = torch.randn(chain_M.shape, device=DEVICE)
    log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
    seq_indices = torch.tensor([ALPHABET.index(a) for a in binder_sequence], device=DEVICE)
    binder_positions = (chain_M_pos[0] == 1).nonzero(as_tuple=True)[0][:len(binder_sequence)]
    per_residue_lp = log_probs[0, binder_positions, :].gather(1, seq_indices.unsqueeze(1)).squeeze(1)
    return per_residue_lp.sum(), per_residue_lp.mean()


def dpo_step(policy_model, ref_model, pair, beta=0.5, lam=1.0):
    pos, neg = pair['preferred'], pair['dispreferred']
    try:
        policy_chosen_lp, policy_chosen_mean = get_full_logprobs_and_mean(
            policy_model, pos['pdb_path'], pos['binder_chain'], pos['target_chain'], pos['sequence'])
        policy_rejected_lp, _ = get_full_logprobs_and_mean(
            policy_model, neg['pdb_path'], neg['binder_chain'], neg['target_chain'], neg['sequence'])
        with torch.no_grad():
            ref_chosen_lp, _ = get_full_logprobs_and_mean(
                ref_model, pos['pdb_path'], pos['binder_chain'], pos['target_chain'], pos['sequence'])
            ref_rejected_lp, _ = get_full_logprobs_and_mean(
                ref_model, neg['pdb_path'], neg['binder_chain'], neg['target_chain'], neg['sequence'])
        policy_logratio = policy_chosen_lp - policy_rejected_lp
        ref_logratio = ref_chosen_lp - ref_rejected_lp
        dpo_loss = -torch.nn.functional.logsigmoid(beta * (policy_logratio - ref_logratio))
        sft_loss = -policy_chosen_mean
        return dpo_loss + lam * sft_loss
    except Exception as e:
        print(f"Skipping pair due to error: {e}", flush=True)
        return None


def train(pairs_path, output_prefix, n_rounds=6, steps_per_round=400, lr=5e-6):
    base_checkpoint = torch.load(
        os.path.expanduser('~/ProteinMPNN/vanilla_model_weights/v_48_020.pt'),
        map_location='cpu')

    ref_model = build_model()
    ref_model.load_state_dict(base_checkpoint['model_state_dict'])
    ref_model = ref_model.to(DEVICE).eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    policy_model = build_model()
    policy_model.load_state_dict(base_checkpoint['model_state_dict'])
    policy_model = policy_model.to(DEVICE).train()

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)

    with open(pairs_path) as f:
        all_pairs = json.load(f)

    filtered_pairs = [
        p for p in all_pairs
        if seq_is_valid(p['preferred']['sequence']) and seq_is_valid(p['dispreferred']['sequence'])
    ]
    print(f"Training on {len(filtered_pairs)} valid preference pairs", flush=True)

    by_source = {}
    for p in filtered_pairs:
        src = p['preferred'].get('source', 'unknown')
        by_source.setdefault(src, []).append(p)
    print(f"Sources: { {k: len(v) for k, v in by_source.items()} }", flush=True)

    random.seed(42)
    for src in by_source:
        random.shuffle(by_source[src])

    source_names = list(by_source.keys())
    source_idx = {s: 0 for s in source_names}
    global_step = 0

    for round_num in range(1, n_rounds + 1):
        ref_model.load_state_dict(policy_model.state_dict())
        for p in ref_model.parameters():
            p.requires_grad = False

        total_loss_sum = 0.0
        n_done = 0
        for _ in range(steps_per_round):
            src = source_names[global_step % len(source_names)]
            pool = by_source[src]
            global_step += 1
            if not pool:
                continue
            idx = source_idx[src] % len(pool)
            pair = pool[idx]
            source_idx[src] += 1

            optimizer.zero_grad()
            loss = dpo_step(policy_model, ref_model, pair)
            if loss is None:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss_sum += loss.item()
            n_done += 1

        avg_loss = total_loss_sum / max(n_done, 1)
        print(f"Round {round_num}/{n_rounds}: avg_loss={avg_loss:.4f}", flush=True)

        torch.save(
            {'model_state_dict': policy_model.state_dict(), 'num_edges': 48,
             'round': round_num, 'base': 'ProteinMPNN_raw'},
            f"{output_prefix}_round{round_num}.pt")

    return policy_model


def sft_finetune(checkpoint_path, pairs_path, output_path, n_steps=1200, lr=2e-6):
    """Fine-tune an existing checkpoint with simple supervised learning on
    the preferred (successful) sequences only, without any preference
    comparison. Intended to be run after DPO training (see README for the
    DPO+SFT ablation result)."""
    model = load_checkpoint(checkpoint_path)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with open(pairs_path) as f:
        all_pairs = json.load(f)

    seen = set()
    positive_examples = []
    for p in all_pairs:
        pos = p['preferred']
        key = pos['pdb_path'] + pos['sequence']
        if key not in seen and seq_is_valid(pos['sequence']):
            seen.add(key)
            positive_examples.append(pos)

    print(f"SFT fine-tuning on {len(positive_examples)} unique positive examples", flush=True)
    random.seed(42)
    random.shuffle(positive_examples)

    total_loss = 0.0
    n_done = 0
    for step in range(n_steps):
        example = positive_examples[step % len(positive_examples)]
        try:
            lp_sum, lp_mean = get_full_logprobs_and_mean(
                model, example['pdb_path'], example['binder_chain'],
                example['target_chain'], example['sequence'])
        except Exception:
            continue
        loss = -lp_mean
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_done += 1
        if (step + 1) % 200 == 0:
            print(f"Step {step+1}/{n_steps}: avg_loss={total_loss/n_done:.4f}", flush=True)

    torch.save({'model_state_dict': model.state_dict(), 'num_edges': 48, 'base': 'DPO_then_SFT'}, output_path)
    print(f"\nSaved {output_path}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Generation (uses ProteinMPNN's own sample() method, unmodified)
# ---------------------------------------------------------------------------

def generate_sequence(model, pdb_path, binder_chain, target_chain, binder_len,
                       temperature=0.1, seed=0):
    torch.manual_seed(seed)
    parsed = parse_PDB(pdb_path, input_chain_list=[binder_chain, target_chain])
    pdb_name = parsed[0]['name']
    chain_id_dict = {pdb_name: ([binder_chain], [target_chain])}
    (X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list,
     visible_list_list, masked_list_list, masked_chain_length_list_list,
     chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask,
     tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all,
     bias_by_res_all, tied_beta) = tied_featurize(
        parsed, DEVICE, chain_id_dict, None, None, None, None, ca_only=False)
    randn = torch.randn(chain_M.shape, device=DEVICE)
    omit_AAs_np = np.zeros(21).astype(np.float32)
    bias_AAs_np = np.zeros(21).astype(np.float32)
    bias_by_res = torch.zeros((X.shape[0], X.shape[1], 21), device=DEVICE)
    with torch.no_grad():
        sample_dict = model.sample(
            X, randn, S, chain_M, chain_encoding_all, residue_idx, mask=mask,
            temperature=temperature, omit_AAs_np=omit_AAs_np,
            bias_AAs_np=bias_AAs_np, chain_M_pos=chain_M_pos,
            bias_by_res=bias_by_res)
    S_sample = sample_dict['S']
    seq_full = ''.join(ALPHABET[a] for a in S_sample[0].cpu().numpy())
    return seq_full[:binder_len]


# ---------------------------------------------------------------------------
# Evaluation helper: load a trained checkpoint and generate for one target
# ---------------------------------------------------------------------------

def design_binder(checkpoint_path, pdb_path, binder_chain, target_chain,
                   binder_len, n_samples=5):
    model = load_checkpoint(checkpoint_path)
    return [
        generate_sequence(model, pdb_path, binder_chain, target_chain, binder_len, seed=s)
        for s in range(n_samples)
    ]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="OryqevaMPNN training and inference")
    subparsers = parser.add_subparsers(dest='command', required=True)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--pairs', required=True, help='Path to preference-pairs JSON')
    train_parser.add_argument('--output-prefix', required=True)
    train_parser.add_argument('--rounds', type=int, default=6)
    train_parser.add_argument('--steps-per-round', type=int, default=400)

    sft_parser = subparsers.add_parser('sft-finetune')
    sft_parser.add_argument('--checkpoint', required=True, help='DPO checkpoint to fine-tune')
    sft_parser.add_argument('--pairs', required=True)
    sft_parser.add_argument('--output', required=True)
    sft_parser.add_argument('--steps', type=int, default=1200)
    sft_parser.add_argument('--lr', type=float, default=2e-6)

    design_parser = subparsers.add_parser('design')
    design_parser.add_argument('--checkpoint', required=True)
    design_parser.add_argument('--pdb', required=True)
    design_parser.add_argument('--binder-chain', required=True)
    design_parser.add_argument('--target-chain', required=True)
    design_parser.add_argument('--binder-len', type=int, required=True)
    design_parser.add_argument('--n-samples', type=int, default=5)

    args = parser.parse_args()

    if args.command == 'train':
        train(args.pairs, args.output_prefix, n_rounds=args.rounds,
              steps_per_round=args.steps_per_round)
    elif args.command == 'sft-finetune':
        sft_finetune(args.checkpoint, args.pairs, args.output,
                     n_steps=args.steps, lr=args.lr)
    elif args.command == 'design':
        sequences = design_binder(
            args.checkpoint, args.pdb, args.binder_chain, args.target_chain,
            args.binder_len, n_samples=args.n_samples)
        for i, seq in enumerate(sequences):
            print(f">design_{i}\n{seq}")
