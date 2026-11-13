import sys, os, json, torch, random
sys.path.insert(0, os.path.expanduser('~/ProteinMPNN'))
from protein_mpnn_utils import ProteinMPNN, tied_featurize, parse_PDB

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}", flush=True)
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

def make_model():
    return ProteinMPNN(num_letters=21, node_features=128, edge_features=128, hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3, augment_eps=0.1, k_neighbors=48)

# Stage 2 starts from the best DPO checkpoint (round 5)
base_checkpoint = torch.load(os.path.expanduser('~/dpo_only_best.pt'), map_location='cpu')

model = make_model()
model.load_state_dict(base_checkpoint['model_state_dict'])
model = model.to(device).train()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)  # lower lr to preserve what the model already learned

with open(os.path.expanduser('~/train_pairs_TRULY_CLEAN.json')) as f:
    all_pairs = json.load(f)

def seq_is_valid(seq):
    return all(c in ALPHABET for c in seq)

positive_examples = []
seen = set()
for p in all_pairs:
    pos = p['preferred']
    key = pos['pdb_path'] + pos['sequence']
    if key not in seen and seq_is_valid(pos['sequence']):
        seen.add(key)
        positive_examples.append(pos)

print(f"DPO+SFT: fine-tuning on {len(positive_examples)} positive examples, starting from dpo_only_best", flush=True)

random.seed(42)
random.shuffle(positive_examples)

def sft_step(example):
    try:
        parsed = parse_PDB(example['pdb_path'], input_chain_list=[example['binder_chain'], example['target_chain']])
        pdb_name = parsed[0]['name']
        chain_id_dict = {pdb_name: ([example['binder_chain']], [example['target_chain']])}
        X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(parsed, device, chain_id_dict, None, None, None, None, ca_only=False)
        randn = torch.randn(chain_M.shape, device=device)
        log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all, randn)
        seq_indices = torch.tensor([ALPHABET.index(a) for a in example['sequence']], device=device)
        binder_positions = (chain_M_pos[0] == 1).nonzero(as_tuple=True)[0][:len(example['sequence'])]
        per_residue_lp = log_probs[0, binder_positions, :].gather(1, seq_indices.unsqueeze(1)).squeeze(1)
        loss = -per_residue_lp.mean()
        return loss
    except Exception:
        return None

N_STEPS = 1200  # shorter than stage 1: refinement, not training from scratch
total_loss = 0.0
n_done = 0

for step in range(N_STEPS):
    example = positive_examples[step % len(positive_examples)]
    optimizer.zero_grad()
    loss = sft_step(example)
    if loss is None:
        continue
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    total_loss += loss.item()
    n_done += 1
    if (step+1) % 200 == 0:
        print(f"Step {step+1}/{N_STEPS}: avg_loss={total_loss/n_done:.4f}", flush=True)

torch.save({'model_state_dict': model.state_dict(), 'num_edges': 48, 'base': 'DPO_then_SFT'},
           os.path.expanduser('~/dpo_plus_sft_model.pt'))
print("\nSaved dpo_plus_sft_model.pt", flush=True)
