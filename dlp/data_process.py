import pandas as pd
import torch
import random
import json
import os
import pickle
from transformers import T5Tokenizer, AutoTokenizer, EsmModel
import pickle
from collections import Counter
from Bio.Seq import Seq
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import numpy as np

from batch import Batch, GPTBatch

max_seq_len = 1000
max_tax_len = 150

# Character vocabulary for protein sequences (20 amino acids + 1 padding)
vocab = "ACDEFGHIKLMNPQRSTVWY"
char_to_idx = {char: idx + 1 for idx, char in enumerate(vocab)}  # Start index from 1 for padding
# Sequence encoder: Convert the protein sequence into integers
def encode_sequence(sequence):
    return [char_to_idx.get(char, 0) for char in sequence] + [0 for _ in range(max_seq_len - len(sequence))]  # 0 for unknown characters or padding 

###################################### 0_Classifier ##############################################
#     df = pd.read_csv("../data/tax_names.csv")
#     df.dropna(inplace = True)
#     taxid_idx = {tax_id: idx + 1 for idx, tax_id in enumerate(df['Scientific name'].values)}
#     tax_vocab_size = len(taxid_idx)


###################################### 1_Hierarchy ##############################################
rank2name_file = "../data/rank2name.pkl"
name2rank_file = "../data/name2rank.pkl"

# Check if the file exists
if os.path.exists(rank2name_file) and os.path.exists(name2rank_file):
    # Load the dictionary if the file exists
    with open(rank2name_file, "rb") as f:
        rank2name = pickle.load(f)
    with open(name2rank_file, "rb") as f:
        name2rank = pickle.load(f)
    print("Loaded dictionary.")
else:
    df = pd.read_csv("../data/rank_tax.csv")
    df.dropna(inplace = True)
    
    name2rank = {}
    rank2name = {}
    for _, r in df.iterrows():
        name = r['Scientific name']
        rank = r['Rank']

        if name in ["cellular organisms", "other entries", "unclassified entries", "Viruses"]:
            rank = "begining root"
        
        name2rank[name] = rank
    
        if rank in rank2name:
            rank2name[rank].append(name)
        else:    
            rank2name[rank] = [name]
            
    with open(rank2name_file, "wb") as f:
        pickle.dump(rank2name, f)
    with open(name2rank_file, "wb") as f:
        pickle.dump(name2rank, f)
    print("Created and saved dictionary.")

tax_vocab_sizes = {
    k: len(v) for k,v in rank2name.items() if k != "species"
}

# print(tax_vocab_sizes)
# Print tax_vocab_sizes sorted by value (number of taxa per rank)
# sorted_sizes = dict(sorted(tax_vocab_sizes.items(), key=lambda x: x[1], reverse=True))
# print("\nTaxonomic ranks sorted by number of taxa:")
# for rank, size in sorted_sizes.items():
#     print(f"{rank}: {size}")


hierarchy = list(rank2name.keys())
# Assign importance values, starting with higher values for broader ranks
importance_dict = {rank: len(hierarchy) - i for i, rank in enumerate(rank2name.keys()) if rank != "species"}
# Increase the importance for "no rank" specifically, if needed
importance_dict["begining root"] = 55

level_encoder = {
    k: {name: idx + 1 for idx, name in enumerate(v)}
    for k,v in rank2name.items() if k != "species"
}

level_decoder = {
    k: {idx + 1: name for idx, name in enumerate(v)}
    for k,v in rank2name.items() if k != "species"
}

# Sequence encoder: Convert the protein sequence into integers
def encode_lineage(lineage_str):
    taxes_str = lineage_str.split(", ")
    # print(taxes_str)
    ranks = [name2rank.get(s, "species") for s in taxes_str]
    # print(ranks)
    encoded = {k: [0] for k in rank2name.keys() if k != "species"}
    for tax_str, r in zip(taxes_str, ranks):
        if r == "species":
            continue
        encoded[r][0] = level_encoder[r].get(tax_str, 0)

    return {k : v for k,v in encoded.items() if k!= "species"}


def data_to_tensor_batch(b):
    sequences = [encode_sequence(e['sequence']) for e in b]
    tax_ids = [encode_lineage(e['tax_lineage']) for e in b]
    combined_dict = {}
    for d in tax_ids:
        for key, value in d.items():
            combined_dict.setdefault(key, []).extend(value)

    tensor_encoded = {k: torch.LongTensor(v) for k,v in combined_dict.items()}
    return Batch(torch.LongTensor(sequences), tensor_encoded)


###################################### 2_Tokenizer ######################################
# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
len_tokenizer = len(tokenizer.vocab)
# print(len_tokenizer)
# tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")


def encode_lineage_tokenizer(tax_lineage):
    return tokenizer.encode(tax_lineage.split(", "), add_special_tokens=True, padding='max_length', truncation=True, max_length=max_tax_len, is_split_into_words= True)

def tokenizer_data_to_tensor_batch(b):
    sequences = [encode_sequence(e['sequence']) for e in b]
    tax_ids = [encode_lineage_tokenizer(e['tax_lineage']) for e in b]
    encoded_list = [[1 if _ in tax_id else 0 for _ in range(len_tokenizer)] for tax_id in tax_ids]
    return Batch(torch.LongTensor(sequences), torch.LongTensor(encoded_list))

###################################### 3_4_T5 ##############################################
def T5_data_to_tensor_batch(b):
    sequences = [encode_sequence(e['sequence']) for e in b]
    tax_ids = [encode_lineage_tokenizer(e['tax_lineage']) for e in b]
    return Batch(torch.LongTensor(sequences), torch.LongTensor(tax_ids))

###################################### 5_GPT ##############################################
# Character vocabulary for protein sequences (20 amino acids + start + end + padding)
vocab = "ACDEFGHIKLMNPQRSTVWY"
START_TOKEN = "<s>"  # Start token
END_TOKEN = "</s>"   # End token
UNK_TOKEN = "<unk>"  # Padding token

# Create vocabulary with special tokens
# 0: padding, 1: start, 2: end, 3+: amino acids
special_char_to_idx = {
    UNK_TOKEN: 0,
    START_TOKEN: 1,
    END_TOKEN: 2,
    **{char: idx + 3 for idx, char in enumerate(vocab)}
}

special_idx_to_char = {v: k for k, v in special_char_to_idx.items()}


def special_encode_sequence(sequence):
    return [special_char_to_idx.get(char, 0) for char in sequence]


def GPT_data_to_tensor_batch(b, chunk_size):
    i_sequences = []
    o_sequences = []
    
    for e in b:
        sequence = START_TOKEN + e['sequence'] + END_TOKEN
        if len(sequence) <= chunk_size:
            for _ in  range(chunk_size - len(sequence) + 1):
                sequence = sequence + UNK_TOKEN
        
        ix = torch.randint(len(sequence) - chunk_size, (1,)).item()

        i_sequences.append(sequence[ix:ix+chunk_size])
        o_sequences.append(sequence[ix+1:ix+1+chunk_size])

    input_ids = [special_encode_sequence(s) for s in i_sequences]
    output_ids = [special_encode_sequence(s) for s in o_sequences]
    
    return GPTBatch(torch.LongTensor(input_ids), torch.LongTensor(output_ids))


######################################## 7_ESM ####################################
model_name = "esm2_t6_8M_UR50D"
_model = None
_tokenizer = None
_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def get_esm_model():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        _model = EsmModel.from_pretrained(f"facebook/{model_name}").to(_device)
    return _model, _tokenizer

def esm_embedding_sequence(sequences):
    model, tokenizer = get_esm_model()
    inputs = tokenizer(
        sequences,
        return_tensors="pt", 
        padding='max_length', 
        truncation=True, 
        max_length=max_seq_len
    ).to(_device)
    
    with torch.no_grad():  # Add this to reduce memory usage
        # Forward pass through the model
        outputs = model(**inputs).last_hidden_state
        # Compute mean across the sequence dimension (or any other pooling method)
        output_embeddings = outputs.cpu()  # Move back to CPU
    return output_embeddings

def embedding_data_to_tensor_batch(split, i):
    sequences, lineage_str = torch.load(f'../embeddings/esm_embeddings/{split}/{i}.pt')
    tax_ids = [encode_lineage_tokenizer(s) for s in lineage_str]
    
    return Batch(sequences, torch.LongTensor(tax_ids))

#################################### 9_ESM_Hierarchy ####################################
def esm_hierarchy_data_to_tensor_batch(split, i):
    sequences, lineage_str, sequence_strs = torch.load(f'../embeddings/esm_embeddings/{split}/{i}.pt')
    tax_ids = [encode_lineage(s) for s in lineage_str]

    combined_dict = {}
    for d in tax_ids:
        for key, value in d.items():
            combined_dict.setdefault(key, []).extend(value)

    tensor_encoded = {k: torch.LongTensor(v) for k,v in combined_dict.items()}

    return Batch(sequences[0], tensor_encoded), sequence_strs

#################################### 11_ESM_FineTune ####################################
tokenizer_ = AutoTokenizer.from_pretrained(f"facebook/esm1b_t33_650M_UR50S")
def simple_data_to_tensor_batch(b, max_seq_len_):
    inputs = tokenizer_(
        [e['sequence'] for e in b],
        return_tensors="pt", 
        padding='max_length', 
        truncation=True, 
        max_length=max_seq_len_
    ).to('cuda:0')
    
    tax_ids = [encode_lineage(e['tax_lineage']) for e in b]
    combined_dict = {}
    for d in tax_ids:
        for key, value in d.items():
            combined_dict.setdefault(key, []).extend(value)

    tensor_encoded = {k: torch.LongTensor(v) for k,v in combined_dict.items()}
    return Batch(inputs, tensor_encoded)


#################################### 12_ESM_FineTune_Sampling data ####################################
def mix_data_to_tensor_batch(b_virues, b_non_virues, max_seq_len_, partition=0.25):
    split_point = int(len(b_virues) * partition)
    b = b_virues[:split_point] + b_non_virues[-len(b_virues) + split_point:]
    random.shuffle(b)  # In-place shuffle

    inputs = tokenizer_(
        [e['sequence'] for e in b],
        return_tensors="pt", 
        padding='max_length', 
        truncation=True, 
        max_length=max_seq_len_
    ).to('cuda:0')
    
    tax_ids = [encode_lineage(e['tax_lineage']) for e in b]
    combined_dict = {}
    for d in tax_ids:
        for key, value in d.items():
            combined_dict.setdefault(key, []).extend(value)

    tensor_encoded = {k: torch.LongTensor(v) for k,v in combined_dict.items()}
    return Batch(inputs, tensor_encoded)


#################################### 13_Pure CNN ####################################
class SequenceEncoder:
    def __init__(self, max_length=max_seq_len):
        self.max_length = max_length
        
        # Standard amino acid vocabulary
        self.aa_vocab = {
            'PAD': 0,  # Padding token
            'UNK': 1,  # Unknown amino acid
            'A': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
            'G': 7, 'H': 8, 'I': 9, 'K': 10, 'L': 11,
            'M': 12, 'N': 13, 'P': 14, 'Q': 15, 'R': 16,
            'S': 17, 'T': 18, 'V': 19, 'W': 20, 'Y': 21
        }
        
        # Physicochemical properties
        self.hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        self.volume = {
            'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
            'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
            'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
            'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
        }
        
        self.polarity = {
            'A': 0.0, 'C': 1.48, 'D': 49.7, 'E': 49.9, 'F': 0.35,
            'G': 0.0, 'H': 51.6, 'I': 0.13, 'K': 49.5, 'L': 0.13,
            'M': 1.43, 'N': 3.38, 'P': 1.58, 'Q': 3.53, 'R': 52.0,
            'S': 1.67, 'T': 1.66, 'V': 0.13, 'W': 2.1, 'Y': 1.61
        }
    
    def encode_sequence(self, sequence):
        """
        Encode a protein sequence with multiple features
        """
        
        # Basic sequence encoding
        seq_encoding = np.zeros((self.max_length, 4))  # 4 features: index, hydrophobicity, volume, polarity
        
        for i, aa in enumerate(sequence[:self.max_length]):
            # One-hot encoding index
            seq_encoding[i, 0] = self.aa_vocab.get(aa, self.aa_vocab['UNK'])
            
            # Physicochemical properties
            seq_encoding[i, 1] = self.hydrophobicity.get(aa, 0)
            seq_encoding[i, 2] = self.volume.get(aa, 0)
            seq_encoding[i, 3] = self.polarity.get(aa, 0)
        
        # Create attention mask
        attention_mask = np.zeros(self.max_length)
        attention_mask[:min(len(sequence), self.max_length)] = 1
        
        return seq_encoding, attention_mask


    def compute_sequence_features(self, sequence):
        """
        Compute additional sequence features
        """
        try:
            analysis = ProteinAnalysis(sequence)
            
            # Calculate various protein properties
            features = {
                'molecular_weight': analysis.molecular_weight(),
                'aromaticity': analysis.aromaticity(),
                'instability_index': analysis.instability_index(),
                'isoelectric_point': analysis.isoelectric_point(),
                'gravy': analysis.gravy(),  # Grand average of hydropathy
                'secondary_structure_fraction': analysis.secondary_structure_fraction()  # Returns (helix, turn, sheet)
            }
            
            # Amino acid composition
            aa_composition = analysis.get_amino_acids_percent()
            
            # Combine all features
            feature_vector = np.array([
                features['molecular_weight'],
                features['aromaticity'],
                features['instability_index'],
                features['isoelectric_point'],
                features['gravy'],
                *features['secondary_structure_fraction'],  # Unpack helix, turn, sheet
                *[aa_composition.get(aa, 0) for aa in 'ACDEFGHIKLMNPQRSTVWY']  # AA composition
            ])
            
        except:
            # If there's any error in computation, return zeros
            feature_vector = np.zeros(28)  # 6 properties + 20 AA frequencies
        
        return feature_vector


encoder = SequenceEncoder()

def CNN_prepare_batch(b_virues, b_non_virues=None, max_seq_len_=500, partition=0.25):
    if b_non_virues is not None:
        split_point = int(len(b_virues) * partition)
        b = b_virues[:split_point] + b_non_virues[-len(b_virues) + split_point:]
        random.shuffle(b)  # In-place shuffle
    else:
        b = b_virues
    sequences = [e['sequence'] for e in b]
    
    batch_encodings = []
    batch_masks = []
    batch_global_features = []
    
    for seq in sequences:
        # Get sequence encodings and mask
        encoding, mask = encoder.encode_sequence(seq)
        global_features = encoder.compute_sequence_features(seq)
        
        batch_encodings.append(encoding)
        batch_masks.append(mask)
        batch_global_features.append(global_features)
    
    # Convert to tensors
    batch_encodings = torch.FloatTensor(np.array(batch_encodings))
    batch_masks = torch.FloatTensor(np.array(batch_masks))
    batch_global_features = torch.FloatTensor(np.array(batch_global_features))
    
    inputs = {
        "batch_encoding": batch_encodings,
        "batch_maks": batch_masks,
        "batch_global_features": batch_global_features
    }

    tax_ids = [encode_lineage(e['tax_lineage']) for e in b]
    combined_dict = {}
    for d in tax_ids:
        for key, value in d.items():
            combined_dict.setdefault(key, []).extend(value)

    tensor_encoded = {k: torch.LongTensor(v) for k,v in combined_dict.items()}
    return Batch(inputs, tensor_encoded)

