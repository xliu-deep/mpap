# Note: Server addresses that need to be modified are at lines 1137, 1991, 2053, 2054, 2059, 2063
import streamlit as st
import sys
import os
from pathlib import Path

# Get the project root directory (parent of web_client)
_project_root = Path(__file__).parent.parent.parent.resolve()

# Add project root to path to import mpap package
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Change to project root directory so relative paths in config work correctly
os.chdir(_project_root)

import pandas as pd
import numpy as np
from rdkit import Chem
from collections import defaultdict
from rdkit.Chem.rdchem import BondType
import torch
from rdkit.Chem import AllChem
import re
import glob
import logging

# Import configuration and utilities
from mpap.config import Config
from mpap.utils import get_device, setup_seed
from mpap.data_loader import load_dataset

# Import model components from model.py
import importlib.util
model_path = _project_root / "MPAP_model_training" / "model.py"
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

spec = importlib.util.spec_from_file_location("model", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# Import model components from model.py (only what we need)
Predictor = model_module.Predictor
ResMLP = model_module.ResMLP
Affine = model_module.Affine
pack = model_module.pack

# Setup logging
logger = logging.getLogger(__name__)

# Get web_client directory for images
_web_client_dir = Path(__file__).parent.parent.resolve()

# Helper function to get image path
def get_image_path(filename):
    """Get path to image file in web_client directory."""
    img_path = _web_client_dir / filename
    return str(img_path.resolve())

# Load configuration
config_path = os.getenv('MPAP_CONFIG', None)
config = Config(config_path)

# Setup device
setup_seed(config.get('training.seed'))
device = get_device(config.get('device.cuda_device'))

# Graph feature extraction functions
def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
  """Maps inputs not in the allowable set to the last element."""
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))


def get_intervals(l):
  """For list of lists, gets the cumulative products of the lengths"""
  intervals = len(l) * [0]
  # Initalize with 1
  intervals[0] = 1
  for k in range(1, len(l)):
    intervals[k] = (len(l[k]) + 1) * intervals[k - 1]

  return intervals


def safe_index(l, e):
  """Gets the index of e in l, providing an index of len(l) if not found"""
  try:
    return l.index(e)
  except:
    return len(l)


possible_atom_list = [
    'C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Mg', 'Na', 'Br', 'Fe', 'Ca', 'Cu',
    'Mc', 'Pd', 'Pb', 'K', 'I', 'Al', 'Ni', 'Mn'
]
possible_numH_list = [0, 1, 2, 3, 4]
possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_hybridization_list = [
    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]
possible_number_radical_e_list = [0, 1, 2]
possible_chirality_list = ['R', 'S']

reference_lists = [
    possible_atom_list, possible_numH_list, possible_valence_list,
    possible_formal_charge_list, possible_number_radical_e_list,
    possible_hybridization_list, possible_chirality_list
]

intervals = get_intervals(reference_lists)


def get_feature_list(atom):
  features = 6 * [0]
  features[0] = safe_index(possible_atom_list, atom.GetSymbol())
  features[1] = safe_index(possible_numH_list, atom.GetTotalNumHs())
  features[2] = safe_index(possible_valence_list, atom.GetImplicitValence())
  features[3] = safe_index(possible_formal_charge_list, atom.GetFormalCharge())
  features[4] = safe_index(possible_number_radical_e_list,
                           atom.GetNumRadicalElectrons())
  features[5] = safe_index(possible_hybridization_list, atom.GetHybridization())
  return features


def features_to_id(features, intervals):
  """Convert list of features into index using spacings provided in intervals"""
  id = 0
  for k in range(len(features)):
    id += features[k] * intervals[k]

  # Allow 0 index to correspond to null molecule 1
  id = id + 1
  return id


def atom_to_id(atom):
  """Return a unique id corresponding to the atom type"""
  features = get_feature_list(atom)
  return features_to_id(features, intervals)


def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False):
  if bool_id_feat:
    return np.array([atom_to_id(atom)])
  else:
    results = one_of_k_encoding_unk(atom.GetSymbol(),
                                     ['C',
                                      'N',
                                      'O',
                                      'S',
                                      'F',
                                      'Si',
                                      'P',
                                      'Cl',
                                      'Br',
                                      'Mg',
                                      'Na',
                                      'Ca',
                                      'Fe',
                                      'As',
                                      'Al',
                                      'I',
                                      'B',
                                      'V',
                                      'K',
                                      'Tl',
                                      'Yb',
                                      'Sb',
                                      'Sn',
                                      'Ag',
                                      'Pd',
                                      'Co',
                                      'Se',
                                      'Ti',
                                      'Zn',
                                      'H',  # H?
                                      'Li',
                                      'Ge',
                                      'Cu',
                                      'Au',
                                      'Ni',
                                      'Cd',
                                      'In',
                                      'Mn',
                                      'Zr',
                                      'Cr',
                                      'Pt',
                                      'Hg',
                                      'Pb',
                                      'Unknown']) + \
              one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                                    Chem.rdchem.HybridizationType.SP3D2]) +\
              [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
      results = results + \
                one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if use_chirality:
      try:
        results = results + \
                  one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) +\
                  [atom.HasProp('_ChiralityPossible')]
      except:
        results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)






# rdkit GetBondType() result -> int
BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)


def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])

    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return node_features, adjacency


def inputfile(dataset, fileinput):
    data=pd.read_csv(dataset,sep='\t')

    psmiles=data['psmiles']
    poly_smiles=data['poly_smiles']
    compound_smiles=data['smiles']
    label=data['logkd'].to_list()
    size=data['average size'].to_list()
    water=data['water3'].to_list()

    pgraph, morgan, graph, adjacencies, p_adjs = [], [], [], [], []
    
    
    for polysmiles in poly_smiles:
        p_atomfeatures,p_adj=smile_to_graph(polysmiles)
        pgraph.append(p_atomfeatures)
        p_adjs.append(p_adj)
    i=1    
    for smiles in compound_smiles:
        atom_features,adj=smile_to_graph(smiles)
        graph.append(atom_features)
        adjacencies.append(adj)


        mol = Chem.MolFromSmiles(smiles)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
        npfp = np.array(list(fp.ToBitString())).astype('int8')
        morgan.append(npfp)

        i+=1
        print(i)


    dir_input = fileinput
    os.makedirs(dir_input, exist_ok=True)
    
    # Generate fingerprints placeholder (600-dimensional zero vectors)
    # Note: In production, fingerprints should be generated using proper methods
    # For now, use zero vectors as placeholder to match expected format
    # Save as regular numpy array (float64) like training data, not object array
    n_samples = len(compound_smiles)
    fingerprints = np.zeros((n_samples, 600), dtype=np.float64)
    
    # Save numpy arrays in the same format as training data
    # fingerprints: regular array (float64), morgan: regular array (int8)
    # pgraph, padjs, graph, adjacencies: object arrays for variable-length arrays
    # Ensure each element in object arrays is a proper numpy array (not nested object)
    np.save(dir_input + 'fingerprints', fingerprints)
    
    # For object arrays, ensure each element is a proper numpy array
    pgraph_clean = [np.array(p, copy=False) if isinstance(p, np.ndarray) else np.array(p) for p in pgraph]
    np.save(dir_input + 'pgraph', np.array(pgraph_clean, dtype=object))
    
    p_adjs_clean = [np.array(p, copy=False) if isinstance(p, np.ndarray) else np.array(p) for p in p_adjs]
    np.save(dir_input + 'padjs', np.array(p_adjs_clean, dtype=object))
    
    # morgan: convert list of arrays to 2D array (n_samples, 2048)
    morgan_array = np.array(morgan, dtype=np.int8)  # Shape: (n_samples, 2048)
    np.save(dir_input + 'morgan', morgan_array)
    
    graph_clean = [np.array(g, copy=False) if isinstance(g, np.ndarray) else np.array(g) for g in graph]
    np.save(dir_input + 'graph', np.array(graph_clean, dtype=object))
    
    adjacencies_clean = [np.array(a, copy=False) if isinstance(a, np.ndarray) else np.array(a) for a in adjacencies]
    np.save(dir_input + 'adjacencies', np.array(adjacencies_clean, dtype=object))
    np.save(dir_input + 'label', np.array(label))
    np.save(dir_input + 'size', np.array(size))
    np.save(dir_input + 'water', np.array(water))



# Data loading functions are now imported from mpap.data_loader
# The old functions (load_tensor, load_tensor_label, shuffle_dataset, dataset) 
# have been replaced by mpap.data_loader.load_dataset



# Model definitions are imported from MPAP_model_training/model.py
# No need to duplicate them here

def find_best_model(model_dir: str) -> str:
    """
    Find the best model by traversing the model directory and selecting
    the one with the smallest validation loss (based on filename).
    
    Models are saved with pattern: best_model_{validation_loss:.6f}.tar
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        Path to the best model file
        
    Raises:
        FileNotFoundError: If no model files are found
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Find all .tar files matching the pattern best_model_*.tar
    pattern = os.path.join(model_dir, 'best_model_*.tar')
    model_files = glob.glob(pattern)
    
    if not model_files:
        # Try alternative pattern or subdirectories
        pattern = os.path.join(model_dir, '**', '*.tar')
        model_files = glob.glob(pattern, recursive=True)
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}")
    
    # Extract validation loss from filenames
    model_losses = []
    for model_file in model_files:
        # Extract loss from filename: best_model_0.475353.tar -> 0.475353
        filename = os.path.basename(model_file)
        match = re.search(r'best_model_([\d.]+)\.tar', filename)
        if match:
            try:
                loss = float(match.group(1))
                model_losses.append((loss, model_file))
            except ValueError:
                continue
    
    if not model_losses:
        raise FileNotFoundError(
            f"No valid model files found matching pattern 'best_model_*.tar' in {model_dir}"
        )
    
    # Find model with smallest validation loss
    best_loss, best_model_path = min(model_losses, key=lambda x: x[0])
    logger.info(f"Found best model: {best_model_path} with loss {best_loss:.6f}")
    
    return best_model_path


def load_model(model_path: str, config: Config, device: torch.device):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        config: Configuration object
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    model = Predictor(
        ResMLP, Affine,
        dim=config.get('model.dim'),
        window=config.get('model.window'),
        window2=config.get('model.window2'),
        layer_gnn=config.get('model.layer_gnn'),
        layer_cnn=config.get('model.layer_cnn'),
        layer_output=config.get('model.layer_output'),
        dropout=config.get('model.dropout')
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    logger.info("Model loaded successfully")
    return model


def predict(model, dataset_test, batch_size, device):
    """
    Make predictions on test dataset.
    Simplified version for Streamlit app - only returns predictions and labels.
    
    Args:
        model: Trained model
        dataset_test: Test dataset
        batch_size: Batch size for prediction
        device: Device to run on
        
    Returns:
        Tuple of (predictions, labels)
    """
    model.eval()
    predictions = []
    labels = []
    
    N = len(dataset_test)
    i = 0
    fingerprints, pgraphs, padjacencys, graphs, morgans, adjs, sizes, label_list, waters = (
        [], [], [], [], [], [], [], [], []
    )
    
    with torch.no_grad():
        for data in dataset_test:
            i += 1
            fingerprint, pgraph, padjacency, graph, morgan, adj, size, label, water = data
            fingerprints.append(fingerprint)
            pgraphs.append(pgraph)
            padjacencys.append(padjacency)
            graphs.append(graph)
            morgans.append(morgan)
            adjs.append(adj)
            sizes.append(size)
            label_list.append(label)
            waters.append(water)
            
            if i % batch_size == 0 or i == N:
                # Pack batch
                packed = pack(
                    fingerprints, pgraphs, padjacencys, graphs,
                    morgans, adjs, sizes, label_list, waters, device
                )
                fingerprints1, pgraphs1, padjacencys1, graphs1, morgans1, adjs1, sizes1, labels1, waters1 = packed
                data_batch = (fingerprints1, pgraphs1, padjacencys1, graphs1, morgans1, adjs1, sizes1, labels1, waters1)
                
                # Forward pass
                outputs = model(data_batch)
                
                # Store results
                outputs_list = outputs.detach().cpu().numpy().flatten().tolist()
                predictions.extend(outputs_list)
                labels.extend(label_list)
                
                # Reset batch
                fingerprints, pgraphs, padjacencys, graphs, morgans, adjs, sizes, label_list, waters = (
                    [], [], [], [], [], [], [], [], []
                )
    
    return predictions, labels


def streamlit_app():


    st.write("""
    <style>
        .centered-text {
            text-align: center;
        }
    </style>

    <h1 class="centered-text">Use MPAP! 😎</h1>

    """, unsafe_allow_html=True)
    st.write("")
    st.image(get_image_path("fig1.png"))


    st.markdown("""## 1.Microplastics type""")
    microplastic_type = st.selectbox(" ", ["Polyamide (PA)", "Polyethylene (PE)","Polypropylene (PP)","Polyethylene terephthalate (PET)","Polystyrene (PS)","Polyvinyl chloride (PVC)"])
    st.markdown("""## 2.MP particle size (µm)""")
    microplastics_size = st.text_input(" ")
    st.markdown("""## 3.Water environment""")
    water_type = st.selectbox(" ", ["fresh water", "pure water","sea water"])
    st.markdown("""## 4.Compound SMILES""")
    op_smiles = st.text_input(".")
    st.write('<p style="color:blue;">You can convert molecular structure files (e.g., `.mol`, `.sdf`) into SMILES using open-source cheminformatics libraries (such as **RDKit**). Alternatively, if the compound name or **CAS number** is known, its SMILES notation can be retrieved via open-source tools.</p>',unsafe_allow_html=True)
    st.write('<p style="color:red;">If you need to batch process the dataset, please contact us: yangxihe@zju.edu.cn  / xianliu@rcees.ac.cn</p>',unsafe_allow_html=True)
    if st.button('5.Run Prediction'):
        if microplastic_type == "Polyamide (PA)":
            psmiles='[*]NCCCCCCCC(=O)[*]'
            category='PA'
            poly_smiles='NCCCCCCCC(=O)'
        elif microplastic_type == "Polyethylene (PE)":
            psmiles='[*]C=C[*]'
            category='PE'
            poly_smiles='C=C'
        elif microplastic_type == "Polypropylene (PP)":
            psmiles='[*]CC([*])C'
            category='PP'
            poly_smiles='CC=C'
        elif microplastic_type == "Polyethylene terephthalate (PET)":
            psmiles='[*]OCCOC(=O)c1ccc(C([*])=O)cc1'
            category='PET'
            poly_smiles='OCCOC(=O)c1ccc(C=O)cc1'
        elif microplastic_type == "Polystyrene (PS)":
            psmiles='[*]C=C[*]c1ccccc1'
            category='PS'
            poly_smiles='C=CC1=CC=CC=C1'
        elif microplastic_type == "Polyvinyl chloride (PVC)":
            psmiles='[*]C=C[*]Cl'
            category='PVC'
            poly_smiles='C=CCl'
        compound='ops'
        logkd='1'
        if water_type == "fresh water":
            water3='1'
        elif water_type == "pure water":
            water3='2'
        elif water_type == "sea water":
            water3='3'
        # Prepare content
        content = f"{category}\t{psmiles}\t{compound}\t{op_smiles}\t{microplastics_size}\t{water3}\t{logkd}\t{poly_smiles}"
        # Add header row
        header = "category\tpsmiles\tcompound\tsmiles\taverage size\twater3\tlogkd\tpoly_smiles\n"

        # Save as txt file
        filename = "test_input.txt"
        with open(filename, "w", encoding="utf-8") as file:  # Add encoding parameter
            file.write(header)      # Write header row first
            file.write(content)     # Then write data row

    # Generate npy files
        test_input_dir = config.get('paths.test_input_dir')
        # Ensure test_input directory exists
        os.makedirs(test_input_dir, exist_ok=True)
        inputfile('test_input.txt', test_input_dir + '/')
        
    # Neural network model prediction
        # Load test dataset only (training data is not needed for prediction)
        logger.info(f"Loading test dataset from {test_input_dir}")
        dataset_test = load_dataset(test_input_dir, device, shuffle=False)
        logger.info(f"Test samples: {len(dataset_test)}")
        
        # Find best model automatically or use environment variable
        model_dir = config.get('paths.model_dir')
        model_path = os.getenv('MPAP_MODEL_PATH', None)
        if model_path is None:
            logger.info(f"Searching for best model in {model_dir}...")
            model_path = find_best_model(model_dir)
        else:
            logger.info(f"Using model specified by MPAP_MODEL_PATH: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model using load_model function (similar to prediction.py)
        model = load_model(model_path, config, device)
        
        # Make predictions
        batch_size = config.get('training.batch_size', 128)
        outputs, labels = predict(model, dataset_test, batch_size, device)
        preds=pd.DataFrame({'labels':labels,'preds':outputs})
        
        # Save predictions to output directory
        output_dir = config.get('paths.output_dir')
        os.makedirs(output_dir, exist_ok=True)
        preds.to_csv(os.path.join(output_dir, 'test_preds.txt'), sep='\t')
        st.markdown("<h3 style='text-align: left;'>Result log<sub>10</sub>K<sub>d</sub></h3>", unsafe_allow_html=True)
        st.text_area(f" ", value=str(round(outputs[0], 2)), height=100)
        # Display prediction results in text area
        

        
if __name__ == '__main__':
    streamlit_app()

