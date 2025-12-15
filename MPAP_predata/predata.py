"""
Refactored data preprocessing script using the new configuration and utility system.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import mpap package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Import new utilities
from mpap.config import Config
from mpap.utils import setup_logging, create_output_dir

# Import existing utilities
from graph_features import atom_features
from prints import getprints

import logging
logger = logging.getLogger(__name__)

# Bond type mapping
from collections import defaultdict
from rdkit.Chem.rdchem import BondType

BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)


def smile_to_graph(smile: str):
    """
    Convert SMILES string to graph representation.
    
    Args:
        smile: SMILES string
        
    Returns:
        Tuple of (node_features, adjacency_matrix)
    """
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        raise ValueError(f"Could not parse SMILES: {smile}")
    
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


def process_data(input_file: str, output_dir: str, config: Config):
    """
    Process input data file and generate numpy arrays.
    
    Args:
        input_file: Path to input text file
        output_dir: Directory to save output files
        config: Configuration object
    """
    logger.info(f"Reading input file: {input_file}")
    data = pd.read_csv(input_file, sep='\t')
    
    psmiles = data['psmiles']
    poly_smiles = data['poly_smiles']
    compound_smiles = data['smiles']
    labels = data['logkd'].to_list()
    sizes = data['average size'].to_list()
    waters = data['water3'].to_list()
    
    fingerprints = []
    pgraph = []
    morgan = []
    graph = []
    adjacencies = []
    p_adjs = []
    
    logger.info(f"Processing {len(psmiles)} samples...")
    
    # Process microplastic fingerprints
    for i, psmile in enumerate(psmiles):
        if i % 100 == 0:
            logger.info(f"Processing microplastics: {i}/{len(psmiles)}")
        
        finger = getprints(psmile)  # Use existing getprints function
        if finger is None:
            logger.warning(f"Could not generate fingerprint for psmiles: {psmile}")
            # Use zero vector as fallback
            finger = np.zeros(600)
        fingerprints.append(finger)
    
    # Process polymer graphs
    for i, polysmile in enumerate(poly_smiles):
        if i % 100 == 0:
            logger.info(f"Processing polymer graphs: {i}/{len(poly_smiles)}")
        
        try:
            p_atomfeatures, p_adj = smile_to_graph(polysmile)
            pgraph.append(p_atomfeatures)
            p_adjs.append(p_adj)
        except Exception as e:
            logger.error(f"Error processing polymer {i}: {e}")
            raise
    
    # Process compound graphs and Morgan fingerprints
    for i, smiles in enumerate(compound_smiles):
        if i % 100 == 0:
            logger.info(f"Processing compounds: {i}/{len(compound_smiles)}")
        
        try:
            # Graph features
            atom_features, adj = smile_to_graph(smiles)
            graph.append(atom_features)
            adjacencies.append(adj)
            
            # Morgan fingerprint
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Could not parse SMILES: {smiles}")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, config.get('data.morgan_fp_size'))
            npfp = np.array(list(fp.ToBitString())).astype('int8')
            morgan.append(npfp)
        except Exception as e:
            logger.error(f"Error processing compound {i}: {e}")
            raise
    
    # Create output directory
    output_dir = create_output_dir(output_dir)
    logger.info(f"Saving processed data to {output_dir}")
    
    # Save numpy arrays
    np.save(os.path.join(output_dir, 'fingerprints'), np.array(fingerprints, dtype=object))
    np.save(os.path.join(output_dir, 'pgraph'), np.array(pgraph, dtype=object))
    np.save(os.path.join(output_dir, 'padjs'), np.array(p_adjs, dtype=object))
    np.save(os.path.join(output_dir, 'morgan'), np.array(morgan, dtype=object))
    np.save(os.path.join(output_dir, 'graph'), np.array(graph, dtype=object))
    np.save(os.path.join(output_dir, 'adjacencies'), np.array(adjacencies, dtype=object))
    np.save(os.path.join(output_dir, 'label'), np.array(labels))
    np.save(os.path.join(output_dir, 'size'), np.array(sizes))
    np.save(os.path.join(output_dir, 'water'), np.array(waters))
    
    logger.info("Data processing completed successfully!")


def main():
    """Main entry point."""
    # Load configuration
    config_path = os.getenv('MPAP_CONFIG', None)
    config = Config(config_path)
    
    # Setup logging
    setup_logging(
        log_dir=config.get('logging.log_dir'),
        log_file=config.get('logging.log_file'),
        level=config.get('logging.level')
    )
    
    logger.info("Starting data preprocessing...")
    
    # Get input and output paths
    input_file = config.get('paths.predata_input')
    output_dir = config.get('paths.predata_output')
    
    # Allow override via command line or environment
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess MPAP data')
    parser.add_argument('--input', type=str, default=input_file, help='Input text file')
    parser.add_argument('--output', type=str, default=output_dir, help='Output directory')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Process data
    try:
        process_data(args.input, args.output, config)
    except Exception as e:
        logger.error(f"Data processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
