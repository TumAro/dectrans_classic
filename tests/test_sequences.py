import pickle
from src.dataset import create_sequences

def test_create_sequences():
    """Test that sequences are created correctly"""
    # Load dataset
    with open("data/dataset.pkl", "rb") as f:
        episodes = pickle.load(f)
    
    # Create sequences
    sequences = create_sequences(episodes, seq_len=20)
    
    # Basic checks
    assert len(sequences) > 0, "Should create at least one sequence"
    
    # Check first sequence structure
    seq = sequences[0]
    assert 'states' in seq
    assert 'actions' in seq
    assert 'rtg' in seq
    assert 'mask' in seq
    
    # Check lengths are all 20
    assert len(seq['states']) == 20
    assert len(seq['actions']) == 20
    assert len(seq['rtg']) == 20
    assert len(seq['mask']) == 20
    
    # Check mask has correct values
    assert all(m in [0, 1] for m in seq['mask']), "Mask should only have 0s and 1s"
    
    print(f"✓ Created {len(sequences)} sequences from {len(episodes)} episodes")
    print(f"✓ First sequence has correct structure and length")