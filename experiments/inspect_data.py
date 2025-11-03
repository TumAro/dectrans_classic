import pickle

def inspect_data():
    with open("data/dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    rtgs = [ep['rtg'][0] for ep in dataset]
    print(f"RTG Range: [{min(rtgs)} to {max(rtgs)}]")
    print(f"Average RTG: {sum(rtgs)/len(rtgs):.1f}")

    print("======================")

    lengths = [len(ep['states']) for ep in dataset]
    print(f"Episode length range: [{min(lengths)} to {max(lengths)}]")
    print(f"Average length: {sum(lengths)/len(lengths):.1f}")