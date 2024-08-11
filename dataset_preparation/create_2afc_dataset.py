import argparse
import random
import os
from datasets import Dataset, Features, Value

def main(args):
    random.seed(4727)
    mini_dataset = BAPPS2AFCDataset(args.data_root, args.phase)
    generator_fn = gen_examples(mini_dataset)
    features = {"ref": Value("string"), "p0": Value("string"), "p1": Value("string"), "judge": Value("string")}
    features = Features(features)
    print(f"Creating dataset of size {len(mini_dataset)}...")
    mini_ds = Dataset.from_generator(generator_fn, features=features)
    print("Saving dataset...")
    mini_ds.save_to_disk(args.save_path)

def gen_examples(dataset):
    def fn():
        for sample in dataset:
            yield {"ref": sample["ref"], "p0": sample["p0"], "p1": sample["p1"], "judge": sample["judge"]}
    return fn

class BAPPS2AFCDataset:
    def __init__(self, root_path: str = "/root/data/dataset/vision_general/bapps/2afc", phase: str="train"):
        if phase == "train":
            subdirs = ["cnn", "mix", "traditional"]
        elif phase == "val":
            subdirs = ["cnn", "color", "deblur", "frameinterp", "superres", "traditional"]
        elif phase == "val_traditional":
            subdirs = ["traditional"]
            phase = "val"
        elif phase == "val_cnn":
            subdirs = ["cnn"]
            phase = "val"
        elif phase == "val_real":
            subdirs = ["color", "deblur", "frameinterp", "superres"]
            phase = "val"
        else:
            raise ValueError(f"Phase {phase} not recognized. Must be one of ['train', 'val', 'val_traditional', 'val_cnn', 'val_real']")
        self.phase_dir = os.path.join(root_path, phase)
        self.ref_samples = []
        for i, subdir in enumerate(subdirs):
            ref_subdir_path = os.path.join(self.phase_dir, subdir, "ref")
            self.ref_samples += [os.path.join(ref_subdir_path, i) for i in os.listdir(ref_subdir_path) if i.endswith(".png")]
        random.shuffle(self.ref_samples)
    
    def __len__(self):
        return len(self.ref_samples)

    def __getitem__(self, i: int):
        ref = self.ref_samples[i]
        p0 = self.ref_samples[i].replace("ref", "p0")
        p1 = self.ref_samples[i].replace("ref", "p1")
        judge = self.ref_samples[i].replace("ref", "judge").replace(".png", ".npy")
        return {"ref": ref, "p0": p0, "p1": p1, "judge": judge}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare an accelerate-style dataset for BAPPS 2afc Dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root directory containing the subdirs of samples")
    parser.add_argument("--phase", type=str, default="train", help="Phase of the dataset to use in ['train', 'val']")
    parser.add_argument("--save_path", type=str, default="./2afc_dataset_train", help="Path to save the dataset")
    args = parser.parse_args()
    main(args)