import argparse
import random
import os
from datasets import Dataset, Features, Value

def main(args):
    random.seed(4727)
    mini_dataset = ImageNetDataset(args.data_root, args.phase)
    generator_fn = gen_examples(mini_dataset)
    features = {"image": Value("string"), "label": Value("int32"), "class": Value("string")}
    features = Features(features)
    print(f"Creating dataset of size {len(mini_dataset)}...")
    mini_ds = Dataset.from_generator(generator_fn, features=features)
    print("Saving dataset...")
    mini_ds.save_to_disk(args.save_path)

def gen_examples(dataset):
    def fn():
        for sample in dataset:
            yield {"image": sample["image"], "label": sample["label"], "class": sample["class"]}
    return fn

class ImageNetDataset:
    def __init__(self, root_path: str = "/root/data/dataset/vision_general/ImageNet", phase: str = "train"):
        self.phase_dir = os.path.join(root_path, phase)
        classes = sorted(os.listdir(self.phase_dir))
        self.samples = []
        for i, cls in enumerate(classes):
            cls_dir = os.path.join(self.phase_dir, cls)
            images = [i for i in os.listdir(cls_dir) if i.endswith(".JPEG")]
            for img in images:
                self.samples.append([os.path.join(cls_dir, img), i, cls])
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i: int):
        return {"image": self.samples[i][0], "label": self.samples[i][1], "class": self.samples[i][2]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare an accelerate-style dataset for ImageNet Dataset")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the root directory containing the subdirs of samples")
    parser.add_argument("--phase", type=str, default="train", help="Phase of the dataset to use in ['train', 'val']")
    parser.add_argument("--save_path", type=str, default="./imagenet_dataset")
    args = parser.parse_args()
    main(args)