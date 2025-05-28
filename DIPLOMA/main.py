import argparse
import sys
import os

def run_training():
    import train_own_dataset # assumes you have train.py with a main() function
    train_own_dataset.main()

def run_evaluation():
    import dir_class_predit  # assumes you have dir_class_predit.py with a main() function
    dir_class_predit.main()

def run_dataset_creation():
    import dataset_generator  # assumes you have dataset_generator.py with a main() function
    dataset_generator.main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select mode: train, eval, or create_dataset")
    parser.add_argument("mode", choices=["train", "eval", "create_dataset"], help="Mode to run")
    args = parser.parse_args()

    if args.mode == "train":
        run_training()
    elif args.mode == "eval":
        run_evaluation()
    elif args.mode == "create_dataset":
        run_dataset_creation()
    else:
        print("Unknown mode")
        sys.exit(1)