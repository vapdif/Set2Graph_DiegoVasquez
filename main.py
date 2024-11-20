import argparse         # handle command line arguments.

from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.download_data import download_data

def main():
    parser = argparse.ArgumentParser(description="Jet Classification Pipeline")
    parser.add_argument("action", choices=["download", "train", "evaluate"], help="Pipeline step to execute")
    parser.add_argument("--data_path", default="./data/", help="Path to the dataset directory")
    parser.add_argument("--model_path", default="set2graph_model.pth", help="Path to save or load the model")
    args = parser.parse_args()

    if args.action == "download":
        download_data()
    elif args.action == "train":
        train_model(data_path=args.data_path, model_path=args.model_path)
    elif args.action == "evaluate":
        evaluate_model(data_path=args.data_path, model_path=args.model_path)

if __name__ == "__main__":
    main()
