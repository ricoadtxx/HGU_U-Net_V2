import argparse
from train import train_model
from config import EPOCHS, BATCH_SIZE, VALID_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description="Training U-Net for semantic segmentation")
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--valid_threshold', type=float, default=VALID_THRESHOLD, help='Minimum valid data ratio per patch')
    
    args = parser.parse_args()

    print(f"Starting training with {args.epochs} epochs, batch size {args.batch_size}, and threshold {args.valid_threshold}")
    train_model(epochs=args.epochs, batch_size=args.batch_size, valid_threshold=args.valid_threshold)

if __name__ == '__main__':
    main()
