import torch
import argparse
from utils.dataset_utils import download_tiny_imagenet, get_dataloaders
from utils.training_utils import train_model, load_model_weights
from models.example_models import get_resnet18, get_efficientnet_v2_s
from attacks.evasion_attacks import evaluate_transferability

def main():
    parser = argparse.ArgumentParser(description="Transferability-based Black-box Attacks on Tiny ImageNet")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Attack perturbation budget")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory for dataset")
    args = parser.parse_args()

    # Hardware Agnostic device selection
    device = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else \
             'cuda' if torch.cuda.is_available() else \
             'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Dataset Setup
    download_tiny_imagenet(args.data_dir)
    target_loader, surrogate_loader, val_loader = get_dataloaders(
        data_dir=f"{args.data_dir}/tiny-imagenet-200", 
        batch_size=args.batch_size
    )

    # 2. Model Initialization
    # Target: EfficientNet-V2-S (Stronger)
    # Surrogate: ResNet-18 (Lighter, used for gradient generation)
    target_model = get_efficientnet_v2_s(pretrained=True)
    surrogate_model = get_resnet18(pretrained=True)
    train_type = "labels"  # Surrogate training type (using teacher labels from target)

    # 3. Training / Loading Weights
    n_epochs = args.epochs
    # Load or train target model (Standard training)
    if not load_model_weights(target_model, "clean", device=device):
        print("\nTraining Target Model...")
        train_model(
            target_model,
            target_loader,
            val_loader,
            epochs=n_epochs,
            device=device,
        )
    else:
        print(f"\nLoaded Target Model: {target_model.name}")

    # Load or train surrogate model (using teacher labels from target)
    if not load_model_weights(surrogate_model, train_type, device=device, target_model=target_model):
        print("\nTraining Surrogate Model with Teacher Labels...")
        train_model(
            surrogate_model, surrogate_loader, val_loader, 
            epochs=n_epochs, device=device, 
            teacher=target_model,
        )
    else:
        print(f"\nLoaded Surrogate Model: {surrogate_model.name} trained with {train_type} from {target_model.name}")

    # 4. Evaluation
    print(f"\n--- Evaluating Gradient-based Adversarial Evasion Attacks (Epsilon={args.epsilon}) ---")
    
    # White-box: Attack Target using Target's own gradients
    print("Running White-box Evaluation...")
    wb_clean, wb_fgsm, wb_pgd = evaluate_transferability(
        target_model, target_model, val_loader, 
        epsilon=args.epsilon, device=device
    )
    
    # Black-box: Attack Target using Surrogate's gradients
    print("Running Black-box (Transfer) Evaluation...")
    bb_clean, bb_fgsm, bb_pgd = evaluate_transferability(
        surrogate_model, target_model, val_loader, 
        epsilon=args.epsilon, device=device
    )

    print("\nFinal Results:")
    print(f"Target Clean Accuracy: {wb_clean*100:.2f}%")
    print("-" * 30)
    print("White-box Attack Accuracy (Direct):")
    print(f"  FGSM: {wb_fgsm*100:.2f}%")
    print(f"  PGD:  {wb_pgd*100:.2f}%")
    print("-" * 30)
    print("Black-box Attack Accuracy (Transfer from Surrogate):")
    print(f"  FGSM: {bb_fgsm*100:.2f}%")
    print(f"  PGD:  {bb_pgd*100:.2f}%")

if __name__ == "__main__":
    main()
