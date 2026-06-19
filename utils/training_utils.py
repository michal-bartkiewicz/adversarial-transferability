import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Union
from models.model_wrapper import ModelWrapper
from tqdm.auto import tqdm, trange

def train_model(
    model: ModelWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    device: str = 'cpu',
    teacher: Optional[ModelWrapper] = None,
    use_distillation: bool = False,
    temperature: float = 3.0,
    lr: float = 1e-4,
    max_lr: float = 5e-3,
    show_tqdm_bar = False,
):
    """
    Trains a model using standard training, teacher labels, or logit distillation.

    Args:
        model (ModelWrapper): The student/target model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Number of training epochs. Defaults to 5.
        device (str): Device to train on ('cuda', 'cpu', etc.).
        teacher (ModelWrapper, optional): Teacher model for distillation or labels.
        use_distillation (bool): If True, uses logit distillation (MSE). 
                                 If False and teacher is present, uses teacher hard labels.
        temperature (float): Temperature for smoothing logits. Defaults to 3.0.
        lr (float): Initial learning rate.
        max_lr (float): Max learning rate for OneCycleLR.
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.2,
        anneal_strategy='cos'
    )

    if teacher is not None:
        teacher.to(device).eval()

    if use_distillation and teacher is not None:
        criterion = nn.MSELoss()
        training_type = "logits"
    elif teacher is not None:
        criterion = nn.CrossEntropyLoss()
        training_type = "labels"
    else:
        criterion = nn.CrossEntropyLoss()
        training_type = "clean"

    print(f"Starting training: {model.name} ({training_type}) on {device}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # tqdm for batches
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) if show_tqdm_bar else train_loader
        
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_distillation and teacher is not None:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                student_logits = model(images)
                # MSE loss on softened logits
                loss = criterion(student_logits / temperature, teacher_logits / temperature)
            else:
                if teacher is not None:
                    with torch.no_grad():
                        labels = teacher(images).argmax(dim=1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if show_tqdm_bar:
                train_iter.set_postfix(loss=loss.item()) # pyright: ignore

        # Fast Validation
        val_acc = evaluate_accuracy(model, val_loader, device, limit=1000)
        print(f"Epoch {epoch+1}/{epochs}: Loss {running_loss/len(train_loader):.4f} | Val Acc {val_acc:.4f}")

    save_path = save_model_weights(model, training_type)
    print(f"Training complete. Weights saved to {save_path}")

def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: str, limit: Optional[int] = None) -> float:
    """
    Evaluates model accuracy on a given loader.

    Args:
        model (nn.Module): Model to evaluate.
        loader (DataLoader): Data loader.
        device (str): Device to run on.
        limit (int, optional): Max number of samples to evaluate.

    Returns:
        float: Accuracy in range [0, 1].
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            if limit and total >= limit:
                break
    return correct / total if total > 0 else 0.0

def save_model_weights(model: ModelWrapper, training_type: str, weights_dir: str = './weights', target_model: Optional[ModelWrapper] = None) -> str:
    """
    Saves model weights with a standardized naming convention.

    Args:
        model (ModelWrapper): Wrapped model.
        training_type (str): Type of training ('clean', 'labels', 'logits').
        weights_dir (str): Directory to save weights.

    Returns:
        str: Final save path.
    """
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    if (target_model is None) or (training_type == "clean"):
        filename = f"{model.name}_{training_type}.pth"
    else:
        filename = f"{model.name}_{training_type}_from_{target_model.name}.pth"
    save_path = os.path.join(weights_dir, filename)
    torch.save(model.state_dict(), save_path)
    return save_path

def load_model_weights(model: ModelWrapper, training_type: str, weights_dir: str = './weights', device: str = 'cpu', target_model: Optional[ModelWrapper] = None) -> bool:
    """
    Loads model weights if they exist.

    Args:
        model (ModelWrapper): Wrapped model.
        training_type (str): Type of training.
        weights_dir (str): Directory where weights are stored.
        device (str): Device to load weights to.

    Returns:
        bool: True if loaded, False otherwise.
    """
    if (target_model is None) or (training_type == "clean"):
        filename = f"{model.name}_{training_type}.pth"
    else:
        filename = f"{model.name}_{training_type}_from_{target_model.name}.pth"
    load_path = os.path.join(weights_dir, filename)
    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device))
        model.to(device)
        return True
    return False
