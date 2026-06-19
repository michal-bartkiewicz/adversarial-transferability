import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """
    Performs Fast Gradient Sign Method (FGSM) attack.

    Args:
        image (torch.Tensor): Original clean image.
        epsilon (float): Perturbation budget.
        data_grad (torch.Tensor): Gradient of the loss with respect to the input image.

    Returns:
        torch.Tensor: Perturbed image.
    """
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * data_grad.sign()
    
    # Tiny ImageNet was normalized with ImageNet stats. 
    # For safety, we clamp to roughly [min, max] range of normalized images.
    # Standard normalization range is roughly [-2.1, 2.6].
    return torch.clamp(perturbed_image, -3, 3)

def pgd_attack(
    model: nn.Module, 
    images: torch.Tensor, 
    labels: torch.Tensor, 
    epsilon: float = 0.05, 
    alpha: float = 0.005, 
    iters: int = 10, 
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Performs Projected Gradient Descent (PGD) attack (Adversarial Evasion).

    Args:
        model (nn.Module): Target or surrogate model.
        images (torch.Tensor): Original clean images.
        labels (torch.Tensor): Ground truth labels.
        epsilon (float): Perturbation budget (max L-infinity norm).
        alpha (float): Step size for each iteration.
        iters (int): Number of iterations.
        device (str): Device to run the attack on.

    Returns:
        torch.Tensor: Perturbed images.
    """
    images = images.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    adv_images = images.clone().detach()
    
    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        model.zero_grad()
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        # Step and projection
        adv_images = adv_images.detach() + alpha * adv_images.grad.sign()
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=-3, max=3).detach()
        
    return adv_images

def evaluate_transferability(
    surrogate: nn.Module, 
    target: nn.Module, 
    loader: DataLoader, 
    epsilon: float = 0.03, 
    device: str = 'cpu',
    limit: int = 256
) -> Tuple[float, float, float]:
    """
    Evaluates transferability-based black-box attacks.

    Args:
        surrogate (nn.Module): The surrogate model used to generate adversarial examples.
        target (nn.Module): The target model whose vulnerability is being tested.
        loader (DataLoader): Validation data loader.
        epsilon (float): Perturbation budget.
        device (str): Device to run evaluation on.
        limit (int): Max samples to evaluate for speed.

    Returns:
        Tuple[float, float, float]: (Clean Acc, FGSM Transfer Acc, PGD Transfer Acc)
    """
    surrogate.to(device).eval()
    target.to(device).eval()
    
    clean_acc, fgsm_acc, pgd_acc, total = 0.0, 0.0, 0.0, 0.0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        batch_size = labels.size(0)
        total += batch_size
        
        # 1. Clean Accuracy
        with torch.no_grad():
            outputs = target(images)
            clean_acc += (outputs.argmax(1) == labels).sum().item()
        
        # 2. FGSM Transfer (Gradient-based Evasion)
        images.requires_grad = True
        outputs_surr = surrogate(images)
        loss = nn.CrossEntropyLoss()(outputs_surr, labels)
        surrogate.zero_grad()
        loss.backward()
        
        adv_fgsm = fgsm_attack(images, epsilon, images.grad)
        with torch.no_grad():
            fgsm_acc += (target(adv_fgsm).argmax(1) == labels).sum().item()
            
        # 3. PGD Transfer (Iterative Gradient-based Evasion)
        adv_pgd = pgd_attack(
            surrogate, images, labels, 
            epsilon=epsilon, alpha=epsilon/4, iters=10, 
            device=device
        )
        with torch.no_grad():
            pgd_acc += (target(adv_pgd).argmax(1) == labels).sum().item()
            
        if total >= limit:
            break
            
    return (clean_acc / total, fgsm_acc / total, pgd_acc / total)
