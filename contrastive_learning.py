import requests
import torch
import os

from PIL import Image
from pathlib import Path
from prismatic import load
from prismatic.models.backbones.vision.vggt import VGGTBackbone
from datasets import load_dataset


def load_models(model_id="dinov2-224px+7b", hf_token=""):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub)
    vlm = load(model_id, hf_token=hf_token)

    # Extract the vision backbone and projector from the VLM
    vision_backbone = vlm.vision_backbone.to(device)
    vision_backbone.eval()
    vision_projector = vlm.vision_projector.to(device)
    vision_projector.eval()

    vggt_backbone = vlm.vggt_backbone.to(device)
    vggt_backbone.eval()
    vggt_projector = vlm.vggt_projector.to(device)
    return vision_backbone, vision_projector, vggt_backbone, vggt_projector

def get_dataloaders():
    raise NotImplementedError("get_dataloaders() not implemented yet")

def train_vggt_projector(vision_backbone, vision_projector, vggt_backbone, vggt_projector, train_dataloader, val_dataloader, device):
    vision_backbone.eval()
    vision_projector.eval()
    vggt_backbone.eval()
    vggt_projector.train()

    total_loss = 0

    for batch in train_dataloader:
        images = batch['image'].to(device)

        with torch.no_grad():
            feats1 = vision_backbone(images)
            proj1 = vision_projector(feats1)
        feats2 = vggt_backbone(images)
        proj2 = vggt_projector(feats2)

        loss = contrastive_loss(proj1, proj2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Train Loss: {avg_loss:.4f}")

def epoch_step(vision_backbone, vision_projector, vggt_backbone, vggt_projector, train_dataloader, val_dataloader, device):
    total_loss = 0

    for batch in train_dataloader:
        images = batch['image'].to(device)

        with torch.no_grad():
            feats1 = vision_backbone(images)
            proj1 = vision_projector(feats1)
        feats2 = vggt_backbone(images)
        proj2 = vggt_projector(feats2)

        loss = contrastive_loss(proj1, proj2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Train Loss: {avg_loss:.4f}")

def contrastive_loss(z1, z2, temperature=0.07, loss_type="SimCLR"):
    if loss_type == "CLIP":
        # Normalize
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        logits = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        loss = criterion(logits, labels)
        return loss
    elif loss_type == "SimCLR":
        # Normalize
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        logits = torch.matmul(z1, z2.T) / temperature
        labels = torch.arange(z1.size(0)).to(z1.device)
        loss = criterion(logits, labels)
        return loss



if __name__ == "__main__":

    vision_backbone, vision_projector, vggt_backbone, vggt_projector = load_models()
    vggt_projector.train()
    #TODO: add image dataset to train
    train_dataloader, val_dataloader = get_dataloaders()

    # Freeze all parameters except vggt_projector
    for param in vision_backbone.parameters():
        param.requires_grad = False
    for param in vision_projector.parameters():
        param.requires_grad = False
    for param in vggt_backbone.parameters():
        param.requires_grad = False
    for param in vggt_projector.parameters():
        param.requires_grad = True

    # Define optimizer for vggt_projector only
    optimizer = torch.optim.Adam(vggt_projector.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()


    num_epochs = 10
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for epoch in range(num_epochs):
        vggt_projector.train()
        total_loss = 0
        for images in train_dataloader:
            # Assume batch is a dict with 'image' key, adjust as needed
            images = images.to(device)

            with torch.no_grad():
                feats1 = vision_backbone(images)
                proj1 = vision_projector(feats1)
            feats2 = vggt_backbone(images)
            proj2 = vggt_projector(feats2)

            loss = contrastive_loss(proj1, proj2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
