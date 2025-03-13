import torch
import torch.optim as optim
from models.encoder import STMAViTEncoder
from models.decoders.segmentation import SegmentationDecoder

def finetune(model, dataloader, config):
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["finetune"]["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["finetune"]["epochs"])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config["training"]["finetune"]["epochs"]):
        model.train()
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{config['training']['finetune']['epochs']}, Loss: {running_loss/len(dataloader)}")

if __name__ == "__main__":
    config = "configs/config.json"
    model = STMAViTEncoder(config)
    decoder = SegmentationDecoder(num_classes=10)
    # finetune(model, train_loader, config)
