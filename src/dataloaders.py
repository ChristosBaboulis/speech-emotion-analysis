from torch.utils.data import DataLoader
from iemocap_dataset import IEMOCAPDataset

def create_dataloaders(train_samples, val_samples, test_samples, batch_size=32):
    train_dataset = IEMOCAPDataset(train_samples)
    val_dataset = IEMOCAPDataset(val_samples)
    test_dataset = IEMOCAPDataset(test_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
