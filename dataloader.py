import torch
import rasterio
import geopandas as gpd
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import RandomRotation

class SatelliteDataset(Dataset):
    def __init__(self, image_paths, metadata_paths, transform=None):
        self.image_paths = image_paths  # 위성영상 이미지 경로 목록
        self.metadata_paths = metadata_paths  # 메타데이터 경로 목록 (예: 위도, 경도, 시간)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 위성영상 읽기 (GeoTIFF)
        image_path = self.image_paths[idx]
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3, 4])  # 예시: 4개 채널 (RGB, NIR 등)
            image = np.moveaxis(image, 0, -1)  # (C, H, W) -> (H, W, C)로 변환
        
        # 메타데이터 읽기 (위도, 경도, 시간)
        metadata_path = self.metadata_paths[idx]
        metadata = gpd.read_file(metadata_path)
        lat = metadata['latitude'].values[0]
        lon = metadata['longitude'].values[0]
        timestamp = metadata['timestamp'].values[0]

        # 이미지 전처리 및 데이터 증강
        if self.transform:
            image = self.transform(image)

        # 이미지와 메타데이터를 함께 반환
        sample = {
            'image': torch.tensor(image, dtype=torch.float32),
            'metadata': {'latitude': lat, 'longitude': lon, 'timestamp': timestamp}
        }
        
        return sample

# 데이터셋 전처리 예시
transform = RandomRotation(degrees=30)

# 사용 예시
image_paths = ['image1.tif', 'image2.tif']
metadata_paths = ['meta1.json', 'meta2.json']
dataset = SatelliteDataset(image_paths, metadata_paths, transform=transform)

# DataLoader로 로딩
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
