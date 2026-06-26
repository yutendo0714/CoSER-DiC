from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(root: str | Path) -> list[Path]:
    path = Path(root)
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    image_id: str


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        recursive: bool = True,
    ) -> None:
        self.root = Path(root)
        if recursive:
            paths = list_images(self.root)
        else:
            paths = sorted(p for p in self.root.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        self.records = [ImageRecord(path=p, image_id=p.relative_to(self.root).as_posix()) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = Image.open(record.path).convert("RGB")
        width, height = image.size
        tensor = TF.to_tensor(image)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return {
            "image": tensor,
            "path": str(record.path),
            "id": record.image_id,
            "height": height,
            "width": width,
        }

