"""Identical catalog/query preprocessing and validated float32 OpenCLIP vectors."""
import math
import struct
from importlib.metadata import version

MODEL_NAME = 'ViT-B-32'
PRETRAINED = 'laion2b_s34b_b79k'
DIMENSIONS = 512
PIPELINE_VERSION = 1


class ImageReadError(ValueError):
    pass


def device_for_run(requested='auto', torch_module=None):
    if torch_module is None:
        import torch as torch_module
    mps = getattr(torch_module.backends, 'mps', None)
    available = {'cpu': True, 'cuda': torch_module.cuda.is_available(),
                 'mps': bool(mps and mps.is_available())}
    if requested == 'auto':
        return next(device for device in ('cuda', 'mps', 'cpu') if available[device])
    if requested not in available or not available[requested]:
        raise ValueError(f'Device {requested!r} is unavailable; use --device cpu or auto')
    return requested


def validate_embedding(values):
    if len(values) != DIMENSIONS or not all(math.isfinite(v) for v in values):
        raise ValueError(f'Embedding must contain {DIMENSIONS} finite coordinates')
    if abs(math.sqrt(math.fsum(v * v for v in values)) - 1.0) > 1e-5:
        raise ValueError('Embedding must have unit L2 norm')
    return values


def normalize_embedding(values):
    if len(values) != DIMENSIONS or not all(math.isfinite(v) for v in values):
        raise ValueError(f'Model must produce {DIMENSIONS} finite coordinates')
    norm = math.sqrt(math.fsum(v * v for v in values))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('Model produced a zero or invalid embedding')
    # Explicit IEEE float32 conversion, independent of the host's Python float type.
    vector = list(struct.unpack(f'<{DIMENSIONS}f', struct.pack(f'<{DIMENSIONS}f', *(v / norm for v in values))))
    return validate_embedding(vector)


class ClipEncoder:
    def __init__(self, device='auto'):
        try:
            import open_clip
            import torch
            from PIL import Image, ImageOps
        except ImportError as exc:
            raise RuntimeError('Embedding dependencies are missing. Run: '
                               'python -m pip install -r requirements-embeddings.txt') from exc
        self.torch, self.Image, self.ImageOps = torch, Image, ImageOps
        self.device = device_for_run(device, torch)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, precision='fp32')
        self.model = self.model.to(self.device).eval()
        self.signature = {name: version(name) for name in ('torch', 'torchvision', 'open_clip_torch', 'Pillow')}

    def prepare_image(self, path):
        try:
            with self.Image.open(path) as image:
                return self.preprocess(self.ImageOps.exif_transpose(image).convert('RGB'))
        except (OSError, ValueError, self.Image.DecompressionBombError) as exc:
            raise ImageReadError(f'{path}: {exc}') from exc

    def encode_batch(self, images):
        batch = self.torch.stack(images).to(self.device)
        with self.torch.inference_mode():
            features = self.model.encode_image(batch).float()
            if tuple(features.shape) != (len(images), DIMENSIONS):
                raise ValueError(f'Unexpected model output shape: {tuple(features.shape)}')
            values = features.cpu().tolist()
        return [normalize_embedding(vector) for vector in values]
