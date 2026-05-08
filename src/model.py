"""Model wrappers used by the zero-shot retrieval pipeline."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

from i3d import InceptionI3D, clean_i3d_state_dict


EPS = 1e-8
DEFAULT_ALIGNMENT_MODEL = "google/siglip2-base-patch16-224"


def default_i3d_checkpoint(project_root):
    return Path(project_root) / "outputs" / "models" / "i3d" / "rgb_imagenet.pt"


def l2_normalize_numpy(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, EPS)


class SigLIPEncoder:
    """Frozen SigLIP2 encoder used for snippet-query correlation."""

    def __init__(self, model_name=DEFAULT_ALIGNMENT_MODEL, device=None, image_batch_size=32):
        self.name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_batch_size = int(image_batch_size)
        self.processor = self._load_processor(model_name)
        self.model = self._load_model(model_name).to(self.device)
        self.model.eval()

    def _load_processor(self, model_name):
        # Prefer the local Hugging Face cache so notebooks keep working offline
        # after the model has been downloaded once.
        try:
            return AutoProcessor.from_pretrained(model_name, local_files_only=True)
        except Exception as error:
            try:
                return AutoProcessor.from_pretrained(model_name)
            except Exception:
                raise error

    def _load_model(self, model_name):
        # Same local-first strategy as the processor loader.
        try:
            return AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception as error:
            try:
                return AutoModel.from_pretrained(model_name)
            except Exception:
                raise error

    def _to_feature_tensor(self, output):
        # Transformers model outputs changed across SigLIP/SigLIP2 versions.
        # This keeps the wrapper stable across those output containers.
        if torch.is_tensor(output):
            return output
        for attr in ("image_embeds", "text_embeds", "pooler_output"):
            value = getattr(output, attr, None)
            if value is not None:
                return value
        if getattr(output, "last_hidden_state", None) is not None:
            return output.last_hidden_state.mean(dim=1)
        raise TypeError(f"Unsupported model output type: {type(output)!r}")

    @torch.inference_mode()
    def encode_text(self, texts):
        """Return one normalized text embedding per query/sub-query."""
        inputs = self.processor(
            text=texts,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        features = self._to_feature_tensor(self.model.get_text_features(**inputs))
        features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy()

    @torch.inference_mode()
    def encode_frames(self, frames):
        """Encode PIL frames in batches to avoid GPU/CPU memory spikes."""
        batches = []
        for start in range(0, len(frames), self.image_batch_size):
            batch_frames = frames[start : start + self.image_batch_size]
            inputs = self.processor(
                images=batch_frames,
                return_tensors="pt",
            ).to(self.device)
            features = self._to_feature_tensor(self.model.get_image_features(**inputs))
            features = torch.nn.functional.normalize(features, dim=-1)
            batches.append(features.cpu().numpy())
        return np.concatenate(batches, axis=0) if batches else np.empty((0, 0), dtype=np.float32)

    def encode_snippets(self, snippets):
        """Average SigLIP2 frame embeddings into one alignment feature per snippet."""
        flat_frames = []
        snippet_lengths = []
        for frames in snippets:
            snippet_lengths.append(len(frames))
            flat_frames.extend(frames)

        frame_features = self.encode_frames(flat_frames)
        snippet_features = []
        offset = 0
        for length in snippet_lengths:
            local = frame_features[offset : offset + length]
            offset += length
            snippet_features.append(local.mean(axis=0))
        return l2_normalize_numpy(np.asarray(snippet_features))


class I3DFeatureExtractor:
    """Real Inception-I3D feature extractor for snippet-level visual features."""

    def __init__(self, checkpoint_path, num_classes=400, device=None, batch_size=4):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.num_classes = int(num_classes)
        self.batch_size = int(batch_size)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"i3d:{self.checkpoint_path}"

        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "A real I3D checkpoint is required. Put it at "
                "outputs/models/i3d/rgb_imagenet.pt or pass i3d_checkpoint."
            )

        self.model = InceptionI3D(num_classes=self.num_classes, in_channels=3, final_endpoint="Mixed_5c")
        self._load_checkpoint(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoints saved either as raw state_dict or wrapped dict."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        state_dict = clean_i3d_state_dict(checkpoint)
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        loaded_count = len(state_dict) - len(incompatible.unexpected_keys)
        if loaded_count < 20:
            raise RuntimeError(
                f"The checkpoint at {checkpoint_path} does not look compatible with Inception-I3D. "
                f"Loaded only {loaded_count} tensors."
            )
        if incompatible.missing_keys:
            print(f"I3D checkpoint missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"I3D checkpoint unexpected keys ignored: {len(incompatible.unexpected_keys)}")

    @torch.inference_mode()
    def extract(self, snippets):
        """Return one 1024-d I3D feature for each sampled video snippet."""
        tensors = [preprocess_i3d_clip(frames) for frames in snippets]
        features = []
        for start in range(0, len(tensors), self.batch_size):
            batch = torch.stack(tensors[start : start + self.batch_size]).to(self.device)
            output = self.model.extract_features(batch)
            output = F.adaptive_avg_pool3d(output, output_size=1).flatten(1)
            output = torch.nn.functional.normalize(output, dim=-1)
            features.append(output.cpu().numpy())
        return l2_normalize_numpy(np.concatenate(features, axis=0))


def preprocess_i3d_clip(frames, size=224, min_frames=16):
    """Convert PIL frames to the RGB input format expected by the I3D checkpoint."""
    if len(frames) < min_frames:
        frames = list(frames) + [frames[-1]] * (min_frames - len(frames))

    arrays = []
    for frame in frames:
        array = np.asarray(frame).astype(np.float32) / 127.5 - 1.0
        arrays.append(array)
    clip = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    clip = F.interpolate(clip, size=(size, size), mode="bilinear", align_corners=False)
    return clip.permute(1, 0, 2, 3).contiguous()


def build_visual_feature_extractor(
    i3d_checkpoint,
    i3d_num_classes=400,
    i3d_batch_size=4,
    device=None,
):
    return I3DFeatureExtractor(
        checkpoint_path=i3d_checkpoint,
        num_classes=i3d_num_classes,
        batch_size=i3d_batch_size,
        device=device,
    )


def cosine_scores(image_features, text_feature):
    return np.asarray(image_features) @ np.asarray(text_feature)
