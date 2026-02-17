
from typing import List, Optional
import io, requests
from PIL import Image
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

def _load_image_from_url(url: str) -> Optional[Image.Image]:
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None

class ImageTextRelevance:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model_name = model_name
        self.model = None

    def load(self):
        if SentenceTransformer is None:
            return
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception:
            self.model = None

    def score(self, image_urls: List[str], text: str) -> Optional[float]:
        if not image_urls or not text:
            return None
        if self.model is None and SentenceTransformer is not None:
            self.load()
        if self.model is None:
            return None
        ims = []
        for u in image_urls[:5]:
            im = _load_image_from_url(u)
            if im is not None:
                ims.append(im)
        if not ims:
            return None
        try:
            i_emb = self.model.encode(ims, convert_to_tensor=True, normalize_embeddings=True)
            t_emb = self.model.encode([text], convert_to_tensor=True, normalize_embeddings=True)[0]
            sims = (i_emb @ t_emb).cpu().numpy()
            sims = (sims + 1) / 2.0
            return float(np.mean(sims))
        except Exception:
            return None
