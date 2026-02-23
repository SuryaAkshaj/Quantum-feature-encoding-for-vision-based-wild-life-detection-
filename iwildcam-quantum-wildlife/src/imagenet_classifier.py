"""
imagenet_classifier.py
-----------------------
Uses a pre-trained ResNet-50 classification head to map image crops to one of
the 10 COCO animal classes without any fine-tuning. Works much better than the
MLP with small training data for clean/web photos.

Classes: bear, bird, cat, cow, dog, elephant, giraffe, horse, sheep, zebra
"""

import torch
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image

# ── ImageNet class index → our 10 target classes ─────────────────────────────
# Source: torchvision ImageNet 1000-class list (0-indexed)
IMAGENET_TO_COCO = {
    # bear (21 = black bear, 294 = brown bear, 295 = American black bear,
    #        296 = ice bear / polar bear, 297 = sloth bear)
    21: "bear", 294: "bear", 295: "bear", 296: "bear", 297: "bear",

    # bird — common ImageNet bird classes
    **{i: "bird" for i in list(range(7, 25)) + list(range(80, 100)) +
       [127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
        141, 142, 143, 144, 145, 146, 147, 148, 149]},

    # cat (domestic + wild felids)
    281: "cat", 282: "cat", 283: "cat", 284: "cat", 285: "cat",  # domestic
    286: "cat",  # cougar / puma
    287: "cat",  # lynx
    288: "cat",  # leopard
    289: "cat",  # snow leopard
    290: "cat",  # jaguar
    291: "cat",  # lion
    292: "cat",  # tiger
    293: "cat",  # cheetah

    # cow / bovine
    345: "cow",  # ox
    346: "cow",  # water buffalo
    347: "cow",  # bison
    349: "cow",  # bighorn (also sheep-like but mapped to cow for diversity)

    # dog — all dog breeds in ImageNet
    **{i: "dog" for i in range(151, 269)},
    275: "dog",  # Dingo
    276: "dog",  # African hunting dog

    # elephant
    385: "elephant", 386: "elephant", 387: "elephant",

    # giraffe
    366: "giraffe",

    # horse
    339: "horse",  # sorrel
    603: "horse",  # horse (n02374451)

    # sheep
    348: "sheep",  # ram (domestic)
    350: "sheep",  # ibex (wild sheep-like)

    # zebra
    340: "zebra",
}


def _build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class ImageNetClassifier:
    """
    Thin wrapper around a pre-trained ResNet-50 that returns one of the 10
    COCO animal class names (or None if no matching class is found in the top-K).
    """

    def __init__(self, top_k: int = 10, device: str = "cpu"):
        self.device    = torch.device(device)
        self.top_k     = top_k
        self.transform = _build_transform()

        # Full ResNet-50 with classification head
        self.model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def classify(self, img_path: str):
        """
        Returns (class_name, confidence) where class_name is one of the 10
        target classes, or (None, 0.0) if the top-K ImageNet predictions don't
        map to any of our target classes.
        """
        img = Image.open(img_path).convert("RGB")
        x   = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs  = F.softmax(logits, dim=1).squeeze(0)

        top_indices = probs.topk(self.top_k).indices.tolist()
        top_probs   = probs.topk(self.top_k).values.tolist()

        # Walk through top predictions; return first that maps to a target class
        for idx, prob in zip(top_indices, top_probs):
            coco_class = IMAGENET_TO_COCO.get(idx)
            if coco_class is not None:
                return coco_class, round(prob * 100, 1)

        return None, 0.0

    @torch.no_grad()
    def classify_all(self, img_path: str):
        """
        Returns a dict {class_name: probability%} for all 10 target classes,
        summing probabilities across all matching ImageNet classes.
        """
        TARGET_CLASSES = ["bear","bird","cat","cow","dog","elephant","giraffe","horse","sheep","zebra"]
        totals = {c: 0.0 for c in TARGET_CLASSES}

        img = Image.open(img_path).convert("RGB")
        x   = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs  = F.softmax(logits, dim=1).squeeze(0).tolist()

        for idx, prob in enumerate(probs):
            coco_class = IMAGENET_TO_COCO.get(idx)
            if coco_class in totals:
                totals[coco_class] += prob

        # Normalise to sum=1 over our 10 classes
        total_sum = sum(totals.values()) or 1e-9
        return {c: round(v / total_sum * 100, 1) for c, v in totals.items()}
