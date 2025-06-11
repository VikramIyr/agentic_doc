# orchestrator.py

import os
import yaml
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import layoutparser as lp
from transformers import VisionEncoderDecoderModel, AutoProcessor
from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering
from PIL import Image


class AgenticOrchestrator:
    def __init__(self, manifest_path: str):
        # 1) Load your manifest
        with open(manifest_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # 2) Layout detection via PubLayNet (LayoutParser)
        self.layout_model = lp.Detectron2LayoutModel(
            config_path  = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            model_path   = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model",
            label_map    = {0:"Text", 1:"Title", 2:"List", 3:"Table", 4:"Figure"},
            extra_config = [
                "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
                str(self.cfg["layout"]["params"]["confidence_threshold"])
            ]
        )

        # 3) Donut “vision-first” parser
        self.donut_processor = AutoProcessor.from_pretrained(
            "naver-clova-ix/donut-base", trust_remote_code=True
        )
        self.donut_model = VisionEncoderDecoderModel.from_pretrained(
            "naver-clova-ix/donut-base", trust_remote_code=True
        )

        # 4) LayoutLMv3 fallback for text-first parsing
        self.llm3_proc  = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        self.llm3_model = LayoutLMv3ForQuestionAnswering.from_pretrained(
            "microsoft/layoutlmv3-base"
        )

    def extract(self, doc_path: str) -> dict:
        img     = self._load_image(doc_path)
        layouts = self.layout_model.detect(img)

        result = {"fields": {}, "warnings": []}
        for region in layouts:
            x1, y1, x2, y2 = map(int, region.coordinates)
            crop = img[y1:y2, x1:x2]

            if region.type == "Table":
                parsed = self._parse_with_donut(crop)
                result["fields"].setdefault("tables", []).append(parsed)
            else:
                words, boxes = self._ocr(crop)
                kv = self._parse_with_layoutlm(
                    crop, words, boxes, "Extract all key-value pairs."
                )
                result["fields"].setdefault("kv_pairs", []).append(kv)

        return result

    def _load_image(self, path: str) -> np.ndarray:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            pages = convert_from_path(path, dpi=300)
            pil   = pages[0].convert("RGB")
            img   = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image from {path}")
        return img

    def _ocr(self, crop: np.ndarray):
        data = pytesseract.image_to_data(
            crop,
            lang=self.cfg["ocr"]["params"]["lang"],
            config=f"--psm {self.cfg['ocr']['params']['psm']}",
            output_type=pytesseract.Output.DICT
        )
        words = data["text"]
        h, w  = crop.shape[:2]
        boxes = [
            [
                data["left"][i]/w,
                data["top"][i]/h,
                (data["left"][i] + data["width"][i])/w,
                (data["top"][i]  + data["height"][i])/h
            ]
            for i in range(len(words))
        ]
        return words, boxes

    def _parse_with_donut(self, crop: np.ndarray) -> str:
        pil    = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        inputs = self.donut_processor(images=pil, return_tensors="pt")
        outputs = self.donut_model.generate(
            **inputs,
            max_length=self.cfg["parser"][0]["params"]["max_length"]
        )
        return self.donut_processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def _parse_with_layoutlm(self, crop, words, boxes, question: str) -> str:
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        enc = self.llm3_proc(images=pil, words=words, boxes=boxes, return_tensors="pt")
        out = self.llm3_model(**enc, questions=[question])
        return out.answers[0].answer
