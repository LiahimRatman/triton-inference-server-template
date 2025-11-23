import os
import logging
from typing import List

import torch
import numpy as np
from transformers import AutoModel
import triton_python_backend_utils as pb_utils

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

MODEL_NAME = "jinaai/jina-embeddings-v4"
# По умолчанию 2048, как в доке Jina v4
DIM = int(os.getenv("JINA_EMB_V4_DIM", "2048"))


class TritonPythonModel:
    def initialize(self, args):
        """
        Вызывается один раз при загрузке модели Triton.
        Здесь грузим HF-модель и переносим на GPU/CPU.
        """
        device_id = args.get("model_instance_device_id", "0")

        if torch.cuda.is_available():
            # Triton обычно проставляет model_instance_device_id
            self.device = torch.device(f"cuda:{device_id}")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")

        LOGGER.info(f"[{MODEL_NAME}] Initializing on device: {self.device}")

        # Загрузка модели с trust_remote_code, как в README HF
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

        LOGGER.info(f"[{MODEL_NAME}] Model loaded and ready")

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Обертка над model.encode_text -> numpy [batch, DIM]
        """
        LOGGER.info(f"[{MODEL_NAME}] _encode_texts called, num_texts={len(texts)}")

        with torch.no_grad():
            embeddings = self.model.encode_text(
                texts,
                task="retrieval",
                prompt_name="passage",
                truncate_dim=DIM,
            )

        # embeddings может быть:
        # - Tensor [batch, DIM]
        # - list[Tensor [DIM]] (типичный случай)
        # - в теории что-то numpy-подобное

        # 1) Tensor
        if torch.is_tensor(embeddings):
            t = embeddings.detach()
            if t.is_cuda:
                t = t.cpu()
            arr = t.float().numpy()

        # 2) list[Tensor]
        elif isinstance(embeddings, list) and len(embeddings) > 0 and torch.is_tensor(embeddings[0]):
            # каждый элемент списка — Tensor [DIM] на GPU
            ts = [t.detach().cpu() for t in embeddings]
            t = torch.stack(ts, dim=0)  # [batch, DIM]
            arr = t.float().numpy()

        # 3) fallback — пусть numpy сам разрулит (на случай future-изменений API)
        else:
            arr = np.asarray(embeddings, dtype="float32")

        # ожидаем [batch, DIM]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return arr.astype("float32")

    def execute(self, requests):
        """
        Triton вызывает этот метод для батча запросов.
        Каждый request содержит один входной тензор TEXT.
        """
        responses = []

        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            raw = in_tensor.as_numpy()      # может быть [batch] или [batch, 1]

            # Приводим к одномерному списку элементов
            flat = raw.reshape(-1)

            # BYTES -> python str
            texts = [
                x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                for x in flat
            ]

            embeddings = self._encode_texts(texts)  # [batch, DIM]

            out_tensor = pb_utils.Tensor("EMBEDDINGS", embeddings)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        LOGGER.info(f"[{MODEL_NAME}] Finalizing model")
