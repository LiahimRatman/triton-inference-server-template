import logging
from typing import List

import torch
import numpy as np
from transformers import AutoModel
import triton_python_backend_utils as pb_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


MODEL_NAME = "jinaai/jina-clip-v2"
TRUNCATE_DIM = 1024   # full dimension; adjust if you want Matryoshka truncation


class TritonPythonModel:
    def initialize(self, args):
        """
        Called once when the model is loaded.
        """
        device_id = args.get("model_instance_device_id", "0")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Initializing {MODEL_NAME} on {self.device}")

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        logger.info("jina-clip-v2 loaded successfully")

    def _tensors_to_numpy_batch(self, embeddings) -> np.ndarray:
        """
        Универсальная функция: Tensor / list[Tensor] -> numpy [batch, dim] на CPU.
        """
        # Tensor [batch, dim]
        if torch.is_tensor(embeddings):
            t = embeddings.detach()
            if t.is_cuda:
                t = t.cpu()
            arr = t.float().numpy()

        # list[Tensor [dim]]
        elif isinstance(embeddings, list) and len(embeddings) > 0 and torch.is_tensor(embeddings[0]):
            ts = [t.detach().cpu() for t in embeddings]
            t = torch.stack(ts, dim=0)      # [batch, dim]
            arr = t.float().numpy()

        else:
            arr = np.asarray(embeddings, dtype="float32")

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return arr.astype("float32")

    def _encode_text(self, texts: List[str]) -> np.ndarray:
        logger.info(f"[{MODEL_NAME}] _encode_text called, num_texts={len(texts)}")
        with torch.no_grad():
            embeddings = self.model.encode_text(
                texts,
                truncate_dim=TRUNCATE_DIM,
            )
        return self._tensors_to_numpy_batch(embeddings)

    def _encode_image(self, image_refs: List[str]) -> np.ndarray:
        """
        image_refs can be:
          - public URLs
          - local file paths
          - data URIs
        as supported by jina-clip-v2 encode_image.
        """
        logger.info(f"[{MODEL_NAME}] _encode_image called, num_images={len(image_refs)}")
        with torch.no_grad():
            embeddings = self.model.encode_image(
                image_refs,
                truncate_dim=TRUNCATE_DIM,
            )
        return self._tensors_to_numpy_batch(embeddings)

    def execute(self, requests):
        """
        Each request:
          - INPUT: BYTES [batch, 1]  -> texts or image URLs/paths
          - TYPE:  BYTES [1, 1]      -> "text" or "image"
        """
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            type_tensor = pb_utils.get_input_tensor_by_name(request, "TYPE")

            raw_inputs = input_tensor.as_numpy()               # [batch, 1] или [batch]
            flat_inputs = raw_inputs.reshape(-1)               # [batch]

            raw_type_arr = type_tensor.as_numpy().reshape(-1)  # [1]
            raw_type = raw_type_arr[0]

            if isinstance(raw_type, (bytes, bytearray)):
                mode = raw_type.decode("utf-8").strip().lower()
            else:
                mode = str(raw_type).strip().lower()

            items: List[str] = []
            for x in flat_inputs:
                if isinstance(x, (bytes, bytearray)):
                    items.append(x.decode("utf-8"))
                else:
                    items.append(str(x))

            if mode == "text":
                emb = self._encode_text(items)       # [batch, 1024]
            elif mode == "image":
                emb = self._encode_image(items)      # [batch, 1024]
            else:
                err = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        f"Unsupported TYPE value '{mode}', expected 'text' or 'image'"
                    )
                )
                responses.append(err)
                continue

            out_tensor = pb_utils.Tensor("EMBEDDINGS", emb)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses

    def finalize(self):
        logger.info("Finalizing jina_clip_v2 TritonPythonModel")
