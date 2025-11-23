import requests
import base64
import json
import numpy as np

TRITON_URL = "http://99.27.206.243:57248"   # ← поменяй на свой IP/порт из vast.ai


EMB_MODEL = "jina_embeddings_v4"   # name из config.pbtxt
CLIP_MODEL = "jina_clip_v2"        # name из второго config.pbtxt


def triton_infer(model, inputs):
    url = f"{TRITON_URL}/v2/models/{model}/infer"
    r = requests.post(url, json={"inputs": inputs})
    print(f"\n=== Request to {model} ===")
    print("Status:", r.status_code)
    if r.status_code != 200:
        print("Body:", r.text)
        r.raise_for_status()
    return r.json()


def decode_tensor(resp, name):
    for output in resp["outputs"]:
        if output["name"] == name:
            data = output["data"]
            shape = output.get("shape", None)
            arr = np.asarray(data, dtype=np.float32)
            if shape is not None:
                arr = arr.reshape(shape)
            return arr
    raise KeyError(f"Output tensor {name} not found")


# -------- health --------
print("🔎 Checking Triton health...")
r_live = requests.get(f"{TRITON_URL}/v2/health/live")
r_ready = requests.get(f"{TRITON_URL}/v2/health/ready")
print("LIVE :", r_live.status_code, r_live.text)
print("READY:", r_ready.status_code, r_ready.text)


# -------- jina_embeddings_v4 --------
print("\n🧪 Testing jina_embeddings_v4...")

texts = [
    "Hello world",
    "Artificial intelligence is transforming society.",
    "Привет, как дела?"
]

# max_batch_size > 0 и dims: [-1] → Triton ждёт [batch, -1]
emb_inputs = [
    {
        "name": "TEXT",
        "datatype": "BYTES",           # TYPE_STRING в config → BYTES в HTTP
        "shape": [len(texts), 1],      # [batch, 1]
        "data": texts
    }
]

emb_resp = triton_infer(EMB_MODEL, emb_inputs)
emb = decode_tensor(emb_resp, "EMBEDDINGS")
print("Embeddings shape:", emb.shape)           # ожидаем (3, 2048)
print("First vector (5 dims):", emb[0][:5])


# -------- jina_clip_v2 (TEXT) --------
print("\n🧪 Testing jina_clip_v2 (TEXT)...")

clip_texts = [
    "A photo of a beautiful sunset by the ocean",
    "A dog playing in the snow"
]

clip_text_inputs = [
    {
        "name": "INPUT",
        "datatype": "BYTES",
        "shape": [len(clip_texts), 1],    # [batch=2, 1]
        "data": clip_texts
    },
    {
        "name": "TYPE",
        "datatype": "BYTES",
        "shape": [len(clip_texts), 1],    # <-- вот здесь было [1, 1]
        "data": ["text"] * len(clip_texts)  # ["text", "text"]
    }
]

clip_text_resp = triton_infer(CLIP_MODEL, clip_text_inputs)
clip_text_emb = decode_tensor(clip_text_resp, "EMBEDDINGS")
print("CLIP text embeddings shape:", clip_text_emb.shape)   # ожидаем (2, 1024)
print("First CLIP text vector (5 dims):", clip_text_emb[0][:5])


# -------- jina_clip_v2 (IMAGE) --------
print("\n🧪 Testing jina_clip_v2 (IMAGE)...")

images = [
    "https://picsum.photos/512",
    "https://picsum.photos/256"
]

clip_img_inputs = [
    {
        "name": "INPUT",
        "datatype": "BYTES",
        "shape": [len(images), 1],
        "data": images
    },
    {
        "name": "TYPE",
        "datatype": "BYTES",
        "shape": [len(images), 1],
        "data": ["image"] * len(images)
    }
]

clip_img_resp = triton_infer(CLIP_MODEL, clip_img_inputs)
clip_img_emb = decode_tensor(clip_img_resp, "EMBEDDINGS")
print("CLIP image embeddings shape:", clip_img_emb.shape)   # ожидаем (2, 1024)
print("First CLIP image vector (5 dims):", clip_img_emb[0][:5])

print("\n🎉 ALL TESTS FINISHED!")
