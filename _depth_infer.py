import sys

import numpy as np
import onnxruntime as ort
from PIL import Image


def main():
    model_path, nw, nh, img_path, out_path = sys.argv[1:6]
    img = Image.open(img_path).convert("RGB")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = img.resize((int(nw), int(nh)), Image.LANCZOS)
    arr = np.asarray(inp, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32)
    pred = sess.run(None, {sess.get_inputs()[0].name: arr.transpose(2, 0, 1)[None]})[0]
    np.save(out_path, pred[0])


if __name__ == "__main__":
    main()
