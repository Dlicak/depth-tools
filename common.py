import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def data_dir():
    d = os.environ.get("DEPTH_TOOLS_HOME")
    if not d:
        d = os.path.join(os.path.expanduser("~"), "Z-depth")
    os.makedirs(d, exist_ok=True)
    return d


def model_dir():
    d = os.environ.get("DEPTH_TOOLS_MODELS")
    if d:
        os.makedirs(d, exist_ok=True)
        return d
    return data_dir()


def find_model(name):
    for d in (model_dir(), SCRIPT_DIR):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(model_dir(), name)


def default_src():
    s = os.environ.get("DEPTH_TOOLS_SRC")
    if s and os.path.exists(s):
        return s
    dl = os.path.join(os.path.expanduser("~"), "Downloads")
    if os.path.isdir(dl):
        try:
            files = [os.path.join(dl, f) for f in os.listdir(dl)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"))]
            if files:
                return max(files, key=os.path.getmtime)
        except Exception:
            pass
    return ""


def home_dir():
    return os.path.expanduser("~")
