def load_model(ckpt_path: str, device: str):
    # TODO: load your torch model checkpoint here
    # return model
    raise NotImplementedError

def predict_saliency(model, pil_rgb, label: str):
    # TODO: run model(pil_rgb, label) -> saliency map
    # Must return np.ndarray of shape (H,W), float.
    raise NotImplementedError