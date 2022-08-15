from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import io


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_prediction(pil_img, output_dict, threshold=.8, id2label=None):
    keep = output_dict["scores"] > threshold
    boxes = output_dict["boxes"][keep].tolist()
    scores = output_dict["scores"][keep].tolist()
    labels = output_dict["labels"][keep].tolist()

    if id2label is not None:
        labels = [id2label[x] for x in labels]

    color_map = {}
    for index, label in enumerate(labels):
        colors_used = 0
        if label not in color_map:
            color_map[label] = sum(np.random.random((3, 1)).tolist(), [])
            colors_used += 1

    # Create updated filtered predictions object
    filtered_preds = {"labels": labels, "scores": scores, "boxes": boxes}

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for score, (xmin, ymin, xmax, ymax), label in zip(scores, boxes, labels):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color_map[label], linewidth=3))
        ax.text(xmin, ymin, f"{label}: {score:0.2f}", fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    return fig2img(plt.gcf()), filtered_preds
