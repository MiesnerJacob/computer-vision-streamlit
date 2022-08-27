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


label_colors = {
'person':(255,0,0),
'bicycle':(233,150,122),
'car':(0,128,128),
'motorcycle':(62, 12, 150),
'airplane':(62, 12, 150),
'bus':(62, 12, 150),
'train':(62, 12, 150),
'truck':(62, 12, 150),
'boat':(178,34,34),
'traffic light':(50, 123, 110),
'fire hydrant':(189, 190, 108),
'street sign':(169, 78, 123),
'stop sign':(0,0,128),
'parking meter':(224, 35, 22),
'bench':(133, 50, 12),
'bird':(241, 238, 184),
'cat':(237, 156, 108),
'dog':(0,0,255),
'horse':(143, 129, 225),
'sheep':(220, 27, 199),
'cow':(57, 172, 54),
'elephant':(190, 241, 114),
'bear':(77, 52, 167),
'zebra':(28, 219, 241),
'giraffe':(244, 174, 167),
'hat':(255,215,0),
'backpack':(184,134,11),
'umbrella':(84, 78, 46),
'shoe':(84, 241, 138),
'eye glasses':(228, 143, 4),
'handbag':(32, 161, 64),
'tie':(123, 171, 197),
'suitcase':(97, 41, 12),
'frisbee':(228, 156, 73),
'skis':(241, 139, 54),
'snowboard':(41, 130, 52),
'sports ball':(128,0,128),
'kite':(194, 226, 143),
'baseball bat':(151, 143, 193),
'baseball glove':(237, 231, 174),
'skateboard':(120, 248, 190),
'surfboard':(238,232,170),
'tennis racket':(147, 208, 90),
'bottle':(255,69,0),
'plate':(54, 114, 237),
'wine glass':(12, 145, 41),
'cup':(0,255,0),
'fork':(23, 159, 50),
'knife':(62, 90, 19),
'spoon':(206, 124, 210),
'bowl':(245, 197, 156),
'banana':(151, 132, 56),
'apple':(161, 43, 202),
'sandwich':(144, 65, 234),
'orange':(118, 82, 26),
'broccoli':(21, 244, 85),
'carrot':(244, 232, 128),
'hot dog':(37, 63, 170),
'pizza':(255,140,0),
'donut':(36, 26, 203),
'cake':(38, 38, 118),
'chair':(238, 204, 47),
'couch':(247, 44, 207),
'potted plant':(142, 22, 88),
'bed':(255,255,0),
'mirror':(21, 173, 249),
'dining table':(244, 91, 165),
'window':(150, 154, 19),
'desk':(205, 54, 216),
'toilet':(234, 6, 58),
'door':(15, 227, 205),
'tv':(0,255,255),
'laptop':(128,0,0),
'mouse':(128,128,0),
'remote':(128,128,128),
'keyboard':(0,128,0),
'cell phone':(255,0,255),
'microwave':(122, 141, 38),
'oven':(243, 137, 199),
'toaster':(166, 242, 79),
'sink':(130, 222, 27),
'refrigerator':(195, 175, 227),
'blender':(67, 78, 238),
'book':(61, 158, 106),
'clock':(175, 118, 203),
'vase':(53, 224, 70),
'scissors':(51, 250, 170),
'teddy bear':(105, 195, 89),
'hair drier':(26, 134, 197),
'toothbrush':(138, 21, 148),
'hair brush':(248, 55, 231)
}
