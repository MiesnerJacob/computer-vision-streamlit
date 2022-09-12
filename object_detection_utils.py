from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import io


def fig2img(fig):
    """
    XXX.

    Parameters:
        xxx (type): ___
    Returns:
        xxx (type): ___
    """

    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_prediction(pil_img, output_dict, threshold=.8, id2label=None):
    """
    XXX.

    Parameters:
        xxx (type): ___
    Returns:
        xxx (type): ___
    """

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


# Label colors for object detection
label_colors = {
'Person':(255,0,0),
'Bicycle':(233,150,122),
'Car':(0,128,128),
'Motorcycle':(62, 12, 150),
'Airplane':(62, 12, 150),
'Bus':(62, 12, 150),
'Train':(62, 12, 150),
'Truck':(62, 12, 150),
'Boat':(178,34,34),
'Traffic Light':(50, 123, 110),
'Fire Hydrant':(189, 190, 108),
'Street Sign':(169, 78, 123),
'Stop Sign':(0,0,128),
'Parking meter':(224, 35, 22),
'Bench':(133, 50, 12),
'Bird':(241, 238, 184),
'Cat':(237, 156, 108),
'Dog':(0,0,255),
'Horse':(143, 129, 225),
'Sheep':(220, 27, 199),
'Cow':(57, 172, 54),
'Elephant':(190, 241, 114),
'Bear':(77, 52, 167),
'Zebra':(28, 219, 241),
'Giraffe':(244, 174, 167),
'Hat':(255,215,0),
'Backpack':(184,134,11),
'Umbrella':(84, 78, 46),
'Shoe':(84, 241, 138),
'Eye glasses':(228, 143, 4),
'Handbag':(32, 161, 64),
'Tie':(123, 171, 197),
'Suitcase':(97, 41, 12),
'Frisbee':(228, 156, 73),
'Skis':(241, 139, 54),
'Snowboard':(41, 130, 52),
'Sports Ball':(128,0,128),
'Kite':(194, 226, 143),
'Baseball Bat':(151, 143, 193),
'Baseball Glove':(237, 231, 174),
'Skateboard':(120, 248, 190),
'Surfboard':(238,232,170),
'Tennis Racket':(147, 208, 90),
'Bottle':(255,69,0),
'Plate':(54, 114, 237),
'Wine Glass':(12, 145, 41),
'Cup':(0,255,0),
'Fork':(23, 159, 50),
'Knife':(62, 90, 19),
'Spoon':(206, 124, 210),
'Bowl':(245, 197, 156),
'Banana':(151, 132, 56),
'Apple':(161, 43, 202),
'Sandwich':(144, 65, 234),
'Orange':(118, 82, 26),
'Broccoli':(21, 244, 85),
'Carrot':(244, 232, 128),
'Hot Dog':(37, 63, 170),
'Pizza':(255,140,0),
'Donut':(36, 26, 203),
'Cake':(38, 38, 118),
'Chair':(238, 204, 47),
'Couch':(247, 44, 207),
'Potted Plant':(142, 22, 88),
'Bed':(255,255,0),
'Mirror':(21, 173, 249),
'Dining Table':(244, 91, 165),
'Window':(150, 154, 19),
'Desk':(205, 54, 216),
'Toilet':(234, 6, 58),
'Door':(15, 227, 205),
'Tv':(0,255,255),
'Laptop':(128,0,0),
'Mouse':(128,128,0),
'Remote':(128,128,128),
'Keyboard':(0,128,0),
'Cell Phone':(255,0,255),
'Microwave':(122, 141, 38),
'Oven':(243, 137, 199),
'Toaster':(166, 242, 79),
'Sink':(130, 222, 27),
'Refrigerator':(195, 175, 227),
'Blender':(67, 78, 238),
'Book':(61, 158, 106),
'Clock':(175, 118, 203),
'Vase':(53, 224, 70),
'Scissors':(51, 250, 170),
'Teddy Bear':(105, 195, 89),
'Hair Drier':(26, 134, 197),
'Toothbrush':(138, 21, 148),
'Hair Brush':(248, 55, 231)
}
