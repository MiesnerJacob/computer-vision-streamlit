import easyocr
from PIL import Image, ImageDraw


class ImageOpticalCharacterRecognition:
    """
    Recognize text in images.
    """

    def __init__(self):
        """
        The constructor for ImageOpticalCharacterRecognition class.
        Attributes:
            reader: model for running ocr
        """

        self.reader = easyocr.Reader(['en'])

    @staticmethod
    def draw_boxes(image_path, bounds, color='yellow', width=2):
        """
        Draw boxes around the text identified within the images.

        Parameters:
            image_path (PIL images): image to draw boxes on
            bounds (list): locations for bounding boxes within image
            color (string): color to draw bounding boxes with
            width (int): thickness of bounding box lines
        Returns:
            xxx (type): ___
        """

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        draw = ImageDraw.Draw(image)
        for bound in bounds:
            p0, p1, p2, p3 = bound
            draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)

        return image

    def run_recognition(self, image_path):
        """
        Run ocr and extract relevant results.

        Parameters:
            image_path (str): location for image file
        Returns:
            boxes (list): list for bounding boxes to apply to image
            text (str): extracted text from image
        """

        extracted_text = self.reader.readtext(image_path)
        boxes = [i[0] for i in extracted_text]
        text_list = [i[1] for i in extracted_text]
        probs = [i[2] for i in extracted_text]
        text = "  \n".join(text_list)

        return boxes, text

    def image_ocr(self, image_path):
        """
        Extract text from image and annotate image.

        Parameters:
            image_path (str): path to image file
        Returns:
            annotated_image (PIL Image): image with bounding boxes drawn on it
            text (str): text extracted from image
        """

        boxes, text = self.run_recognition(image_path)
        annotated_image = self.draw_boxes(image_path, boxes, color='green', width=3)

        return annotated_image, text

