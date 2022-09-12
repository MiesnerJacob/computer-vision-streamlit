import easyocr
from PIL import Image, ImageDraw


class ImageOpticalCharacterRecognition:
    """
    XXXXX.
    """

    def __init__(self):
        """
        The constructor for XXX class.
        Attributes:
            xxx: ___
            xxx: ___
            xxx: ___
        """

        self.reader = easyocr.Reader(['en'])

    @staticmethod
    def draw_boxes(image_path, bounds, color='yellow', width=2):
        """
        XXX.

        Parameters:
            xxx (type): ___
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
        XXX.

        Parameters:
            xxx (type): ___
        Returns:
            xxx (type): ___
        """

        extracted_text = self.reader.readtext(image_path)
        boxes = [i[0] for i in extracted_text]
        text_list = [i[1] for i in extracted_text]
        probs = [i[2] for i in extracted_text]
        text = "  \n".join(text_list)

        return boxes, text

    def image_ocr(self, image_path):
        """
        XXX.

        Parameters:
            xxx (type): ___
        Returns:
            xxx (type): ___
        """

        boxes, text = self.run_recognition(image_path)
        annotated_image = self.draw_boxes(image_path, boxes, color='green', width=3)

        return annotated_image, text

