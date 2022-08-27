from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image, ImageDraw, ImageFont


class ImageCaptioning:
    def __init__(self):
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_text_size(self, text, image, font):
        im = Image.new('RGB', (image.width, image.height))
        draw = ImageDraw.Draw(im)
        return draw.textsize(text, font)

    def find_font_size(self, text, font, image, width_ratio):
        tested_font_size = 100
        tested_font = ImageFont.truetype(font, tested_font_size)
        observed_width, observed_height = self.get_text_size(text, image, tested_font)
        estimated_font_size = tested_font_size / (observed_width / image.width) * width_ratio
        return round(estimated_font_size)

    def generate_caption(self, image_path):

        max_length = 16
        num_beams = 4
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        clean_preds = [pred.strip() for pred in preds]
        caption = "\"" + clean_preds[0].capitalize() + "\""

        return caption

    def create_captioned_image(self, image_path, caption):
        image = Image.open(image_path)
        width, height = image.size

        bi = Image.new("RGBA", (width + 10, height + (height // 5)), "white")
        bi.paste(image, (5, 5, (width + 5), (height + 5)))

        font_family = 'resources/DePixelBreit.ttf'
        font_size = self.find_font_size(caption, font_family, bi, width_ratio=.8)
        font = ImageFont.truetype(font_family, size=font_size)
        w, h = font.getsize(caption)

        draw = ImageDraw.Draw(bi)
        draw.text(((width - w) / 2, (height + ((height / 5) - h) / 2)), caption, font=font, fill="black")

        return bi

    def caption(self, image_path):
        caption = self.generate_caption(image_path)
        captioned_image = self.create_captioned_image(image_path, caption)

        return captioned_image

