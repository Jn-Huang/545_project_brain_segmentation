import torch
from torch import nn
from transformers import SegformerConfig, SegformerModel, SegformerDecodeHead
from transformers import SegformerFeatureExtractor as SegformerImageProcessor

class CustomSegModel(nn.Module):
    def __init__(self, config):
        super(CustomSegModel, self).__init__()
        self.config = config
        if config.from_zoo is not None:
            self.processor = SegformerImageProcessor.from_pretrained(config.from_zoo)
            self.model = SegformerModel.from_pretrained(config.from_zoo)
            self.decode_head = SegformerDecodeHead.from_pretrained(config.from_zoo)
            self.decode_head.classifier = torch.nn.Conv2d(
                self.decode_head.config.decoder_hidden_size,
                config.num_labels,
                kernel_size=1
            )
        else:
            self.processor = SegformerImageProcessor(config)
            self.model = SegformerModel(config)
            self.decode_head = SegformerDecodeHead(config)
    
    def forward(self, images):
        inputs = self.processor(images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(inputs['pixel_values'].to(self.model.device),
                             output_attentions=False,
                             output_hidden_states=True,
                             return_dict=True)
        encoder_hidden_states = outputs.hidden_states
        logits = self.decode_head(encoder_hidden_states)
        return logits
