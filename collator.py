from typing import Any, Dict, List, Union, cast, Optional

from PIL.Image import Image
import torch
from colpali_engine.models.idefics3 import ColIdefics3Processor
from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


def prefix_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Prefix all keys in a dictionary with the given prefix.
    """
    return {f"{prefix}{k}": v for k, v in data.items()}


class VisualRetrieverCollator:
    """
    Collator for training vision retrieval models.
    """

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        teacher_processor: Optional[BaseVisualRetrieverProcessor] = None,
        max_length: int = 60,
    ):  
        self.teacher_processor = teacher_processor
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None

        # If processor is one of the supported types, extract the <image> token id.
        if isinstance(self.processor, (ColPaliProcessor, ColIdefics3Processor)):
            image_token = "<image>"
            try:
                idx = self.processor.tokenizer.additional_special_tokens.index(image_token)
                self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[idx]
            except ValueError:
                self.image_token_id = None

        # Force padding to be on the right for ColPaliProcessor.
        if isinstance(self.processor, ColPaliProcessor) and self.processor.tokenizer.padding_side != "right":
            print("Setting padding side to right")
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts_query: List[Union[str, None]] = []
        images: List[Image] = []
        neg_images: List[Image] = []
        
        batch_teacher_doc = None
        batch_teacher_query = None

        # Parse the examples.
        for example in examples:
            query = example.get("query")
            texts_query.append(query)

            image = example.get("image")
            if image is None:
                raise ValueError("Image is None - This collator does not support None images yet.")
            images.append(cast(Image, image))

            neg_image = example.get("neg_image")
            if neg_image is not None:
                neg_images.append(cast(Image, neg_image))

        # Process images.
        batch_doc = self.processor.process_images(images=images)
        batch_neg_doc = self.processor.process_images(images=neg_images) if neg_images else None
        
        if self.teacher_processor:
            batch_teacher_doc = self.teacher_processor.process_images(images = images)

        # Process queries.
        if all(q is None for q in texts_query):
            batch_query = None
        elif any(q is None for q in texts_query):
            raise ValueError("Some queries are None. This collator does not support None queries yet.")
        else:
            batch_query = self.processor.process_queries(
                queries=cast(List[str], texts_query),
                max_length=self.max_length,
            )

            for key in batch_query:
                tensor = batch_query[key]
                if isinstance(tensor, torch.Tensor) and tensor.dim() == 2:
                    batch_query[key] = tensor[:, :self.max_length]

            if self.teacher_processor:
                batch_teacher_query = self.teacher_processor.process_queries(
                    queries=cast(List[str], texts_query),
                    max_length=self.max_length,
                )

                for key in batch_teacher_query:
                    tensor = batch_teacher_query[key]
                    if isinstance(tensor, torch.Tensor) and tensor.dim() == 2:
                        batch_teacher_query[key] = tensor[:, :self.max_length]
    

        # Prefix keys to avoid collisions.
        batch_all = prefix_keys(batch_doc, "doc_")

        if batch_teacher_doc: 
            batch_all.update(prefix_keys(batch_teacher_query, "teacher_query_"))

        if batch_teacher_query:
            batch_all.update(prefix_keys(batch_teacher_doc, "teacher_doc_"))

        if batch_query:
            batch_all.update(prefix_keys(batch_query, "query_"))

        if batch_neg_doc:
            batch_all.update(prefix_keys(batch_neg_doc, "neg_doc_"))

        return batch_all
