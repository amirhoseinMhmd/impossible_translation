import argparse
import json
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils.utils import get_device, load_sentences_from_file, save_dataset


class Translator:
    def __init__(
        self,
        model: Union[str, Path, GPT2LMHeadModel],
        tokenizer: Optional[Union[str, Path, GPT2Tokenizer]] = None,
        device: Optional[Union[str, torch.device]] = None,
        temperature: float = 0.3,
        do_sample: bool = True,
    ):
        self.device = torch.device(device) if isinstance(device, str) else (device or get_device())
        self.temperature = temperature
        self.do_sample = do_sample

        if isinstance(tokenizer, GPT2Tokenizer):
            self.tokenizer = tokenizer
        else:
            tokenizer_source = tokenizer or model
            self.tokenizer = GPT2Tokenizer.from_pretrained('mission-impossible-lms/partial-reverse-gpt2')

        self.tokenizer.pad_token = self.tokenizer.eos_token

        if isinstance(model, GPT2LMHeadModel):
            self.model = model
        else:
            self.model = GPT2LMHeadModel.from_pretrained(model)

        self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.model = self.model.to(self.device)
        self.model.eval()
        self.max_position_embeddings = self.model.config.max_position_embeddings

    def _process_single_chunk(self, input_text: str) -> str:
        prompt = f"Fix this text: {input_text}\nCorrected:"
        prompt_encoding = self.tokenizer(prompt, return_tensors="pt")
        input_ids = prompt_encoding["input_ids"].to(self.device)
        attention_mask = prompt_encoding["attention_mask"].to(self.device)

        input_tokens = self.tokenizer.encode(input_text)
        input_length = prompt_encoding["input_ids"].shape[1]
        max_new_tokens = min(len(input_tokens) + 5, self.max_position_embeddings - input_length)
        min_new_tokens = max(1, len(input_tokens) - 5)

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature

        with torch.inference_mode():
            output = self.model.generate(**generation_kwargs)

        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "Corrected:" in generated:
            return generated.split("Corrected:", 1)[1].strip()
        return generated.strip()

    def _split_into_chunks(self, text: str, max_chunk_size: int = 800, overlap: int = 100):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + max_chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(self.tokenizer.decode(chunk_tokens, skip_special_tokens=True))

            if end >= len(tokens):
                break
            start = end - overlap

        return chunks

    @staticmethod
    def _merge_chunks(chunks):
        if not chunks:
            return ""

        result = chunks[0]
        for chunk in chunks[1:]:
            result = result.rstrip() + " " + chunk.lstrip()
        return result

    def translate(self, corrupted_text: str) -> str:
        if not corrupted_text:
            return ""

        input_tokens = self.tokenizer.encode(corrupted_text)
        prompt_template = "Fix this text: {}\nCorrected:"
        prompt_overhead = len(self.tokenizer.encode(prompt_template.format("")))
        max_input_size = self.max_position_embeddings - prompt_overhead - 50

        if len(input_tokens) <= max_input_size:
            return self._process_single_chunk(corrupted_text)

        chunks = self._split_into_chunks(
            corrupted_text,
            max_chunk_size=max_input_size,
            overlap=100,
        )
        translated_chunks = [self._process_single_chunk(chunk) for chunk in chunks]
        return self._merge_chunks(translated_chunks)


def load_translation_data(data_path: Union[str, Path]):
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data not found: {data_path}")

    if data_path.suffix.lower() == ".json":
        with open(data_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        sentences = load_sentences_from_file(str(data_path))
        data = [{"input": sentence, "actual": None} for sentence in sentences]

    normalized = []
    for index, item in enumerate(data):
        if isinstance(item, dict):
            input_text = item.get("input")
            actual_text = item.get("actual")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            input_text, actual_text = item
        else:
            raise ValueError(
                f"Unsupported item at index {index} in {data_path}: {type(item).__name__}"
            )

        if input_text is None:
            raise ValueError(f"Missing input text at index {index} in {data_path}")

        normalized.append({
            "input": input_text,
            "actual": actual_text,
        })

    return normalized


def translate_dataset(translator: Translator, data):
    results = []
    for item in data:
        translated_text = translator.translate(item["input"])
        results.append({
            "input": item["input"],
            "prediction": translated_text,
            "actual": item["actual"],
        })
    return results


def default_output_path(model_name: str, data_path: Union[str, Path], language: Optional[str]):
    model_name = Path(str(model_name)).name.replace("/", "_")
    data_stem = Path(data_path).stem
    language_suffix = f"_{language}" if language else ""
    return Path(f"translations_{data_stem}_{model_name}{language_suffix}.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--do_sample", action="store_true")

    args = parser.parse_args()

    translator = Translator(
        model=args.model,
        tokenizer=args.model,
        device=args.device,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )

    data = load_translation_data(args.data)
    results = translate_dataset(translator, data)

    output_path = Path(args.output) if args.output else default_output_path(
        model_name=args.model,
        data_path=args.data,
        language=args.language,
    )
    save_dataset(results, output_path)


if __name__ == "__main__":
    main()
