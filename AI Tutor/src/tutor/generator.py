from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from src.rag.index import RetrievedChunk
from src.tutor.leakage import detect_answer_leakage, rag_faithfulness_check, repair_hint_only
from src.utils.config import get_path


TURBOQUANT_ALIASES = {"turboquant", "turbo-quant", "tq", "tq3", "tq4", "turboquant-kv", "turboquant_kv"}


class SocraticTutor:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.tutor_cfg = cfg.get("tutor", {})
        self.mode = str(self.tutor_cfg.get("mode", "auto")).lower()
        self.log_path = get_path(cfg, "tutor_log")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._local_pipe = None
        self._turboquant_model = None
        self._turboquant_tokenizer = None
        self._turboquant_wrapper = None

    def build_prompt(
        self,
        student_question: str,
        weak_skill: str,
        mastery_probability: float,
        retrieved_chunks: list[RetrievedChunk],
        allow_direct_answer: bool = False,
        stricter: bool = False,
    ) -> str:
        context = "\n\n".join(
            f"[{i + 1}] Source: {Path(c.source).name if c.source else 'retrieved'}\n{c.text}"
            for i, c in enumerate(retrieved_chunks)
        )
        strict_line = (
            "Be stricter than usual: do not reveal the final numerical or symbolic answer, and do not provide a full worked solution."
            if stricter
            else ""
        )
        direct_line = "You may reveal the final answer only if the student explicitly asks for it." if allow_direct_answer else "Do not reveal the final answer."
        return f"""You are a Socratic AI tutor.

Student weak skill: {weak_skill}
Estimated mastery probability: {mastery_probability:.3f}

Retrieved grounding context:
{context if context else "[No retrieved context available.]"}

Student question:
{student_question}

Instructions:
- Use the retrieved context when it is relevant.
- {direct_line}
- Give a helpful hint, not a full solution.
- Ask exactly one guiding question.
- Adapt to the student's current mastery level.
- Include an optional next practice recommendation if useful.
- Avoid phrases like "the answer is", "final answer", or "solution is".
{strict_line}
"""

    def _generate_api(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI()
        model = self.tutor_cfg.get("openai_model", "gpt-4.1-mini")
        max_tokens = int(self.tutor_cfg.get("max_new_tokens", 220))
        temperature = float(self.tutor_cfg.get("temperature", 0.4))
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            text = getattr(response, "output_text", None)
            if text:
                return text
        except Exception:
            pass

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _generate_turboquant_server(self, prompt: str, base_url: str) -> str:
        from openai import OpenAI

        model = self.tutor_cfg.get("turboquant_model") or self.tutor_cfg.get("local_model", "Qwen/Qwen2.5-0.5B-Instruct")
        max_tokens = int(self.tutor_cfg.get("max_new_tokens", 220))
        temperature = float(self.tutor_cfg.get("temperature", 0.4))

        client = OpenAI(base_url=base_url, api_key="local")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _first_model_device(model: Any) -> torch.device:
        try:
            return next(model.parameters()).device
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_turboquant_direct(self):
        if self._turboquant_model is not None and self._turboquant_tokenizer is not None:
            return self._turboquant_model, self._turboquant_tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.tutor_cfg.get("turboquant_model") or self.tutor_cfg.get("local_model", "Qwen/Qwen2.5-0.5B-Instruct")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if not torch.cuda.is_available():
            model.to("cpu")
        self._turboquant_model = model
        self._turboquant_tokenizer = tokenizer
        return model, tokenizer

    def _make_turboquant_cache(self, model: Any):
        bits = float(self.tutor_cfg.get("turboquant_bits", 4))
        try:
            from turboquant import TurboQuantCache

            cache_bits = int(bits) if bits.is_integer() else bits
            return TurboQuantCache(bits=cache_bits)
        except (ImportError, AttributeError):
            from turboquant import TurboQuantModel

            if self._turboquant_wrapper is None:
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                self._turboquant_wrapper = TurboQuantModel(
                    model,
                    bits=bits,
                    device=str(self._first_model_device(model)),
                    dtype=dtype,
                )
            return self._turboquant_wrapper.make_dynamic_cache()

    def _generate_turboquant_direct(self, prompt: str) -> str:
        model, tokenizer = self._load_turboquant_direct()
        cache = self._make_turboquant_cache(model)
        device = self._first_model_device(model)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_len = int(inputs["input_ids"].shape[-1])
        temperature = float(self.tutor_cfg.get("temperature", 0.4))
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.tutor_cfg.get("max_new_tokens", 220)),
            "temperature": temperature,
            "do_sample": temperature > 0,
            "use_cache": True,
            "past_key_values": cache,
        }
        if getattr(tokenizer, "eos_token_id", None) is not None:
            generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

        with torch.inference_mode():
            outputs = model.generate(**inputs, **generate_kwargs)
        generated = outputs[0][input_len:]
        if generated.numel() == 0:
            generated = outputs[0]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_turboquant(self, prompt: str) -> str:
        base_url = str(self.tutor_cfg.get("turboquant_base_url") or "").strip()
        if base_url:
            return self._generate_turboquant_server(prompt, base_url)
        return self._generate_turboquant_direct(prompt)

    def _load_local_pipe(self):
        if self._local_pipe is not None:
            return self._local_pipe

        model_name = self.tutor_cfg.get("local_model", "Qwen/Qwen2.5-0.5B-Instruct")
        quantization = str(self.tutor_cfg.get("quantization", "auto")).lower()
        if quantization in TURBOQUANT_ALIASES:
            raise RuntimeError(
                "TurboQuant quantization was requested, but the plain HuggingFace pipeline cannot apply TurboQuant. "
                "Set tutor.mode to 'turboquant' to use the local TurboQuant cache path, optionally set "
                "tutor.turboquant_base_url for an OpenAI-compatible TurboQuant server, or choose "
                "auto/4bit/8bit/fp16 for the Transformers fallback."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

        kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if quantization in {"4bit", "nf4", "auto"}:
                try:
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except Exception as exc:
                    print(f"4-bit quantization unavailable, falling back to FP16: {exc}")
                    kwargs["torch_dtype"] = torch.float16
            elif quantization == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["device_map"] = None

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self._local_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return self._local_pipe

    def _generate_local(self, prompt: str) -> str:
        pipe = self._load_local_pipe()
        out = pipe(
            prompt,
            max_new_tokens=int(self.tutor_cfg.get("max_new_tokens", 220)),
            temperature=float(self.tutor_cfg.get("temperature", 0.4)),
            do_sample=float(self.tutor_cfg.get("temperature", 0.4)) > 0,
            return_full_text=False,
        )
        return out[0]["generated_text"]

    def _generate_template(self, weak_skill: str, mastery_probability: float, retrieved_chunks: list[RetrievedChunk]) -> str:
        context_hint = ""
        if retrieved_chunks:
            context_hint = f" The retrieved material emphasizes: {retrieved_chunks[0].text[:180].strip()}"
        level = "start with a concrete example" if mastery_probability < 0.45 else "connect this to the rule you already know"
        return (
            f"Hint: For {weak_skill}, {level} and identify which quantity or relationship is changing.{context_hint}\n\n"
            "Guiding question: What is the first relationship you can write down before doing any calculation?\n\n"
            "Next practice: Try one similar problem where only one number changes."
        )

    def _generate_once(self, prompt: str, weak_skill: str, mastery_probability: float, retrieved_chunks: list[RetrievedChunk]) -> tuple[str, str]:
        mode = self.mode
        quantization = str(self.tutor_cfg.get("quantization", "auto")).lower()
        if mode == "auto":
            mode = "turboquant" if quantization in TURBOQUANT_ALIASES else "api" if os.environ.get("OPENAI_API_KEY") else "template"
        if mode in {"turboquant", "tq"}:
            try:
                return self._generate_turboquant(prompt), "turboquant"
            except Exception as exc:
                print(f"TurboQuant LLM backend unavailable; using template fallback: {exc}")
                return self._generate_template(weak_skill, mastery_probability, retrieved_chunks), "template"
        if mode in {"api", "openai", "gpt"}:
            return self._generate_api(prompt), "api"
        if mode in {"local", "hf", "huggingface"}:
            try:
                return self._generate_local(prompt), "local"
            except Exception as exc:
                print(f"Local LLM failed; using template fallback: {exc}")
                return self._generate_template(weak_skill, mastery_probability, retrieved_chunks), "template"
        return self._generate_template(weak_skill, mastery_probability, retrieved_chunks), "template"

    def generate(
        self,
        student_question: str,
        weak_skill: str,
        mastery_probability: float,
        retrieved_chunks: list[RetrievedChunk],
        final_answer: str | None = None,
        allow_direct_answer: bool | None = None,
    ) -> dict[str, Any]:
        allow_direct_answer = bool(self.tutor_cfg.get("allow_direct_answer", False) if allow_direct_answer is None else allow_direct_answer)
        hint_mode = bool(self.tutor_cfg.get("hint_mode", True))
        prompt = self.build_prompt(student_question, weak_skill, mastery_probability, retrieved_chunks, allow_direct_answer)
        start = time.perf_counter()
        response, mode_used = self._generate_once(prompt, weak_skill, mastery_probability, retrieved_chunks)
        latency_ms = (time.perf_counter() - start) * 1000

        leakage = detect_answer_leakage(response, final_answer=final_answer, hint_mode=hint_mode and not allow_direct_answer)
        original = response
        repaired = False
        if leakage["leakage_score"] and not allow_direct_answer:
            strict_prompt = self.build_prompt(
                student_question,
                weak_skill,
                mastery_probability,
                retrieved_chunks,
                allow_direct_answer=False,
                stricter=True,
            )
            try:
                response, mode_used = self._generate_once(strict_prompt, weak_skill, mastery_probability, retrieved_chunks)
            except Exception:
                response = repair_hint_only(original, final_answer=final_answer)
            second = detect_answer_leakage(response, final_answer=final_answer, hint_mode=True)
            if second["leakage_score"]:
                response = repair_hint_only(response, final_answer=final_answer)
                leakage = second
            repaired = True

        faithfulness = rag_faithfulness_check(response, retrieved_chunks)
        result = {
            "response": response,
            "mode": mode_used,
            "latency_ms": latency_ms,
            "leakage": leakage,
            "repaired": repaired,
            "faithfulness": faithfulness,
            "original_response": original if repaired else None,
        }
        self._log(student_question, weak_skill, mastery_probability, result)
        return result

    def _log(self, question: str, weak_skill: str, mastery_probability: float, result: dict[str, Any]) -> None:
        payload = {
            "student_question": question,
            "weak_skill": weak_skill,
            "mastery_probability": mastery_probability,
            **result,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
