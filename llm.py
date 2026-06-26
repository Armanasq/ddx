import json
import requests


class LLM:
    def __init__(self, config, trace=False, ledger=None, session_id="session"):
        self.config = config
        self.trace = trace
        self.ledger = ledger
        self.session_id = session_id
        self.calls = []

    def reset(self):
        self.calls = []

    def totals(self):
        out = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
        for call in self.calls:
            for key in out:
                out[key] += call[key]
        return out

    def complete_json(self, system, payload, max_tokens=None):
        text = self.complete(system, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens, json_mode=True)
        return json.loads(_json_text(text))

    def complete_json_text(self, prompt, max_tokens=None):
        text = self.complete("Return valid JSON only.", prompt, max_tokens=max_tokens, json_mode=True)
        try:
            return json.loads(_json_text(text))
        except json.JSONDecodeError:
            retry_max = (max_tokens or self.config.max_llm_tokens) * 2
            text = self.complete("Return valid JSON only.", prompt, max_tokens=retry_max, json_mode=True)
            return json.loads(_json_text(text))

    def complete_text(self, system, payload, max_tokens=None):
        return self.complete(system, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens).strip()

    def complete_text_text(self, prompt, max_tokens=None):
        return self.complete("Answer directly.", prompt, max_tokens=max_tokens).strip()

    def complete(self, system, user, max_tokens=None, json_mode=False):
        call_id = len(self.calls) + 1
        url = f"{self.config.azure_endpoint}/openai/deployments/{self.config.azure_deployment}/chat/completions"
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_completion_tokens": max_tokens or self.config.max_llm_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.trace:
            print(f"[LLM CALL {call_id}]")
            print("[INPUT]")
            print(f"system: {system}")
            print(f"user: {user}")
        response = requests.post(
            url,
            params={"api-version": self.config.azure_api_version},
            headers={"api-key": self.config.azure_api_key, "content-type": "application/json"},
            json=body,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        call = self._record_usage(call_id, system, user, data.get("usage") or {}, content)
        if self.trace:
            print("[OUTPUT]")
            print(content)
            print(
                "[TOKENS] "
                f"input={call['input_tokens']} "
                f"cached_input={call['cached_input_tokens']} "
                f"output={call['output_tokens']} "
                f"total={call['total_tokens']} "
                f"cost=${call['cost_usd']:.6f}"
            )
        return content

    def _record_usage(self, call_id, system, user, usage, content):
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
        details = usage.get("prompt_tokens_details") or {}
        cached_input_tokens = int(details.get("cached_tokens") or 0)
        billable_input = max(input_tokens - cached_input_tokens, 0)
        input_cost = billable_input * self.config.input_price_per_1m / 1_000_000
        cached_cost = cached_input_tokens * self.config.cached_input_price_per_1m / 1_000_000
        output_cost = output_tokens * self.config.output_price_per_1m / 1_000_000
        cost = input_cost + cached_cost + output_cost
        call = {
            "call": call_id,
            "system": system,
            "user": user,
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost_usd": input_cost,
            "cached_cost_usd": cached_cost,
            "output_cost_usd": output_cost,
            "cost_usd": cost,
            "output": content,
        }
        self.calls.append(call)
        if self.ledger is not None:
            self.ledger.record_call(call, self.session_id, self.config.azure_deployment)
        return call


def _json_text(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]
    return text
