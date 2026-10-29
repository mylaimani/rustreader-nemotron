"""
Minimal client for NVIDIA Nemotron models served by Nebius Token Factory.

Token Factory exposes an OpenAI-compatible Chat Completions API, so one plain
HTTPS POST per call is all that is needed (no SDK added to the Lambda bundle).

Configuration (environment variables):
  NEBIUS_API_KEY         required to enable Nemotron. Without it every call
                         returns None and the agent runs pure-OpenCV as before.
  NEMOTRON_VISION_MODEL   default nvidia/Nemotron-3.5-Super-VL  (looks at the photo)
  NEMOTRON_EXPLAIN_MODEL  default nvidia/Nemotron-3.5-Super-VL  (writes English + Tamil;
                          the 120B model writes far better Tamil than the small one)
  NEMOTRON_TEXT_MODEL     default nvidia/Nemotron-3_5-Lightning (fast fallback explainer)
"""
import json
import os
import urllib.error
import urllib.request

NEBIUS_CHAT_COMPLETIONS_URL = "https://api.tokenfactory.nebius.com/v1/chat/completions"
VISION_MODEL = os.environ.get("NEMOTRON_VISION_MODEL", "nvidia/Nemotron-3.5-Super-VL")
EXPLAIN_MODEL = os.environ.get("NEMOTRON_EXPLAIN_MODEL", "nvidia/Nemotron-3.5-Super-VL")
TEXT_MODEL = os.environ.get("NEMOTRON_TEXT_MODEL", "nvidia/Nemotron-3_5-Lightning")
REQUEST_TIMEOUT_SECONDS = 20


def nemotron_is_configured():
    """True when a Nebius Token Factory API key is present."""
    return bool(os.environ.get("NEBIUS_API_KEY"))


def post_chat_completion(model_name, messages, maximum_tokens, disable_thinking=False):
    """One POST to Token Factory. Returns the reply text, or None on any failure."""
    if not nemotron_is_configured():
        return None
    payload = {"model": model_name, "messages": messages,
               "max_tokens": maximum_tokens, "temperature": 0.2}
    if disable_thinking:  # vLLM-style switch understood by NVIDIA Nemotron chat templates
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    request = urllib.request.Request(
        NEBIUS_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + os.environ["NEBIUS_API_KEY"]})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as reply:
            reply_body = json.loads(reply.read().decode("utf-8"))
        content = reply_body["choices"][0]["message"]["content"]
        if isinstance(content, list):  # some vision models reply as a list of parts
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        if os.environ.get("NEMOTRON_DEBUG"):
            finish_reason = reply_body["choices"][0].get("finish_reason")
            print(f"nemotron_raw_reply model={model_name} finish_reason={finish_reason} "
                  f"start={repr(content)[:300]} ... end={repr(content)[-400:]}")
        if not content:
            print(f"nemotron_empty_reply model={model_name} finish_reason="
                  f"{reply_body['choices'][0].get('finish_reason')}")
            return None
        return content
    except (urllib.error.URLError, OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        print(f"nemotron_call_failed model={model_name} error={error}")
        return None


def ask_nemotron_text(system_prompt, user_text, maximum_tokens=1200, model_name=None):
    """Plain text question, by default to the explainer model (thinking switched off)."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}]
    return post_chat_completion(model_name or EXPLAIN_MODEL, messages, maximum_tokens, disable_thinking=True)


def ask_nemotron_about_image(system_prompt, user_text, jpeg_base64, maximum_tokens=1500):
    """Question about one JPEG to the vision model (Nemotron Super VL)."""
    user_content = [{"type": "text", "text": user_text},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + jpeg_base64}}]
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}]
    return post_chat_completion(VISION_MODEL, messages, maximum_tokens)


def strip_reasoning_block(reply_text):
    """Reasoning models may start with <think>...</think>; keep only what follows."""
    end_tag_position = reply_text.rfind("</think>")
    if end_tag_position != -1:
        return reply_text[end_tag_position + len("</think>"):]
    return reply_text


def parse_json_reply(reply_text):
    """Pull the LAST JSON object out of a model reply.

    Reasoning models write their thinking first, and the thinking may quote
    example JSON, so the real answer is the last object that parses."""
    if not reply_text:
        return None
    cleaned_text = strip_reasoning_block(reply_text)
    end = cleaned_text.rfind("}")
    while end != -1:
        start = cleaned_text.rfind("{", 0, end)
        while start != -1:
            try:
                return json.loads(cleaned_text[start:end + 1])
            except json.JSONDecodeError:
                start = cleaned_text.rfind("{", 0, start)
        end = cleaned_text.rfind("}", 0, end)
    return None
