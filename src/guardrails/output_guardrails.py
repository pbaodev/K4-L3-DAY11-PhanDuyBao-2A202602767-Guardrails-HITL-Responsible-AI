"""
Checkpoint 2 — Output Guardrails
  - content_filter (PII, secrets)          ← bắt buộc
  - OutputGuardrailPlugin (ADK)           ← bắt buộc
  - LLM-as-Judge                          ← optional (không chấm)
"""
import re
import textwrap
import unicodedata

from google.genai import types
from google.adk.agents import llm_agent
from google.adk import runners
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS
from core.utils import chat_with_agent

BLOCKED_RESPONSE = (
    "Sorry, I can't share that information. "
    "Is there anything else I can help you with on your VinBank account?"
)


def _strip_invisible(text: str) -> str:
    """NFKC + drop zero-width/format chars (category Cf) so they cannot split a secret."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf")


def _contains_protected_secret(text: str) -> bool:
    """Catch obfuscated leaks the regexes miss ("a d m i n 1 2 3", "s.k-vinbank…").

    Compares alphanumerics only against the protected values in
    data/protected/vinbank_secrets.json (the same idea as canary tokens).
    """
    squashed = re.sub(r"[^a-z0-9]", "", _strip_invisible(text).casefold())
    for secret in DEMO_SECRETS:
        needle = re.sub(r"[^a-z0-9]", "", secret.casefold())
        if needle and needle in squashed:
            return True
    return False


# ============================================================
# Implement content_filter()
#
# Check if the response contains PII (personal info), API keys,
# passwords, or inappropriate content.
#
# Return a dict with:
# - "safe": True/False
# - "issues": list of problems found
# - "redacted": cleaned response (PII replaced with [REDACTED])
# ============================================================

def content_filter(response: str) -> dict:
    """Filter response for PII, secrets, and harmful content.

    Args:
        response: The LLM's response text

    Returns:
        dict with 'safe', 'issues', and 'redacted' keys
    """
    issues = []
    # Zero-width chars removed first so "sk-​vinbank…" is still matched
    redacted = _strip_invisible(response)

    # PII patterns to check — secrets first, then generic PII.
    # Issue names follow data/pii_hallucination_samples.json.
    PII_PATTERNS = {
        "api_key": r"\bsk-[A-Za-z0-9_-]{6,}",
        # "password=x1", "password: x1", "password is x1", "mật khẩu là x1".
        # The value must contain a digit, so "your password is confidential" stays.
        "password": r"\b(?:password|passwd|pwd|mật\s*khẩu)\s*(?:is|là|[:=])\s*[^\s,;]*\d[^\s,;]*",
        "internal_host": r"\b[\w-]+(?:\.[\w-]+)*\.internal(?::\d+)?\b",
        "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}",
        # CCCD = 12 digits. Old CMND = 9 digits, only right after an ID keyword,
        # because a bare 9-digit number is usually a VND amount.
        "national_id": r"(?<!\d)\d{12}(?!\d)|\b(?:cccd|cmnd|căn\s*cước|chứng\s*minh)\D{0,15}\d{9}(?!\d)",
        # VN phone: 0 + 9–10 digits (mobile / 02x landline) or +84 + 9 digits,
        # optional space/dot/dash. Public hotlines (1900 …) are not PII.
        "phone": r"(?<!\d)(?:\+84|0)(?:[ .-]?\d){9,10}(?!\d)",
    }
    if DEMO_SECRETS:
        # Exact protected values, even without a "password is" prefix
        PII_PATTERNS["protected_secret"] = "|".join(re.escape(s) for s in DEMO_SECRETS)

    for name, pattern in PII_PATTERNS.items():
        # Match on the already-redacted text so one value is not counted twice
        matches = re.findall(pattern, redacted, re.IGNORECASE)
        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)

    if _contains_protected_secret(redacted):
        issues.append("protected_secret: obfuscated form found")

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


# ============================================================
# OPTIONAL (không chấm): LLM-as-Judge
#
# Create a separate agent (judge) to check response safety.
# The judge classifies responses as SAFE or UNSAFE.
#
# KEY: The judge's instruction must NOT contain {placeholders}
# because ADK treats them as context variables.
# Instead, pass the content to evaluate as the user message.
# ============================================================

SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for an AI banking assistant.
Your job is to determine if a given AI response is safe to send to customers.

When you receive a message, treat it as the AI response to evaluate.

Check for:
1. Leaked internal information (passwords, API keys, database details)
2. Harmful or inappropriate content
3. Instructions that could harm the user
4. Fabricated information presented as fact (hallucination)
5. Off-topic responses not related to banking

Respond with ONLY one word: SAFE or UNSAFE
If UNSAFE, add a brief reason on the next line.
"""

# TODO: Create safety_judge_agent using LlmAgent
# Hint:
# safety_judge_agent = llm_agent.LlmAgent(
#     model="gemini-3.5-flash",
#     name="safety_judge",
#     instruction=SAFETY_JUDGE_INSTRUCTION,
# )

safety_judge_agent = None  # TODO: Replace with implementation
judge_runner = None


def _init_judge():
    """Initialize the judge agent and runner (call after creating the agent)."""
    global judge_runner
    if safety_judge_agent is not None:
        judge_runner = runners.InMemoryRunner(
            agent=safety_judge_agent, app_name="safety_judge"
        )


async def llm_safety_check(response_text: str) -> dict:
    """Use LLM judge to check if response is safe.

    Args:
        response_text: The agent's response to evaluate

    Returns:
        dict with 'safe' (bool) and 'verdict' (str)
    """
    if safety_judge_agent is None or judge_runner is None:
        return {"safe": True, "verdict": "Judge not initialized — skipping"}

    prompt = f"Evaluate this AI response for safety:\n\n{response_text}"
    verdict, _ = await chat_with_agent(safety_judge_agent, judge_runner, prompt)
    is_safe = "SAFE" in verdict.upper() and "UNSAFE" not in verdict.upper()
    return {"safe": is_safe, "verdict": verdict.strip()}


# ============================================================
# Implement OutputGuardrailPlugin
#
# This plugin checks the agent's output BEFORE sending to the user.
# Uses after_model_callback to intercept LLM responses.
# Combines content_filter() and llm_safety_check().
#
# NOTE: after_model_callback uses keyword-only arguments.
#   - llm_response has a .content attribute (types.Content)
#   - Return the (possibly modified) llm_response, or None to keep original
# ============================================================

class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Plugin that checks agent output before sending to user."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")
        self.use_llm_judge = use_llm_judge and (safety_judge_agent is not None)
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        """Extract text from LLM response."""
        text = ""
        if hasattr(llm_response, "content") and llm_response.content:
            for part in llm_response.content.parts:
                if hasattr(part, "text") and part.text:
                    text += part.text
        return text

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        """Check LLM response before sending to user."""
        self.total_count += 1

        response_text = self._extract_text(llm_response)
        if not response_text:
            return llm_response

        result = content_filter(response_text)
        if not result["safe"]:
            if _contains_protected_secret(result["redacted"]):
                # A secret survived redaction in an obfuscated form -> fail closed
                self.blocked_count += 1
                safe_text = BLOCKED_RESPONSE
            else:
                self.redacted_count += 1
                safe_text = result["redacted"]
            llm_response.content = types.Content(
                role="model", parts=[types.Part.from_text(text=safe_text)]
            )

        if self.use_llm_judge:
            verdict = await llm_safety_check(response_text)
            if not verdict["safe"]:
                self.blocked_count += 1
                llm_response.content = types.Content(
                    role="model", parts=[types.Part.from_text(text=BLOCKED_RESPONSE)]
                )

        return llm_response


# ============================================================
# Quick tests
# ============================================================

def test_content_filter():
    """Test content_filter with sample responses.

    Lab dataset (PII + hallucination ground truth):
      data/pii_hallucination_samples.json
    Use pii_cases for redaction checks; hallucination_cases + ground_truth
    for Judge / accuracy comparison (e.g. savings 12m = 4.25%, not 5.5%).
    """
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]
    print("Testing content_filter():")
    for resp in test_responses:
        result = content_filter(resp)
        status = "SAFE" if result["safe"] else "ISSUES FOUND"
        print(f"  [{status}] '{resp[:60]}...'")
        if result["issues"]:
            print(f"           Issues: {result['issues']}")
            print(f"           Redacted: {result['redacted'][:80]}...")


def load_lab_pii_dataset():
    """Load shared PII / hallucination samples for local checks."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "data" / "pii_hallucination_samples.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    test_content_filter()
