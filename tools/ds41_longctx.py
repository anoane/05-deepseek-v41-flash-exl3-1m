# Long-context fill test. Capacity at 1M is proven; sustained throughput at DEPTH is not.
# Builds a large prompt, sends it, reports prompt_tokens, wall time, prefill and decode rates.
import json, sys, time, urllib.request

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 128000
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 32
B = "http://127.0.0.1:8111"

# ~10 tokens per repetition; overshoot slightly, the server reports the true count.
unit = "The quick brown fox jumps over the lazy dog. "
body_text = unit * int(TARGET / 9)
prompt = ("Below is a long document. Read it and then answer.\n\n"
          + body_text
          + "\n\nQuestion: what animal jumps over the lazy dog? Answer in three words.")

req = urllib.request.Request(
    B + "/v1/chat/completions",
    data=json.dumps({"model": "ds41",
                     "messages": [{"role": "user", "content": prompt}],
                     "max_tokens": MAXTOK, "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"})

print("target ~%d tokens; sending %d chars" % (TARGET, len(prompt)), flush=True)
t0 = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=14400))
except Exception as e:
    print("FAILED after %.1fs: %r" % (time.time() - t0, e))
    sys.exit(2)
dt = time.time() - t0

u = r.get("usage", {}) or {}
pt = u.get("prompt_tokens") or 0
ct = u.get("completion_tokens") or 0
msg = r["choices"][0]["message"]
print("PROMPT_TOKENS: %d" % pt)
print("COMPLETION_TOKENS: %d" % ct)
print("WALL: %.1f s" % dt)
if pt:
    print("AGGREGATE PREFILL: %.0f tok/s (prompt/wall, decode included)" % (pt / dt))
print("FINISH: %s" % r["choices"][0].get("finish_reason"))
print("CONTENT: %s" % repr(msg.get("content"))[:300])
print("REASONING: %s" % repr(msg.get("reasoning_content"))[:200])
