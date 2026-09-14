# Prove the endpoint SERVES, not just loads. No run has ever reached a token yet.
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8111
WAIT = int(sys.argv[2]) if len(sys.argv) > 2 else 2400
base = "http://127.0.0.1:%d" % PORT

t0 = time.time()
while True:
    try:
        urllib.request.urlopen(base + "/v1/models", timeout=5).read()
        break
    except Exception:
        if time.time() - t0 > WAIT:
            print("TIMEOUT: endpoint never came up in %ds" % WAIT)
            sys.exit(1)
        time.sleep(5)
print("endpoint up after %.0fs" % (time.time() - t0))

body = json.dumps({
    "model": "ds41",
    "prompt": "Explain in two sentences why a pipeline-parallel split cannot cut through a KV-sharing group.",
    "max_tokens": 64,
    "temperature": 0,
}).encode()
req = urllib.request.Request(base + "/v1/completions", data=body,
                             headers={"Content-Type": "application/json"})
s = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=3600))
except Exception as e:
    print("GENERATION FAILED: %r" % (e,))
    sys.exit(2)
e_ = time.time()

c = r["choices"][0]
u = r.get("usage", {}) or {}
ct = u.get("completion_tokens") or 0
print("FINISH:", c.get("finish_reason"))
print("PROMPT_TOKENS:", u.get("prompt_tokens"), " COMPLETION_TOKENS:", ct)
if ct and e_ > s:
    print("WALL: %.2fs   DECODE: %.2f tok/s" % (e_ - s, ct / (e_ - s)))
else:
    print("WALL: %.2fs" % (e_ - s))
print("TEXT:", repr(c.get("text", ""))[:600])
