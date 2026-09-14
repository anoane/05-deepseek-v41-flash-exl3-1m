# Reproduce the client's exact failure and verify the fix:
#   400 '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
# Hits the PUBLIC endpoint with a real tools payload and tool_choice="auto".
import json, sys, time, urllib.request, urllib.error

URL = "https://ds4.example.invalid:8443/v1/chat/completions"
KEY = "<API_TOKEN>"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"], "description": "Temperature unit"},
            },
            "required": ["city"],
        },
    },
}]

body = {
    "model": "ds41",
    "messages": [{"role": "user", "content": "What's the weather in Reykjavik? Use the tool. Celsius."}],
    "tools": TOOLS,
    "tool_choice": "auto",
    "max_tokens": 256,
    "temperature": 0,
}

req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json",
                                      "Authorization": "Bearer " + KEY})
t = time.time()
try:
    r = json.load(urllib.request.urlopen(req, timeout=600))
except urllib.error.HTTPError as e:
    print("HTTP %d after %.1fs" % (e.code, time.time() - t))
    print(e.read().decode()[:500])
    sys.exit(2)
except Exception as e:
    print("FAILED after %.1fs: %r" % (time.time() - t, e))
    sys.exit(2)
dt = time.time() - t

ch = r["choices"][0]
msg = ch["message"]
tcs = msg.get("tool_calls") or []
print("HTTP 200 in %.1fs" % dt)
print("finish_reason :", ch.get("finish_reason"))
print("tool_calls    :", len(tcs))
for tc in tcs:
    fn = tc.get("function", {})
    print("   name      :", fn.get("name"))
    print("   arguments :", repr(fn.get("arguments"))[:220])
print("content       :", repr(msg.get("content"))[:160])
print("VERDICT:", "PASS - tool call parsed" if tcs and tcs[0].get("function", {}).get("name") == "get_weather"
      else "CHECK - no parsed tool_calls; inspect content for raw <｜DSML｜ calls> markup")
