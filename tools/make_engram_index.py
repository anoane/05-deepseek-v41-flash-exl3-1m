"""Write model.safetensors.index.json into the engram dir.

DiskEngramTable.__init__ (engram_disk.py:39-47) opens model_dir/model.safetensors.index.json
and requires layers.{1,14}.engram.embed.{weight,scale}. The dir had both 101 GB shards but NO
index, so the reader had the data and no map -> FileNotFoundError.

Built from the shards' own headers (authoritative) rather than copying diffbot's 14 MB index,
so every entry is verified present. Includes ALL tensors in these shards, not just embed, since
the reader may resolve wkv/k_weight/q_weight through the same map.
"""
import json, os, struct

D = "/root/workspace/DeepSeek-V4.1-Flash-engram"
out = os.path.join(D, "model.safetensors.index.json")
if os.path.exists(out):
    raise SystemExit("index already exists: %s" % out)

weight_map, total = {}, 0
for fn in sorted(f for f in os.listdir(D) if f.endswith(".safetensors")):
    p = os.path.join(D, fn)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        weight_map[k] = fn
        a, b = v["data_offsets"]
        total += (b - a)
    print("  %s -> %d tensors" % (fn, len([k for k in hdr if k != '__metadata__'])))

required = ["layers.1.engram.embed.weight", "layers.1.engram.embed.scale",
            "layers.14.engram.embed.weight", "layers.14.engram.embed.scale"]
missing = [k for k in required if k not in weight_map]
if missing:
    raise SystemExit("REFUSING: required keys absent from shards: %s" % missing)

json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
          open(out, "w"), indent=1)
print()
print("wrote %s  (%d keys, total_size %.2f GiB)" % (out, len(weight_map), total/1024**3))
for k in required:
    print("  OK %-40s -> %s" % (k, weight_map[k]))
