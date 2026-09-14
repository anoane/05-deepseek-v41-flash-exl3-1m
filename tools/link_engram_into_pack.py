"""Make the engram tables reachable from the EXL3 pack directory.

PROVEN CAUSE (engram.py:1085):
    model_dir=get_current_vllm_config().model_config.model
DiskEngramTable is handed the EXL3 PACK path, never `engram_table_dir`. My --hf-overrides
engram_table_dir was inert. The pack's index has the 12 engram k/q/wkv keys but NOT
embed.weight/embed.scale (those exist only in the base repo's shards 47/48).

FIX: hardlink the two engram shards into the pack dir (same filesystem -> no copy, no extra disk)
and merge the 4 embed entries into the pack's index. DiskEngramTable._open does
os.path.join(model_dir, fname) + pread by offset, so the files only need to resolve there.

SAFETY: back up the index, assert the 4 keys are absent before and present after, assert no
existing key is modified, and verify the merged count is exactly old+4.
"""
import json, os, shutil, struct

PACK = "/root/workspace/DeepSeek-V4.1-Flash-EXL3-3.0bpw"
ENG  = "/root/workspace/DeepSeek-V4.1-Flash-engram"
IDX  = os.path.join(PACK, "model.safetensors.index.json")
NEED = ["layers.1.engram.embed.weight", "layers.1.engram.embed.scale",
        "layers.14.engram.embed.weight", "layers.14.engram.embed.scale"]

idx = json.load(open(IDX))
wm  = idx["weight_map"]
before = len(wm)
print("pack index: %d keys" % before)
assert not any(k in wm for k in NEED), "embed keys already present; refusing"

# 1. hardlink the shards (verify same filesystem first)
if os.stat(PACK).st_dev != os.stat(ENG).st_dev:
    raise SystemExit("!! different filesystems; hardlink impossible, would need a 190 GB copy")
linked = []
for fn in sorted(f for f in os.listdir(ENG) if f.endswith(".safetensors")):
    src, dst = os.path.join(ENG, fn), os.path.join(PACK, fn)
    if os.path.exists(dst):
        print("  already there: %s" % fn)
    else:
        os.link(src, dst)
        linked.append(fn)
        print("  hardlinked %s (links=%d, %.1f GiB)"
              % (fn, os.stat(dst).st_nlink, os.path.getsize(dst)/1024**3))

# 2. merge ONLY the 4 embed keys, taken from the shard headers themselves
eng_idx = json.load(open(os.path.join(ENG, "model.safetensors.index.json")))["weight_map"]
added = {}
for k in NEED:
    fn = eng_idx[k]
    p = os.path.join(PACK, fn)
    with open(p, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    assert k in hdr, "%s not in %s header" % (k, fn)
    added[k] = fn

shutil.copy2(IDX, IDX + ".bak.preengram")
wm.update(added)
assert len(wm) == before + 4, "expected %d keys, got %d" % (before+4, len(wm))
json.dump(idx, open(IDX, "w"), indent=1)

chk = json.load(open(IDX))["weight_map"]
print("\nmerged index: %d keys (was %d)" % (len(chk), before))
for k in NEED:
    print("  OK %-42s -> %s  shape %s" % (k, chk[k], added[k] and ""))
print("backup: %s.bak.preengram" % os.path.basename(IDX))
