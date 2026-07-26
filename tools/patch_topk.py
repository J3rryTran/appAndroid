# Sua loi OpenCV DNN "TopK: K is out of range (can K < N)" voi ONNX YOLO26/v10 e2e.
# - K rieng tung TopK: doi 300 -> 299 tai cho.
# - K DUNG CHUNG (constant dedup): nhan ban initializer thanh ban 299 (ten cung do dai),
#   tro TopK cuoi sang ban moi, cap nhat length cua GraphProto.
# Dung: python -X utf8 tools/patch_topk.py <model.onnx> [--apply]
import struct
import sys

path = sys.argv[1]
apply = "--apply" in sys.argv
data = bytearray(open(path, "rb").read())

def rd_varint(b, i):
    shift = 0; val = 0
    while True:
        x = b[i]; i += 1
        val |= (x & 0x7f) << shift
        if not (x & 0x80): break
        shift += 7
    return val, i

def enc_varint(v):
    out = bytearray()
    while True:
        x = v & 0x7f
        v >>= 7
        if v:
            out.append(x | 0x80)
        else:
            out.append(x)
            return bytes(out)

def walk(b, start, end):
    i = start
    while i < end:
        tag, i = rd_varint(b, i)
        fn, wt = tag >> 3, tag & 7
        if wt == 0:
            vs = i
            v, i = rd_varint(b, i)
            yield fn, wt, vs, i, v
        elif wt == 2:
            ln, i = rd_varint(b, i)
            yield fn, wt, i, i + ln, None
            i += ln
        elif wt == 1:
            yield fn, wt, i, i + 8, None; i += 8
        elif wt == 5:
            yield fn, wt, i, i + 4, None; i += 4
        else:
            raise ValueError("wire type %d @ %d" % (wt, i))

gs = ge = None
for fn, wt, s, e, v in walk(data, 0, len(data)):
    if fn == 7 and wt == 2:
        gs, ge = s, e

topks = []      # (node_name, k_name, k_input_span)
inits = {}      # name -> (kind, value_off, value, entry_span)
for fn, wt, s, e, v in walk(data, gs, ge):
    if fn == 1 and wt == 2:
        op = None; name = ""; ins = []
        for f2, w2, s2, e2, v2 in walk(data, s, e):
            if f2 == 1 and w2 == 2: ins.append((bytes(data[s2:e2]).decode("utf-8", "replace"), (s2, e2)))
            elif f2 == 3 and w2 == 2: name = bytes(data[s2:e2]).decode("utf-8", "replace")
            elif f2 == 4 and w2 == 2: op = bytes(data[s2:e2]).decode("utf-8", "replace")
        if op == "TopK" and len(ins) >= 2:
            topks.append((name, ins[1][0], ins[1][1]))
    elif fn == 5 and wt == 2:
        name = None; dtype = None; enc = None
        for f2, w2, s2, e2, v2 in walk(data, s, e):
            if f2 == 8 and w2 == 2: name = bytes(data[s2:e2]).decode("utf-8", "replace")
            elif f2 == 2 and w2 == 0: dtype = v2
            elif f2 == 9 and w2 == 2: enc = ("raw", s2)
            elif f2 == 7 and w2 in (0, 2): enc = ("varint", s2)
        if name and enc and dtype == 7:
            kind, off = enc
            val = struct.unpack("<q", bytes(data[off:off + 8]))[0] if kind == "raw" \
                  else rd_varint(data, off)[0]
            inits[name] = (kind, off, val, (s, e))

print("TopK nodes:")
for nname, kname, _ in topks:
    ini = inits.get(kname)
    print("  %s  K=%s  value=%s" % (nname, kname, ini[2] if ini else "?"))

knames = [k for _, k, _ in topks]
shared = len(knames) != len(set(knames)) and len(topks) >= 2

if not shared:
    patched = 0
    for nname, kname, _ in topks:
        if kname not in inits: continue
        kind, off, val, _span = inits[kname]
        if val != 300: continue
        if apply:
            if kind == "raw":
                data[off:off + 8] = struct.pack("<q", 299)
            else:
                data[off] = 0xAB
        print("%s %s: 300 -> 299" % ("PATCHED" if apply else "SE VA", nname))
        patched += 1
    if apply and patched:
        open(path, "wb").write(data)
        print("Da ghi %s" % path)
    sys.exit(0)

# ---- K dung chung: nhan ban initializer, tro TopK CUOI sang ban moi ----
last_name, kname, (is_, ie_) = topks[-1]
kind, voff, val, (es, ee) = inits[kname]
if val != 300:
    print("K dung chung nhung value=%s (!=300) — khong ro cach va, dung lai." % val)
    sys.exit(2)

new_name = kname[:-1] + ("9" if kname[-1] != "9" else "8")
if new_name in inits:
    print("Ten moi %s da ton tai — dung lai." % new_name)
    sys.exit(2)

entry = bytearray(data[es:ee])
fixed_name = fixed_val = False
for f2, w2, s2, e2, v2 in walk(entry, 0, len(entry)):
    if f2 == 8 and w2 == 2 and bytes(entry[s2:e2]).decode("utf-8", "replace") == kname:
        entry[e2 - 1] = ord(new_name[-1]); fixed_name = True
    elif f2 == 9 and w2 == 2 and kind == "raw":
        entry[s2:s2 + 8] = struct.pack("<q", 299); fixed_val = True
    elif f2 == 7 and w2 in (0, 2) and kind == "varint":
        entry[s2] = 0xAB; fixed_val = True
if not (fixed_name and fixed_val):
    print("Khong sua duoc ban sao (name=%s, val=%s) — dung lai." % (fixed_name, fixed_val))
    sys.exit(2)

print("Ke hoach: %s.input[K] -> %s (299); giu %s = 300 cho TopK dau." % (last_name, new_name, kname))
if not apply:
    print("Chay lai voi --apply de ghi.")
    sys.exit(0)

data[ie_ - 1] = ord(new_name[-1])                      # doi ten K-input cua TopK cuoi

old_len = ge - gs
hdr = gs - len(enc_varint(old_len)) - 1
assert data[hdr] == 0x3A, "khong tim thay header GraphProto"
appended = b"\x2a" + enc_varint(len(entry)) + bytes(entry)
new_graph = bytes(data[gs:ge]) + appended
out = bytes(data[:hdr]) + b"\x3a" + enc_varint(len(new_graph)) + new_graph + bytes(data[ge:])
open(path, "wb").write(out)
print("Da ghi %s (them %d bytes)" % (path, len(out) - len(data)))
