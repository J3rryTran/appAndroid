# Soi input/output shape + op_types của file ONNX mà KHÔNG cần cài onnx/onnxruntime.
# Dùng: python -X utf8 tools/onnx_inspect.py <path.onnx>
import sys, collections

data = open(sys.argv[1], "rb").read()

def rd_varint(b, i):
    shift = 0; val = 0
    while True:
        x = b[i]; i += 1
        val |= (x & 0x7f) << shift
        if not (x & 0x80): break
        shift += 7
    return val, i

def fields(b, start, end):
    i = start
    while i < end:
        tag, i = rd_varint(b, i)
        fn = tag >> 3; wt = tag & 7
        if wt == 0:
            v, i = rd_varint(b, i); yield fn, ('v', v)
        elif wt == 2:
            ln, i = rd_varint(b, i); yield fn, ('b', b[i:i+ln]); i += ln
        elif wt == 1:
            yield fn, ('f', b[i:i+8]); i += 8
        elif wt == 5:
            yield fn, ('f', b[i:i+4]); i += 4
        else:
            raise ValueError("wt %d" % wt)

def parse_shape(buf):
    dims = []
    for fn, (t, v) in fields(buf, 0, len(buf)):
        if fn == 1 and t == 'b':
            dv = '?'
            for f2, (t2, v2) in fields(v, 0, len(v)):
                if f2 == 1 and t2 == 'v': dv = v2
                elif f2 == 2 and t2 == 'b': dv = v2.decode('utf-8', 'replace')
            dims.append(dv)
    return dims

def parse_valueinfo(buf):
    name = '?'; shape = []
    for fn, (t, v) in fields(buf, 0, len(buf)):
        if fn == 1 and t == 'b': name = v.decode('utf-8', 'replace')
        elif fn == 2 and t == 'b':
            for f2, (t2, v2) in fields(v, 0, len(v)):
                if f2 == 1 and t2 == 'b':
                    for f3, (t3, v3) in fields(v2, 0, len(v2)):
                        if f3 == 2 and t3 == 'b':
                            shape = parse_shape(v3)
    return name, shape

def parse_node(buf):
    op = '?'; outs = []
    for fn, (t, v) in fields(buf, 0, len(buf)):
        if fn == 2 and t == 'b': outs.append(v.decode('utf-8','replace'))
        elif fn == 4 and t == 'b': op = v.decode('utf-8','replace')
    return op, outs

graph = None
for fn, (t, v) in fields(data, 0, len(data)):
    if fn == 7 and t == 'b': graph = v

inputs = []; outputs = []; ops = collections.Counter(); node_seq = []; init_names = set()
for fn, (t, v) in fields(graph, 0, len(graph)):
    if fn == 11 and t == 'b': inputs.append(parse_valueinfo(v))
    elif fn == 12 and t == 'b': outputs.append(parse_valueinfo(v))
    elif fn == 1 and t == 'b':
        op, outs = parse_node(v); ops[op] += 1; node_seq.append((op, outs))
    elif fn == 5 and t == 'b':
        for f2,(t2,v2) in fields(v,0,len(v)):
            if f2 == 8 and t2 == 'b': init_names.add(v2.decode('utf-8','replace'))

print("=== INPUTS ===")
for n, s in inputs:
    if n not in init_names: print(f"  {n}: {s}")
print("=== OUTPUTS ===")
for n, s in outputs: print(f"  {n}: {s}")
print("=== OP TYPES ===")
for op, c in ops.most_common(): print(f"  {op}: {c}")
