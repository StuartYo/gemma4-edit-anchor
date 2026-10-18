"""Copy a Windows PE executable and set its default thread stack reserve.

The stock llama.cpp b11215 llama-server.exe (1 MB reserve) dies with 0xC00000FD (stack
overflow in llama-common.dll) while parsing very long Gemma 4 tool-call strings. The
paper's runs used a copy with a 256 MB reserve; nothing else in the file is changed.

usage: python tools/set_stack_reserve.py <llama-server.exe> <out.exe> [MB=256]
"""
import shutil, struct, sys

src, dst = sys.argv[1], sys.argv[2]
mb = int(sys.argv[3]) if len(sys.argv) > 3 else 256
shutil.copyfile(src, dst)
with open(dst, "r+b") as f:
    f.seek(0x3C); pe = struct.unpack("<I", f.read(4))[0]
    f.seek(pe); assert f.read(4) == b"PE\0\0", "not a PE file"
    opt = pe + 4 + 20                      # optional header follows the 20-byte COFF header
    f.seek(opt); magic = struct.unpack("<H", f.read(2))[0]
    off, fmt = (opt + 72, "<Q") if magic == 0x20B else (opt + 72, "<I")   # PE32+ / PE32
    f.seek(off); old = struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    f.seek(off); f.write(struct.pack(fmt, mb * 1024 * 1024))
print(f"SizeOfStackReserve {old} -> {mb * 1024 * 1024} bytes in {dst}")
