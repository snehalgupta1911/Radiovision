"""
Robust CSI capture / validation for Radiovision.

Reads the RX serial stream (format: CSI,seq,ts,rssi,len,[i r i r ...]),
validates every frame, discards corrupt ones, and uses the seq counter to
report the TRUE drop rate (corrupt frames + frames that never arrived).
Clean frames are written as .npz ready for preprocess.resample_to_grid.

Usage (live):    python -m radiovision_ml.capture --port /dev/ttyACM0 --out session1
Usage (offline): python -m radiovision_ml.capture --file mylog.txt --out session1
"""
from __future__ import annotations
import argparse, re, time
import numpy as np


HDR5 = re.compile(r'CSI,(\d+),(\d+),(-?\d+),(\d+),\[([^\]]*)\]')   # seq,ts,rssi,len
HDR4 = re.compile(r'CSI,(\d+),(-?\d+),(\d+),\[([^\]]*)\]')          # ts,rssi,len

def parse_frame(text):
    m = HDR5.search(text)
    if m:
        seq, ts, rssi, ln, body = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5]
    else:
        m = HDR4.search(text)
        if not m:
            return None
        seq = -1
        ts, rssi, ln, body = int(m[1]), int(m[2]), int(m[3]), m[4]
    toks = body.split()
    if len(toks) != ln:
        return None
    try:
        raw = np.array([int(t) for t in toks], dtype=np.int8)
    except ValueError:
        return None
    if raw.size % 2 != 0:
        return None
    csi = (raw[1::2].astype(np.float32) + 1j*raw[0::2].astype(np.float32)).astype(np.complex64)
    return seq, ts, rssi, csi

def run(lines, out):
    kept, corrupt = [], 0
    first_seq = last_seq = None
    for line in lines:
        if 'CSI,' not in line:
            continue
        f = parse_frame(line)
        if f is None:
            corrupt += 1
            continue
        seq, ts, rssi, csi = f
        if first_seq is None: first_seq = seq
        last_seq = seq
        kept.append((seq, ts, rssi, csi))

    if not kept:
        print("No valid frames. Check firmware/wiring."); return
    span = (last_seq - first_seq + 1)
    missing = span - len(kept) - corrupt            # frames that never arrived
    total_attempted = span
    print(f"clean frames        : {len(kept)}")
    print(f"corrupt (dropped)   : {corrupt}")
    print(f"missing seq (never)  : {max(missing,0)}")
    print(f"TRUE yield          : {100*len(kept)/max(total_attempted,1):.1f}%  "
          f"(target >95%; if low, lower EMIT_HZ or switch to binary framing)")
    # subcarrier-count sanity
    lens = {}
    for _,_,_,c in kept: lens[c.shape[0]] = lens.get(c.shape[0],0)+1
    print(f"subcarrier counts   : {lens}  (lock preprocess to the dominant one)")

    if out:
        seqs = np.array([k[0] for k in kept])
        ts   = np.array([k[1] for k in kept])
        rssi = np.array([k[2] for k in kept])
        dom  = max(lens, key=lens.get)
        csi  = np.stack([k[3] for k in kept if k[3].shape[0]==dom])
        np.savez(out+'.npz', seq=seqs, ts=ts, rssi=rssi, csi=csi)
        print(f"saved -> {out}.npz  (csi shape {csi.shape})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port'); ap.add_argument('--file'); ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--out'); ap.add_argument('--seconds', type=float, default=60)
    a = ap.parse_args()
    if a.file:
        run(open(a.file, errors='ignore'), a.out)
    else:
        import serial
        ser = serial.Serial(a.port, a.baud, timeout=1)
        t0 = time.time(); buf = []
        while time.time()-t0 < a.seconds:
            buf.append(ser.readline().decode('utf-8','ignore'))
        run(buf, a.out)

if __name__ == '__main__':
    main()