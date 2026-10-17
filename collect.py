"""
collect.py - record labelled gesture data: RX CSI (your existing serial format) + webcam teacher.

    python collect.py --check --port /dev/cu.usbmodem101            # verify CSI only
    python collect.py --list-cams                                     # find the Logitech camera
    python collect.py --gesture hands_up --port /dev/cu.usbmodem101 --cam 1 --session s01

In the window:
    SPACE   start a recording (3 s countdown, then records --duration seconds, then stops)
            press SPACE again during a recording to stop early
    1..5    switch gesture (1 idle, 2 hands_crossed, 3 hands_up, 4 arm_side, 5 arm_forward)
    q       quit

Every recording is saved as  dataset/<gesture>/<session>_<gesture>_<YYYYmmdd_HHMMSS>/
    csi.npz      every CSI line: Mac arrival time, seq, radio timestamp (us), rssi, raw int8 values
    camera.npz   every camera frame: Mac time, 33 MediaPipe landmarks, 14 joints, pose-found flag,
                 14 heatmaps (uint8, 32x32)
    video.mp4    320x240 preview of the recording (for visualize.py)
    meta.json    gesture, label index, session, recording id, rates, counts, settings

Use --no-camera to record CSI only (usable for the lambda = 0 baseline).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import common as C
import heatmaps as HM


def list_cams(cv2):
    for i in range(6):
        cap = cv2.VideoCapture(i)
        ok, f = cap.read()
        print(f"  --cam {i}: " + (f"{f.shape[1]}x{f.shape[0]}" if ok else "not available"))
        cap.release()
    print("On a MacBook, a USB Logitech webcam is usually --cam 1 (0 = built-in camera).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gesture", choices=C.CLASSES)
    ap.add_argument("--port", help="RX board serial port, e.g. /dev/cu.usbmodem101")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--no-camera", action="store_true", help="record CSI only")
    ap.add_argument("--session", default=time.strftime("s%Y%m%d"),
                    help="session id (e.g. a day / place / person). Train/test splits are by session.")
    ap.add_argument("--subject", default="me")
    ap.add_argument("--duration", type=float, default=20.0, help="seconds per recording")
    ap.add_argument("--countdown", type=float, default=3.0)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--teacher", choices=["full", "lite"], default="full")
    ap.add_argument("--check", action="store_true", help="only print CSI statistics")
    ap.add_argument("--list-cams", action="store_true")
    a = ap.parse_args()

    import cv2
    if a.list_cams:
        list_cams(cv2); return
    if not a.port:
        ap.error("--port is required")

    reader = C.SerialCSIReader(a.port, a.baud)
    reader.start()
    if a.check:
        print("CSI check (Ctrl+C to stop). Expect ~80-100 Hz, malformed ~0, subcarriers 192.")
        try:
            while True:
                time.sleep(1)
                print(reader.status() + (f"   last other line: {reader.last_text!r}" if reader.last_text else ""))
        except KeyboardInterrupt:
            pass
        return
    if not a.gesture:
        ap.error("--gesture is required (one of: " + ", ".join(C.CLASSES) + ")")

    print("waiting for CSI...")
    C.wait_for_csi(reader)
    print(reader.status())

    cap, teacher = None, None
    if not a.no_camera:
        teacher = HM.CameraTeacher(a.teacher)
        cap = cv2.VideoCapture(a.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.read()[0]:
            raise SystemExit(f"camera {a.cam} not readable. Run --list-cams, and allow camera access for "
                             "Terminal / VS Code in System Settings > Privacy & Security > Camera.")

    gesture = a.gesture
    state, t_state = "idle", 0.0          # idle -> countdown -> recording -> idle
    rec = None
    n_saved = 0
    t0 = time.monotonic()
    print("SPACE = record, 1-5 = change gesture, q = quit")

    def save(rec):
        rows = reader.stop_recording()
        if rec["writer"] is not None:
            rec["writer"].release()
        dur = rec["t_stop"] - rec["t_start"]
        if len(rows) < C.CSI_FS * 1.5:
            print(f"  NOT saved: only {len(rows)} CSI samples"); return False
        d = rec["dir"]
        C.save_csi_rows(rows, os.path.join(d, "csi.npz"))
        if cap is not None:
            kp33 = np.stack(rec["kp33"]) if rec["kp33"] else np.zeros((0, 33, 4), np.float32)
            ok = np.array(rec["ok"], bool)
            joints = np.stack([HM.mp33_to_joints(k) for k in kp33]) if len(kp33) else np.zeros((0, HM.J, 3), np.float32)
            joints[~ok] = 0
            np.savez_compressed(os.path.join(d, "camera.npz"), host_t=np.array(rec["t"]), kp33=kp33,
                                joints=joints, ok=ok, heatmaps=np.stack(rec["hm"]),
                                hm_size=HM.HM_SIZE, hm_sigma=HM.HM_SIGMA, joint_names=np.array(HM.JOINTS))
        t_rows = np.array([r[0] for r in rows])
        meta = {"gesture": rec["gesture"], "label": C.CLASS_TO_IDX[rec["gesture"]], "classes": C.CLASSES,
                "session": a.session, "subject": a.subject, "recording_id": os.path.basename(d),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "duration_s": round(dur, 2),
                "csi_samples": len(rows), "csi_rate_hz": round(len(rows) / max(dur, 1e-6), 1),
                "csi_subcarriers": reader.n_sub, "csi_malformed_total": reader.n_bad,
                "camera_frames": len(rec["t"]), "pose_frames": int(sum(rec["ok"])),
                "camera": None if cap is None else a.cam, "teacher": None if cap is None else a.teacher,
                "serial_port": a.port, "csi_first_t": float(t_rows[0]), "csi_last_t": float(t_rows[-1]),
                "time_base": "time.monotonic() on the Mac, shared by CSI and camera"}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
        msg = (f"  saved {d}: {dur:.1f}s, CSI {len(rows)} ({meta['csi_rate_hz']} Hz)")
        if cap is not None:
            msg += f", camera {len(rec['t'])} frames, pose in {meta['pose_frames']}"
            if meta["pose_frames"] < 0.8 * max(1, len(rec["t"])):
                msg += "  <- WARNING: pose missing often, keep your whole upper body in view"
        print(msg)
        return True

    while True:
        frame = None
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                print("camera stopped"); break
        else:
            time.sleep(1 / 30)
            frame = np.zeros((480, 640, 3), np.uint8)
        now = time.monotonic()
        kp33 = None
        if teacher is not None:
            kp33 = teacher(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (now - t0) * 1000)
        joints = HM.mp33_to_joints(kp33) if kp33 is not None else None

        # ---- state machine ----
        if state == "countdown" and now >= t_state:
            sess_dir = os.path.join(a.out, gesture, f"{a.session}_{gesture}_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(sess_dir, exist_ok=True)
            writer = None
            if cap is not None:
                writer = cv2.VideoWriter(os.path.join(sess_dir, "video.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
            rec = {"dir": sess_dir, "gesture": gesture, "t_start": now, "t": [], "kp33": [], "ok": [],
                   "hm": [], "writer": writer, "frames_t": []}
            reader.start_recording()
            state = "recording"
            print(f"RECORDING {gesture} -> {sess_dir}")
        if state == "recording":
            if cap is not None:
                rec["t"].append(now); rec["ok"].append(kp33 is not None)
                rec["kp33"].append(kp33 if kp33 is not None else np.zeros((33, 4), np.float32))
                hm = HM.make_heatmaps(joints) if joints is not None else np.zeros((HM.J, HM.HM_SIZE, HM.HM_SIZE))
                rec["hm"].append((hm * 255 + 0.5).astype(np.uint8))
                rec["writer"].write(cv2.resize(frame, (320, 240)))
            if now - rec["t_start"] >= a.duration:
                rec["t_stop"] = now
                n_saved += save(rec); rec = None; state = "idle"

        # ---- display ----
        view = cv2.flip(frame, 1)
        if joints is not None:
            m = joints.copy(); m[:, 0] = 1 - m[:, 0]
            HM.draw_skeleton(cv2, view, m, (0, 255, 0))
        hm_img = cv2.flip(HM.heatmap_image(cv2, HM.make_heatmaps(joints) if joints is not None
                                           else np.zeros((HM.J, HM.HM_SIZE, HM.HM_SIZE)), 240), 1)
        view[-240:, -240:] = hm_img
        C.put_text(cv2, view, f"Gesture: {gesture.upper().replace('_', ' ')}", (12, 40), 1.1, (0, 255, 255), 2)
        if state == "recording":
            el = now - rec["t_start"]
            cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (0, 0, 255), 8)
            C.put_text(cv2, view, f"Recording...  {el:4.1f} / {a.duration:.0f} s", (12, 80), 0.9, (0, 0, 255), 2)
            C.put_text(cv2, view, f"CSI samples: {reader.n_recorded()}", (12, 115), 0.7)
            C.put_text(cv2, view, f"Camera frames: {len(rec['t'])}   with pose: {sum(rec['ok'])}", (12, 145), 0.7)
        elif state == "countdown":
            C.put_text(cv2, view, f"GET READY: hold {gesture.replace('_', ' ')}   {t_state - now:3.1f}",
                       (12, 80), 0.9, (0, 255, 255), 2)
        else:
            C.put_text(cv2, view, "Ready - press SPACE to record   (1-5 change gesture, q quit)",
                       (12, 80), 0.6, (0, 255, 0))
        if cap is not None and kp33 is None:
            C.put_text(cv2, view, "NO PERSON DETECTED", (12, 180), 0.8, (0, 0, 255), 2)
        C.put_text(cv2, view, f"{reader.status()}   saved this run: {n_saved}   session: {a.session}",
                   (12, view.shape[0] - 250), 0.5, (255, 255, 0))
        cv2.imshow("CSI gesture collector", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if state == "idle":
                state, t_state = "countdown", now + a.countdown
            elif state == "recording":
                rec["t_stop"] = now; n_saved += save(rec); rec = None; state = "idle"
        if state == "idle" and ord("1") <= key <= ord("5"):
            gesture = C.CLASSES[key - ord("1")]; print(f"gesture -> {gesture}")

    if state == "recording":
        rec["t_stop"] = time.monotonic(); save(rec)
    reader.stop()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"done, {n_saved} recordings saved")


if __name__ == "__main__":
    main()
