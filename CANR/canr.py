import re, struct, math, argparse, sys
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

LINE_RE = re.compile(
    r"^\s*\((?P<ts>\d+\.\d+)\)\s+\S+\s+"
    r"(?P<id>[0-9A-Fa-f]+)\s+\[\d+\]\s+"
    r"(?P<data>[0-9A-Fa-f\s]+)$"
)
TARGET_IDS = {"0B0", "0A7", "0A6"}

def parse_bytes(hex_str):
    return bytes(int(b, 16) for b in hex_str.strip().split())

def s16le(data, offset):
    return struct.unpack_from("<h", data, offset)[0]

def rpm_to_rad_s(rpm):
    return rpm * 2.0 * math.pi / 60.0

def iter_relevant_lines(filepath):
    with open(filepath, "r", errors="replace") as fh:
        for line in fh:
            m = LINE_RE.match(line)
            if not m:
                continue
            can_id = m.group("id").upper().zfill(3)
            if can_id not in TARGET_IDS:
                continue
            yield {"ts": float(m.group("ts")), "can_id": can_id,
                   "data": parse_bytes(m.group("data"))}

def process(filepath, output_csv=None, speed_threshold=100.0):
    latest = {"0B0": None, "0A7": None, "0A6": None}
    results = []

    for msg in iter_relevant_lines(filepath):
        cid, data, ts = msg["can_id"], msg["data"], msg["ts"]

        if cid == "0B0" and len(data) >= 8:
            latest["0B0"] = {
                "torque_cmd_nm":    s16le(data, 0) / 10.0,
                "torque_fb_nm":     s16le(data, 2) / 10.0,
                "speed_rpm":        float(s16le(data, 4)),   # Angular Velocity, no scaling
                "dc_bus_voltage_v": s16le(data, 6) / 10.0,
            }
        elif cid == "0A7" and len(data) >= 2:
            latest["0A7"] = {"dc_bus_voltage_v": s16le(data, 0) / 10.0}
        elif cid == "0A6" and len(data) >= 8:
            latest["0A6"] = {
                "phase_a_a":    s16le(data, 0) / 10.0,
                "phase_b_a":    s16le(data, 2) / 10.0,
                "phase_c_a":    s16le(data, 4) / 10.0,
                "dc_current_a": s16le(data, 6) / 10.0,      # INV_DC_Bus_Current
            }

        b0 = latest["0B0"]
        if b0 is None:
            continue

        voltage = (latest["0A7"]["dc_bus_voltage_v"]
                   if latest["0A7"] else b0["dc_bus_voltage_v"])
        current = latest["0A6"]["dc_current_a"] if latest["0A6"] else None

        torque_fb = b0["torque_fb_nm"]
        speed     = b0["speed_rpm"]
        omega     = rpm_to_rad_s(speed)
        p_mech    = torque_fb * omega
        p_elec    = voltage * current if current is not None else None

        if current is not None and abs(speed) > speed_threshold and p_elec != 0.0:
            efficiency = (p_mech / p_elec) * 100.0
        else:
            efficiency = None

        results.append({
            "timestamp":        ts,
            "torque_cmd_nm":    round(b0["torque_cmd_nm"], 4),
            "torque_fb_nm":     round(torque_fb, 4),
            "speed_rpm":        speed,
            "omega_rad_s":      round(omega, 6),
            "dc_bus_voltage_v": round(voltage, 4),
            "dc_current_a":     round(current, 4) if current is not None else None,
            "p_mech_w":         round(p_mech, 4),
            "p_elec_w":         round(p_elec, 4) if p_elec is not None else None,
            "efficiency_pct":   round(efficiency, 4) if efficiency is not None else None,
        })

    if not results:
        print("WARNING: No matching CAN frames found.")
        return None

    print_summary(results, speed_threshold)
    if output_csv:
        save_csv(results, output_csv)
        print("\nSaved: {}".format(output_csv))
    return pd.DataFrame(results) if HAS_PANDAS else results

def print_summary(rows, speed_threshold=100.0):
    eff_rows = [r for r in rows if r["efficiency_pct"] is not None]
    def avg(k):
        vals = [r[k] for r in rows if r[k] is not None]
        return sum(vals) / len(vals) if vals else float("nan")
    print("=" * 57)
    print("  Total rows              : {:,}".format(len(rows)))
    print("  Efficiency rows         : {:,}  (|speed| > {} RPM)".format(len(eff_rows), int(speed_threshold)))
    print("  Avg Torque Command      : {:.2f} N.m".format(avg("torque_cmd_nm")))
    print("  Avg Torque Feedback     : {:.2f} N.m".format(avg("torque_fb_nm")))
    print("  Avg Motor Speed         : {:.1f} RPM".format(avg("speed_rpm")))
    print("  Avg DC Bus Voltage      : {:.2f} V".format(avg("dc_bus_voltage_v")))
    print("  Avg DC Current          : {:.2f} A".format(avg("dc_current_a")))
    print("  Avg P_mech              : {:.2f} W".format(avg("p_mech_w")))
    print("  Avg P_elec              : {:.2f} W".format(avg("p_elec_w")))
    if eff_rows:
        avg_eff = sum(r["efficiency_pct"] for r in eff_rows) / len(eff_rows)
        max_eff = max(r["efficiency_pct"] for r in eff_rows)
        print("  Avg Efficiency          : {:.2f} %".format(avg_eff))
        print("  Peak Efficiency         : {:.2f} %".format(max_eff))
    else:
        print("  Efficiency              : 0 rows (motor may not have exceeded threshold)")
    print("=" * 57)

def save_csv(rows, path):
    if HAS_PANDAS:
        pd.DataFrame(rows).to_csv(path, index=False); return
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="Parse BorgWarner/Cascadia candump log")
    ap.add_argument("logfile")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--speed-threshold", type=float, default=100.0)
    args = ap.parse_args()
    p = Path(args.logfile)
    if not p.exists():
        print("File not found: {}".format(p), file=sys.stderr); sys.exit(1)
    print("Parsing: {}  ({:.1f} MB)".format(p, p.stat().st_size / 1e6))
    print("Speed threshold: > {} RPM\n".format(args.speed_threshold))
    process(str(p), output_csv=args.output, speed_threshold=args.speed_threshold)

if __name__ == "__main__":
    main()
