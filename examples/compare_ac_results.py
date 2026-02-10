#!/usr/bin/env python3
"""
Compare AC analysis results between pycircuitsim and NGSPICE.
"""
import pandas as pd
import numpy as np

# Load pycircuitsim results
pycircuit_data = pd.read_csv('../results/rc_lowpass_ac/ac/rc_lowpass_ac_ac_sweep.csv')

# Load NGSPICE results (parse the text file)
ngspice_data = []
with open('rc_lowpass_ngspice.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        # Skip header lines and separators
        if line.startswith('-') or line.startswith(' '):
            continue
        if 'Index' in line or 'frequency' in line:
            continue

        parts = line.split()
        if len(parts) >= 4:
            try:
                # Index, frequency, vdb(2), vp(2)
                idx = int(parts[0])
                freq = float(parts[1])
                vdb = float(parts[2])
                vp_rad = float(parts[3])  # NGSPICE outputs phase in radians
                vp_deg = np.rad2deg(vp_rad)
                ngspice_data.append([freq, vdb, vp_deg])
            except ValueError:
                continue

ngspice_df = pd.DataFrame(ngspice_data, columns=['frequency', 'vdb', 'vp_deg'])

print("=" * 80)
print("AC Analysis Comparison: pycircuitsim vs NGSPICE")
print("=" * 80)
print(f"\nRC Low-Pass Filter: R=1kΩ, C=100nF, fc=1.59kHz")
print(f"pycircuitsim: {len(pycircuit_data)} points")
print(f"NGSPICE:      {len(ngspice_df)} points")
print()

# Find common frequencies (or closest matches)
print("Comparison at selected frequencies:")
print("-" * 80)
print(f"{'Frequency':<12} {'PyCircuit':<20} {'NGSPICE':<20} {'Δ Mag':<12} {'Δ Phase':<12}")
print(f"{'(Hz)':<12} {'Mag(dB) / Phase(°)':<20} {'Mag(dB) / Phase(°)':<20} {'(dB)':<12} {'(degrees)':<12}")
print("-" * 80)

# Sample a few key frequencies
test_frequencies = [100, 1000, 1590, 10000, 100000]

for target_freq in test_frequencies:
    # Find closest frequency in pycircuitsim results
    pycircuit_idx = (pycircuit_data['frequency'] - target_freq).abs().idxmin()
    pycircuit_freq = pycircuit_data.loc[pycircuit_idx, 'frequency']
    pycircuit_mag = pycircuit_data.loc[pycircuit_idx, 'V(2)_mag']
    pycircuit_mag_db = 20 * np.log10(pycircuit_mag)
    pycircuit_phase = pycircuit_data.loc[pycircuit_idx, 'V(2)_phase']

    # Find closest frequency in NGSPICE results
    ngspice_idx = (ngspice_df['frequency'] - target_freq).abs().idxmin()
    ngspice_freq = ngspice_df.loc[ngspice_idx, 'frequency']
    ngspice_mag_db = ngspice_df.loc[ngspice_idx, 'vdb']
    ngspice_phase = ngspice_df.loc[ngspice_idx, 'vp_deg']

    # Calculate differences
    delta_mag = pycircuit_mag_db - ngspice_mag_db
    delta_phase = pycircuit_phase - ngspice_phase

    print(f"{pycircuit_freq:<12.1f} {pycircuit_mag_db:>8.3f} / {pycircuit_phase:>6.2f}   "
          f"{ngspice_mag_db:>8.3f} / {ngspice_phase:>6.2f}   "
          f"{delta_mag:>8.3f}     {delta_phase:>8.3f}")

print("-" * 80)

# Calculate overall statistics
# Interpolate pycircuitsim results to NGSPICE frequencies for comparison
from scipy.interpolate import interp1d

# Convert pycircuitsim mag to dB
pycircuit_mag_db_array = 20 * np.log10(pycircuit_data['V(2)_mag'].values)

# Create interpolation functions
interp_mag = interp1d(pycircuit_data['frequency'], pycircuit_mag_db_array,
                      kind='linear', fill_value='extrapolate')
interp_phase = interp1d(pycircuit_data['frequency'], pycircuit_data['V(2)_phase'].values,
                        kind='linear', fill_value='extrapolate')

# Interpolate at NGSPICE frequencies
pycircuit_mag_db_interp = interp_mag(ngspice_df['frequency'].values)
pycircuit_phase_interp = interp_phase(ngspice_df['frequency'].values)

# Calculate errors
mag_errors = pycircuit_mag_db_interp - ngspice_df['vdb'].values
phase_errors = pycircuit_phase_interp - ngspice_df['vp_deg'].values

print(f"\nOverall Statistics:")
print(f"  Magnitude error (dB):  Mean = {np.mean(np.abs(mag_errors)):.4f} dB, "
      f"Max = {np.max(np.abs(mag_errors)):.4f} dB")
print(f"  Phase error (degrees): Mean = {np.mean(np.abs(phase_errors)):.4f}°, "
      f"Max = {np.max(np.abs(phase_errors)):.4f}°")

# Validation criteria
print(f"\nValidation Criteria:")
mag_pass = np.max(np.abs(mag_errors)) < 1.0  # Within 1 dB
phase_pass = np.max(np.abs(phase_errors)) < 5.0  # Within 5 degrees
print(f"  Magnitude within 1 dB:    {'✓ PASS' if mag_pass else '✗ FAIL'}")
print(f"  Phase within 5 degrees:   {'✓ PASS' if phase_pass else '✗ FAIL'}")

if mag_pass and phase_pass:
    print(f"\n✓ AC analysis validation: PASSED")
    print(f"  pycircuitsim results match NGSPICE (ground truth) within tolerance.")
else:
    print(f"\n✗ AC analysis validation: FAILED")
    print(f"  pycircuitsim results deviate from NGSPICE.")

print("=" * 80)
