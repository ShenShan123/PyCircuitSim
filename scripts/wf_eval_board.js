export const meta = {
  name: 'eval-board-per-seed',
  description: 'Evaluate per-seed DirectNet checkpoints on the complex board (switchcap, opamp, ring_osc, sram_snm) + inverter VTC, on CPU, via env override',
  phases: [{ title: 'Eval', detail: 'one agent per (seed, gate), parallel on CPU' }],
}

// args = { tech: "TSMC5", techkey: "tsmc5", stem: "dn_lg", seeds: [42,17,7,31] }
const A = args || {}
const TECH = A.tech || 'TSMC5'
const TECHKEY = A.techkey || 'tsmc5'
const STEM = A.stem || 'dn_lg'       // checkpoint = <techkey>_<stem>_s<seed>_<dev>
const SEEDS = A.seeds || [42, 17, 7, 31]

const GATES = [
  { key: 'switchcap', script: 'tests/verify_complex_switchcap.py', metric: 'charge err (% of VDD)' },
  { key: 'opamp',     script: 'tests/verify_complex_opamp.py',     metric: 'gain error (%)' },
  { key: 'ring_osc',  script: 'tests/verify_complex_ring_osc.py',  metric: 'period error (%)' },
  { key: 'sram_snm',  script: 'tests/verify_complex_sram_snm.py',  metric: 'butterfly polarity (pos/neg)' },
  { key: 'inverter',  script: 'tests/verify_nn_dc_tran.py',        metric: 'VTC NRMSE (%)', extra: '--inverter-only' },
]

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    seed: { type: 'number' },
    gate: { type: 'string' },
    metric_pct: { type: 'number', description: 'the primary error metric for this gate (charge%/gain%/period%/NRMSE%); use 0 for sram polarity' },
    status: { type: 'string', description: 'PASS / FAIL / ERROR' },
    summary_line: { type: 'string', description: 'the verbatim summary/result line(s) from the gate output' },
  },
  required: ['seed', 'gate', 'status', 'summary_line'],
}

phase('Eval')

const jobs = []
for (const seed of SEEDS) {
  for (const g of GATES) {
    jobs.push({ seed, g })
  }
}

const results = await parallel(jobs.map(({ seed, g }) => () => {
  const cmd =
    `cd /data2/shenshan/PyCircuitSim && CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ` +
    `NGSPICE_BIN=/data2/shenshan/PyCircuitSim/tools/ngspice-45.2/bin/ngspice ` +
    `PYCIRCUITSIM_NN_CHECKPOINT_DN_NMOS=${TECHKEY}_${STEM}_s${seed}_nmos ` +
    `PYCIRCUITSIM_NN_CHECKPOINT_DN_PMOS=${TECHKEY}_${STEM}_s${seed}_pmos ` +
    `conda run -n pycircuitsim python ${g.script} --tech ${TECH}${g.extra ? ' ' + g.extra : ''}`
  return agent(
    `Run exactly this command (CPU-only, ~1-3 min) and report the parsed gate result for ${TECH} seed ${seed} gate ${g.key}.\n\n` +
    `${cmd}\n\n` +
    `Parse the gate's SUMMARY / result line. The relevant metric for this gate is: ${g.metric}.\n` +
    `For switchcap parse "charge err=X% of VDD"; opamp parse "gain error = X%" or the SUMMARY GainErr% column; ` +
    `ring_osc parse "period error = X%"; sram_snm parse whether the butterfly is positive (PASS) — set metric_pct=0; ` +
    `inverter parse the directnet_v4 VTC "NRMSE=X%".\n` +
    `Return seed=${seed}, gate="${g.key}", metric_pct (numeric), status (PASS/FAIL/ERROR), and summary_line ` +
    `(the verbatim result line(s)). If the command errors or a checkpoint is missing, set status="ERROR" and put the error in summary_line.`,
    { label: `${TECHKEY} s${seed} ${g.key}`, phase: 'Eval', schema: SCHEMA }
  )
}))

const ok = results.filter(Boolean)
log(`Board eval complete: ${ok.length}/${jobs.length} runs returned for ${TECH} (${STEM})`)
return { tech: TECH, stem: STEM, results: ok }
