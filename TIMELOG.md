# Time log — reconstructed applicant-active time

Rules (from the admissions doc): **clocked** = anything actively advancing the project
(coding, project-directed reading, analysis, thinking, doc writing). **Un-clocked** =
general prep before the project started, env/tooling setup, model downloads, waiting on
jobs. Also track in Toggl and screenshot for the application doc.

Budget: E1 3h · E2 5h · E3 3h · baselines 3h · analysis 3h · buffer/E4 3h · write-up +2h

The previous automated reconstruction was removed because it incorrectly treated long
agent sessions, parallel agents, idle gaps, and predecessor work as applicant-active
time. Those are not valid under the application's time-counting rules.

The current workspace began on 2026-09-08. The earlier `adl-trace-dissociation`
workspace pursued the same research question, organisms, and E1–E3 plan, so its clocked
research time is disclosed and included rather than treated as a fresh-project reset.
Downloads, environment setup, unattended training, agent-only execution, and waiting are
excluded.

The reconstruction below uses foreground user interactions, short review-session spans,
and artifact-producing windows. Long gaps between messages were not counted. Status
checks were counted only as brief supervision, and overlapping/parallel agent work was
counted once or excluded. Because no stopwatch was running, the daily figures are
deliberately rounded.

| # | Date | Counted activity | Approx. active hours |
|---|------|------------------|---------------------:|
| 1 | 2026-09-08 | Chose the fresh experiment, discussed the plan, started the scaffold/smoke test, and reviewed the resulting summary. Long unattended execution was excluded. | 1.0 |
| 2 | 2026-09-09 | Assessed Project A, decided how to improve it for the application, and initiated the next experiment. Work on `refusal-repro` was excluded. | 1.0 |
| 3 | 2026-09-10 | Directed and intermittently supervised the factorial work, reviewed results and limitations, and worked through the research question/application fit. Training waits and parallel-agent runtime were excluded. | 7.0 |
| 4 | 2026-09-11, through ~10:35 IST | Re-read explanations, learned the experiment well enough to explain it, personally inspected transcripts, and verified headline claims for the application. | 3.0 |
| 5 | 2026-09-11, ~10:35–13:27 IST | Worked through the intervention interpretation, made and reviewed the scaling-extension decision, supervised checkpoints, and inspected the 4B gate, parity, and capability result. Unattended model execution was excluded. | 1.5 |
| 6 | 2026-09-06, earlier same-project workspace | Formal clock opened at 09:58 for building the ADL extraction/readout harness; artifacts and work-log entries continue through roughly 11:03. Preregistration and environment setup were contemporaneously marked unclocked. | 1.1 |
| 7 | project-directed reading | Reading Minder et al. (ADL paper) and its appendices/implementation for this project. Counted per the admissions rule that papers read because they are relevant to the project are clocked. | 2.0 |
| | **Reconstructed total so far** | | **~16.6 h** |

**Recommended application figure at this point: approximately 17 hours.** The underlying
reconstruction is about 16.6 hours, with a reasonable uncertainty band of roughly
15–19 hours. State the method: no stopwatch, reconstructed from foreground messages,
work logs, file timestamps, and the earlier project's open clock. This remains consistent
with the applicant's recollection that the total is below 18 hours. Add subsequent
applicant-active writing and checking separately; do not count unattended compute or
agent-only packaging.

<!-- When starting a block: add a row with Start filled, End = (open). -->
<!-- When stopping: fill End + h, add elapsed to the running total. -->
<!-- Un-clocked (do NOT count): general reading before start, repo cloning, venv setup,
     HF model downloads, feasibility smoke tests, waiting on jobs. -->
