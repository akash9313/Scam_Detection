"""
SpamAdapt Phase 6: Unified Acceptance Test Runner.
Executes all 6 test suites sequentially and summarizes outcomes.
"""

import subprocess
import sys
import time

TEST_SUITES = [
    "test_phase6_app_smoke.py",
    "test_phase5_robustness.py",
    "test_phase4_benchmark.py",
    "test_phase3_augmented_dataset.py",
    "test_telecom_fixture.py",
    "test_app.py",
]

def main():
    print("=" * 80)
    print("SPAMADAPT PHASE 6: UNIFIED FINAL ACCEPTANCE TEST SUITE")
    print("=" * 80)
    
    results = {}
    t_start = time.time()
    
    for test_file in TEST_SUITES:
        print(f"\n[RUNNING SUITE] {test_file} ...")
        t0 = time.time()
        proc = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
        elapsed = time.time() - t0
        
        # Print trimmed output
        lines = [l for l in proc.stdout.splitlines() if l.strip()]
        for l in lines[-8:]:
            print(f"  | {l}")
            
        is_pass = (proc.returncode == 0)
        results[test_file] = {
            "passed": is_pass,
            "returncode": proc.returncode,
            "elapsed_s": round(elapsed, 2),
            "stderr": proc.stderr if not is_pass else "",
        }
        status_str = "PASSED (100%)" if is_pass else "FAILED"
        print(f"  ---> RESULT: {status_str} in {elapsed:.2f}s")
        
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print("FINAL ACCEPTANCE VERIFICATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_file, res in results.items():
        status = "PASSED (100%)" if res["passed"] else "FAILED"
        print(f"  {test_file:34s} : {status:15s} (Duration: {res['elapsed_s']:5.2f}s)")
        if not res["passed"]:
            all_passed = False
            print(f"    ERROR DETAILS:\n{res['stderr']}")
            
    print("-" * 80)
    print(f"Total Suites Run: {len(TEST_SUITES)} | Total Elapsed Time: {total_elapsed:.2f}s")
    if all_passed:
        print("ALL 6 TEST SUITES PASSED WITH ZERO FAILURES (100% SUCCESS RATE)")
    else:
        print("ACCEPTANCE FAILED: One or more suites encountered errors.")
    print("=" * 80)
    
    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
