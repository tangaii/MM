import fcntl
import hashlib
import os
from pathlib import Path
import tempfile

ROOT = Path("/data/adodas/MM-GTUNets")
HANDOFF = ROOT / "handoff.md"
AUDIT = ROOT / "experiments/D4_REPRO_AUDIT_20261020"
SCPDA = Path("/tmp/SCPDA_20261020/handoff_pending.md")
D4 = AUDIT / "handoff_d4.md"
LOCK = AUDIT / "handoff.lock"
EXPECTED_SHA256 = "4c943abb1ea7c9f849fca0cfa84180e690d0ed89cec8b1519c4220fd0680764b"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    with LOCK.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        original = HANDOFF.read_bytes()
        if sha(original) != EXPECTED_SHA256:
            raise RuntimeError("handoff.md changed since the audited starting point; refusing to append")
        if b"<!-- SCPDA_20261020_START -->" in original or b"<!-- D4_REPRO_AUDIT_20261020_START -->" in original:
            raise RuntimeError("one of the target handoff sections already exists; refusing duplicate append")

        scpda = SCPDA.read_bytes()
        d4 = D4.read_bytes()
        required = (
            b"<!-- SCPDA_20261020_START -->" in scpda,
            b"<!-- SCPDA_20261020_END -->" in scpda,
            b"<!-- D4_REPRO_AUDIT_20261020_START -->" in d4,
            b"<!-- D4_REPRO_AUDIT_20261020_END -->" in d4,
        )
        if not all(required):
            raise RuntimeError("pending handoff section markers are incomplete")

        prefix = original + (b"" if original.endswith(b"\n") else b"\n")
        appended = prefix + b"\n" + scpda.rstrip() + b"\n\n" + d4.rstrip() + b"\n"
        fd, temp_name = tempfile.mkstemp(prefix=".handoff.md.tmp.", dir=HANDOFF.parent)
        temp = Path(temp_name)
        try:
            os.fchmod(fd, HANDOFF.stat().st_mode & 0o777)
            with os.fdopen(fd, "wb") as out:
                out.write(appended)
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, HANDOFF)
            dir_fd = os.open(HANDOFF.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if temp.exists():
                temp.unlink()

        verify = HANDOFF.read_bytes()
        if not verify.startswith(original):
            raise RuntimeError("handoff prefix verification failed after atomic replacement")
        if verify.count(b"<!-- SCPDA_20261020_START -->") != 1 or verify.count(b"<!-- D4_REPRO_AUDIT_20261020_START -->") != 1:
            raise RuntimeError("handoff marker verification failed after atomic replacement")
        if verify.index(b"<!-- SCPDA_20261020_START -->") > verify.index(b"<!-- D4_REPRO_AUDIT_20261020_START -->"):
            raise RuntimeError("handoff chronology verification failed")
        print({"status": "complete", "old_sha256": sha(original), "new_sha256": sha(verify),
               "old_bytes": len(original), "new_bytes": len(verify),
               "scpda_bytes": len(scpda), "d4_bytes": len(d4), "prefix_preserved": True})


if __name__ == "__main__":
    main()
