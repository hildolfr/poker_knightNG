"""Host-only parity checks for standalone ADR 0003 Philox/dealer headers."""
from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest

from poker_knight_ng.reference.dealer import canonical_case_hash, deal_cpu

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "src" / "poker_knight_ng" / "cuda-sources"

HARNESS = r'''
#include <cstdint>
#include <iostream>
#include "philox.cuh"
#include "dealer.cuh"
using namespace poker_knight_ng::cuda;
int main() {
  unsigned mode;
  while (std::cin >> mode) {
    if (mode == 0) {
      std::uint32_t c[4], k[2], out[4];
      for (auto& x : c) { std::cin >> x; }
      for (auto& x : k) { std::cin >> x; }
      philox4x32_10(c, k, out);
      for (auto x : out) { std::cout << x << ' '; }
      std::cout << '\n';
    } else if (mode == 1) {
      std::uint32_t k0, k1; std::uint64_t sim; unsigned known_count, opponents;
      std::uint8_t known[7]{}; std::cin >> k0 >> k1 >> sim >> known_count >> opponents;
      for (unsigned i=0;i<known_count;++i) { unsigned x; std::cin >> x; known[i]=static_cast<std::uint8_t>(x); }
      DealerResult r{}; const DealerStatus s=deal_adr0003(known, known_count, opponents, k0, k1, sim, &r);
      std::cout << static_cast<unsigned>(s) << ' ' << static_cast<unsigned>(r.draw_count);
      for (unsigned i=0;i<r.draw_count;++i) std::cout << ' ' << static_cast<unsigned>(r.dealt_card_ids[i]) << ':' << r.final_attempts[i] << ':' << r.rejection_counts[i];
      std::cout << '\n';
    } else {
      std::uint32_t candidate, range, attempt, rejects, index=0; std::cin >> candidate >> range >> attempt >> rejects;
      bool accepted=false;
      const DealerStatus s=dealer_candidate_step(candidate, range, &attempt, &rejects, &index, &accepted);
      std::cout << static_cast<unsigned>(s) << ' ' << accepted << ' ' << index << ' ' << attempt << ' ' << rejects << '\n';
    }
  }
}
'''

@pytest.fixture(scope="module")
def dealer_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("cuda-dealer-host")
    source, binary = d / "h.cpp", d / "h"
    source.write_text(HARNESS)
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", str(INCLUDE), str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary

def _run(binary: Path, lines: list[str]) -> list[str]:
    return subprocess.run([str(binary)], input="\n".join(lines)+"\n", check=True, text=True, capture_output=True).stdout.splitlines()

@pytest.mark.parametrize(("counter", "key", "expected"), [
 ((0,0,0,0),(0,0),(0x6627E8D5,0xE169C58D,0xBC57AC4C,0x9B00DBD8)),
 ((0xffffffff,)*4,(0xffffffff,)*2,(0x408F276D,0x41C83B0E,0xA20BC7C6,0x6D5451FD)),
 ((0x243F6A88,0x85A308D3,0x13198A2E,0x03707344),(0xA4093822,0x299F31D0),(0xD16CFE09,0x94FDCCEB,0x5001E420,0x24126EA1)),
])
def test_philox_random123_kats(dealer_harness, counter, key, expected):
    got = tuple(map(int, _run(dealer_harness, ["0 " + " ".join(map(str, counter + key))])[0].split()))
    assert got == expected

def test_adr_counter_attempt_and_uint64_high_lane(dealer_harness):
    key=(0x31CEE788,0xABEB8F0C)
    lines=["0 7 0 1 1 %d %d" % key, "0 7 1 0 0 %d %d" % key, "0 4294967295 4294967295 16 0 %d %d" % key]
    rows=[tuple(map(int,x.split())) for x in _run(dealer_harness,lines)]
    assert rows[0] == (0xEB06CCC9,0xAFA346CC,0x4A12A579,0x2095D67C)
    assert rows[1] != rows[0] and rows[2] != rows[1]

def test_known_deal_and_selection_rejection_boundaries(dealer_harness):
    line="1 %d %d 7 5 2 12 25 0 14 34" % (0x31CEE788,0xABEB8F0C)
    fields=_run(dealer_harness,[line])[0].split()
    assert fields[:2] == ["0","6"]
    assert [x.split(":")[0] for x in fields[2:]] == ["7","8","37","42","15","10"]
    assert all(x.endswith(":0:0") for x in fields[2:])
    # A: production candidate step rejects then accepts, preserving state on accept.
    assert _run(dealer_harness,["2 4294967295 3 0 0"])[0] == "0 0 0 1 1"
    assert _run(dealer_harness,["2 4 3 1 1"])[0] == "0 1 1 1 1"
    # B: terminal rejection is exhausted without wrapping/changing either counter.
    assert _run(dealer_harness,["2 4294967295 3 4294967295 4294967295"])[0] == "1 0 0 4294967295 4294967295"
    # C: normal acceptance at zero changes neither attempt nor rejection count.
    assert _run(dealer_harness,["2 4 3 0 0"])[0] == "0 1 1 0 0"

def test_cpu_differential_5000_and_scheduling_independence(dealer_harness):
    rng=random.Random(0xD34E1)
    pairs=[(b,o) for b in (0,3,4,5) for o in range(1,7)]
    pair_counts={pair: 0 for pair in pairs}
    cases=[]
    for n in range(5000):
        # Cycle the full Cartesian product rather than correlated moduli.
        b,o=pairs[n%len(pairs)]; pair_counts[(b,o)] += 1
        known=tuple(sorted(rng.sample(range(52),2+b)))
        hero,board=known[:2],known[2:]; seed=rng.getrandbits(64); sim=(rng.getrandbits(64) if n%3 else (1<<32)+n)
        key_digest=__import__('hashlib').sha256(bytes([26])+b'poker-knight-ng/rng-key/v1'+seed.to_bytes(8,'little')+canonical_case_hash(hero,board,o)).digest()
        k0=int.from_bytes(key_digest[:4],'little'); k1=int.from_bytes(key_digest[4:8],'little')
        cases.append((hero,board,o,sim,k0,k1,deal_cpu(seed=seed,canonical_case_hash=canonical_case_hash(hero,board,o),hero_card_ids=hero,board_card_ids=board,opponent_count=o,simulation_id=sim)))
    assert set(pair_counts) == {(b,o) for b in (0,3,4,5) for o in range(1,7)}
    assert min(pair_counts.values()) == 208
    assert max(pair_counts.values()) == 209
    # sorted validated harness inputs; Python also proves semantic ordering independently.
    lines=["1 %d %d %d %d %d %s"%(k0,k1,sim,len(h)+len(b),o," ".join(map(str,h+b))) for h,b,o,sim,k0,k1,_ in cases]
    rows=_run(dealer_harness,lines)
    for row,(*_,expected) in zip(rows,cases):
        f=row.split(); assert f[:2]==["0",str(len(expected.dealt_card_ids))]
        assert [int(x.split(':')[0]) for x in f[2:]]==list(expected.dealt_card_ids)
        assert [(int(x.split(':')[1]),int(x.split(':')[2])) for x in f[2:]]==[(s.final_attempt,s.rejection_count) for s in expected.trace.slots]
    indices=[7,2,4999,17,700]
    rerun=_run(dealer_harness,[lines[i] for i in reversed(indices)])
    assert rerun == [rows[i] for i in reversed(indices)]

def test_canonical_reordering_semantics_are_equal():
    assert canonical_case_hash((25,12),(34,0,14),2)==canonical_case_hash((12,25),(0,14,34),2)
