"""Host-only TDD vectors for deterministic checked TrialResult reduction."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from poker_knight_ng.reference.monte_carlo import _run_trial, run_cpu_monte_carlo

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / "src" / "poker_knight_ng" / "cuda-sources"
SEED_BANK = ROOT / "validation" / "holdem" / "v1" / "rng_seed_bank.json"

HARNESS = r'''
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include "reduce.cuh"
using namespace poker_knight_ng::cuda;
static void out(const AggregateResult& a) {
 std::cout << unsigned(a.status) << ' ' << a.completed_trials << ' ' << a.unique_wins;
 for (auto x:a.tie_by_other_winners) std::cout << ' ' << x;
 std::cout << ' ' << a.losses << ' ' << a.equity_share_units;
 for (auto x:a.hero_category_counts) std::cout << ' ' << x;
 std::cout << ' ' << a.rejection_count.lo << ' ' << a.rejection_count.hi << ' ' << a.failure_simulation_id << ' ' << a.failure_draw_slot << '\n';
}
int main() {
 static_assert(std::is_standard_layout<AggregateResult>::value, "stable layout");
 static_assert(std::is_trivially_copyable<AggregateResult>::value, "partial safe");
 static_assert(sizeof(decltype(AggregateResult::completed_trials)) == 8, "u64");
 static_assert(sizeof(decltype(AggregateResult::rejection_count.lo)) == 8, "u128 lanes");
 static_assert(sizeof(decltype(AggregateResult::failure_draw_slot)) == 4, "slot");
 unsigned mode, n; while(std::cin >> mode >> n) {
   AggregateResult a{};
   for(unsigned i=0;i<n;++i) { unsigned st,oc,tie,cat; std::uint64_t sim,rej; std::uint32_t slot; std::uint64_t units;
     std::cin >> sim >> st >> oc >> tie >> cat >> units >> rej >> slot;
     TrialResult r{}; r.status=static_cast<TrialStatus>(st); r.outcome=static_cast<TrialOutcome>(oc); r.tie_other_winners=tie; r.hero_category=cat; r.equity_share_units=units; r.rejection_count=rej; r.failure_draw_slot=slot;
     if(mode==0) reduce_trial(r,sim,&a); else reduce_range(&r,sim,1,&a);
   } out(a);
 }
}
'''

@pytest.fixture(scope="module")
def reduction_harness(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("cuda-reduction-host")
    source, binary = d / "h.cpp", d / "h"
    source.write_text(HARNESS)
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I", str(INCLUDE), str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    return binary


def _run(binary: Path, rows: list[tuple[int, int, int, int, int, int, int, int]], mode: int = 0) -> tuple[int, ...]:
    text = f"{mode} {len(rows)}\n" + "\n".join(" ".join(map(str, row)) for row in rows) + "\n"
    return tuple(map(int, subprocess.run([str(binary)], input=text, text=True, capture_output=True, check=True).stdout.split()))


def _trial(sim: int, outcome: int, tie: int = 0, category: int = 0, rejection: int = 0) -> tuple[int, int, int, int, int, int, int, int]:
    return sim, 0, outcome, tie, category, 420 if outcome == 0 else (420 // (tie + 1) if outcome == 1 else 0), rejection, 0xFFFFFFFF


def test_direct_known_contributions_all_outcomes_categories_and_u128(reduction_harness: Path) -> None:
    rows = [_trial(0, 0, category=0, rejection=1), _trial(1, 1, 1, 1, 2), _trial(2, 1, 6, 2, 3), _trial(3, 2, category=3, rejection=4)]
    got = _run(reduction_harness, rows)
    # status, completed, wins, ties[6], loss, equity, categories[9], rejection lo/hi, fail id/slot
    assert got == (0, 4, 1, 1, 0, 0, 0, 0, 1, 1, 690, 1, 1, 1, 1, 0, 0, 0, 0, 0, 10, 0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFF)


def _rows_for_case(seed: int, hero: tuple[int, int], board: tuple[int, ...], opponents: int, count: int) -> list[tuple[int, int, int, int, int, int, int, int]]:
    rows=[]
    for sim in range(count):
        r=_run_trial(seed=seed, hero_card_ids=hero, board_card_ids=board, opponent_count=opponents, simulation_id=sim)
        tie=next((i+1 for i,v in enumerate(r.tie_by_other_winners) if v),0)
        outcome=0 if r.unique_wins else (1 if tie else 2)
        cat=next(i for i,v in enumerate(r.hero_category_counts) if v)
        rows.append((sim,0,outcome,tie,cat,r.equity_share_units,r.rejection_count,0xFFFFFFFF))
    return rows


def _published(result: tuple[int, ...]) -> tuple[int, ...]:
    return result[:20]


def test_three_seed_bank_streams_match_cpu_every_counter(reduction_harness: Path) -> None:
    for vector in json.loads(SEED_BANK.read_text())["exact_vectors"]:
        rows=_rows_for_case(int(vector["seed"],16),tuple(vector["hero_card_ids"]),tuple(vector["board_card_ids"]),vector["opponent_count"],int(vector["requested_trials"]))
        got=_run(reduction_harness,rows)
        expected=run_cpu_monte_carlo(seed=int(vector["seed"],16),hero_card_ids=tuple(vector["hero_card_ids"]),board_card_ids=tuple(vector["board_card_ids"]),opponent_count=vector["opponent_count"],requested_trials=len(rows))
        assert got[0] == 0 and got[1] == expected.completed_trials and got[2] == expected.unique_wins
        assert got[3:9] == expected.tie_by_other_winners and got[9:11] == (expected.losses, expected.equity_share_units)
        assert got[11:20] == expected.hero_category_counts
        assert got[20] + (got[21] << 64) == expected.rejection_count


def test_partitions_orders_failure_and_invalid_rejection(reduction_harness: Path) -> None:
    rows=_rows_for_case(0x0123456789ABCDEF,(12,25),(0,14,34),2,5000)
    whole=_run(reduction_harness,rows)
    assert _run(reduction_harness,list(reversed(rows))) == whole
    assert _run(reduction_harness,rows,mode=1) == whole
    failures=[(7,1,2,0,0,0,0,16),(2,1,2,0,0,0,0,9),(2,1,2,0,0,0,0,3)]
    f=_run(reduction_harness,failures)
    assert f[0] == 1 and f[-2:] == (2,3)
    assert _run(reduction_harness,[(3,1,2,0,0,0,0,17)])[0] == 2
    assert _run(reduction_harness,[(0xFFFFFFFFFFFFFFFF,1,2,0,0,0,0,0)])[0] == 2
    bad=[(0,99,0,0,0,0,0,0xFFFFFFFF),(1,0,99,0,0,0,0,0xFFFFFFFF),(2,0,1,0,0,210,0,0xFFFFFFFF),(3,0,0,0,9,420,0,0)]
    assert all(_run(reduction_harness,[x])[0] == 2 for x in bad)


def test_real_chunk_reverse_tree_and_identity_merges_match_whole_range(reduction_harness: Path, tmp_path: Path) -> None:
    source, binary = tmp_path / "partitions.cpp", tmp_path / "partitions"
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include "reduce.cuh"
using namespace poker_knight_ng::cuda;
static bool same(const AggregateResult& a, const AggregateResult& b) {
  if (a.status != b.status || a.completed_trials != b.completed_trials || a.unique_wins != b.unique_wins ||
      a.losses != b.losses || a.equity_share_units != b.equity_share_units ||
      a.rejection_count.lo != b.rejection_count.lo || a.rejection_count.hi != b.rejection_count.hi ||
      a.failure_simulation_id != b.failure_simulation_id || a.failure_draw_slot != b.failure_draw_slot) return false;
  for (unsigned i=0;i<6;++i) if (a.tie_by_other_winners[i] != b.tie_by_other_winners[i]) return false;
  for (unsigned i=0;i<9;++i) if (a.hero_category_counts[i] != b.hero_category_counts[i]) return false;
  return true;
}
int main() {
  std::vector<TrialResult> rows(5000);
  for (unsigned i=0;i<rows.size();++i) { TrialResult& r=rows[i]; r.status=TrialStatus::OK; r.hero_category=i%9; r.rejection_count=(i*17U)%23U; r.failure_draw_slot=NO_FAILURE_DRAW_SLOT;
    if (i%8==0) { r.outcome=TrialOutcome::UNIQUE_WIN; r.equity_share_units=420; }
    else if (i%8==1) { r.outcome=TrialOutcome::LOSS; r.equity_share_units=0; }
    else { r.outcome=TrialOutcome::TIE; r.tie_other_winners=(i%6)+1; r.equity_share_units=420/(r.tie_other_winners+1); }
  }
  AggregateResult whole{}; reduce_range(rows.data(),1000,rows.size(),&whole);
  std::vector<AggregateResult> chunks; const unsigned sizes[]={1,7,31,257,3,509,1024,11};
  for (unsigned first=0,k=0;first<rows.size();++k) { unsigned n=sizes[k%8]; if(n>rows.size()-first)n=rows.size()-first; AggregateResult p{}; reduce_range(rows.data()+first,1000+first,n,&p); chunks.push_back(p); first+=n; }
  AggregateResult forward{}; for(const auto& p:chunks) merge_aggregate(p,&forward); assert(same(whole,forward));
  AggregateResult reverse{}; for(auto it=chunks.rbegin();it!=chunks.rend();++it) merge_aggregate(*it,&reverse); assert(same(whole,reverse));
  std::vector<AggregateResult> tree=chunks; while(tree.size()>1) { std::vector<AggregateResult> next; for(unsigned i=0;i+1<tree.size();i+=2) { AggregateResult p=tree[i]; merge_aggregate(tree[i+1],&p); next.push_back(p); } if(tree.size()%2) next.push_back(tree.back()); tree.swap(next); } assert(same(whole,tree[0]));
  AggregateResult left{}, right{}; merge_aggregate(whole,&left); merge_aggregate(right,&left); merge_aggregate(whole,&right); AggregateResult identity{}; merge_aggregate(identity,&right); assert(same(whole,left) && same(whole,right));
}
''')
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I", str(INCLUDE), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_checked_overflow_and_algebra_in_host_header(reduction_harness: Path, tmp_path: Path) -> None:
    source = tmp_path / "algebra.cpp"
    binary = tmp_path / "algebra"
    source.write_text(r'''
#include <cassert>
#include <cstdint>
#include <limits>
#include "reduce.cuh"
using namespace poker_knight_ng::cuda;
int main() {
  const auto M=std::numeric_limits<std::uint64_t>::max();
  TrialResult win{}; win.status=TrialStatus::OK; win.outcome=TrialOutcome::UNIQUE_WIN; win.equity_share_units=420; win.failure_draw_slot=NO_FAILURE_DRAW_SLOT;
  AggregateResult a{}, b{}; reduce_trial(win,0,&a); b=a; a.status=AggregateStatus::COUNTER_OVERFLOW; a.completed_trials=M-5; a.unique_wins=M-5; a.equity_share_units=M-5; b.status=AggregateStatus::COUNTER_OVERFLOW; b.completed_trials=3; b.unique_wins=3; b.equity_share_units=1260;
  AggregateResult c=b; c.completed_trials=4; c.unique_wins=4; c.equity_share_units=1680;
  AggregateResult left=a; merge_aggregate(b,&left); merge_aggregate(c,&left);
  AggregateResult bc=b; merge_aggregate(c,&bc); AggregateResult right=a; merge_aggregate(bc,&right);
  assert(left.status==AggregateStatus::COUNTER_OVERFLOW && left.completed_trials==M && left.unique_wins==M);
  assert(right.status==AggregateStatus::COUNTER_OVERFLOW && right.completed_trials==M && right.unique_wins==M);
  AggregateResult x{}, y{}; x.rejection_count={M,7}; y.rejection_count={1,0}; merge_aggregate(y,&x);
  assert(x.rejection_count.lo==0 && x.rejection_count.hi==8 && x.status==AggregateStatus::OK);
  y.rejection_count={0,M}; merge_aggregate(y,&x);
  assert(x.status==AggregateStatus::COUNTER_OVERFLOW && x.rejection_count.lo==M && x.rejection_count.hi==M);
  AggregateResult destination{}; reduce_trial(win,9,&destination); const AggregateResult clean=destination;
  AggregateResult malformed{}; malformed.status=static_cast<AggregateStatus>(99); malformed.completed_trials=7;
  merge_aggregate(malformed,&destination); assert(destination.status==AggregateStatus::INVALID_INPUT && destination.completed_trials==clean.completed_trials);
  destination=clean; malformed={}; malformed.status=AggregateStatus::OK; malformed.failure_simulation_id=2; malformed.failure_draw_slot=1;
  merge_aggregate(malformed,&destination); assert(destination.status==AggregateStatus::INVALID_INPUT && destination.failure_simulation_id==clean.failure_simulation_id);
  destination=clean; malformed={}; malformed.status=AggregateStatus::RNG_REJECTION_EXHAUSTED;
  merge_aggregate(malformed,&destination); assert(destination.status==AggregateStatus::INVALID_INPUT && destination.completed_trials==1);
  destination=clean; malformed={}; malformed.status=AggregateStatus::OK; malformed.completed_trials=1;
  merge_aggregate(malformed,&destination); assert(destination.status==AggregateStatus::INVALID_INPUT && destination.completed_trials==1);
  destination=clean; malformed={}; malformed.status=AggregateStatus::OK; malformed.completed_trials=1; malformed.unique_wins=1; malformed.hero_category_counts[0]=1; malformed.equity_share_units=419;
  merge_aggregate(malformed,&destination); assert(destination.status==AggregateStatus::INVALID_INPUT && destination.completed_trials==1);
  destination=clean; malformed={}; malformed.status=AggregateStatus::OK; malformed.completed_trials=1; malformed.unique_wins=1; malformed.equity_share_units=420;
  merge_aggregate(malformed,&destination); assert(destination.status==AggregateStatus::INVALID_INPUT && destination.completed_trials==1);
  destination=clean; malformed={}; malformed.status=AggregateStatus::RNG_REJECTION_EXHAUSTED; malformed.failure_simulation_id=1; malformed.failure_draw_slot=17;
  merge_aggregate(malformed,&destination); assert(destination.status==AggregateStatus::INVALID_INPUT);
  destination=clean; malformed={}; malformed.status=AggregateStatus::INVALID_INPUT; malformed.failure_simulation_id=M; malformed.failure_draw_slot=0;
  merge_aggregate(malformed,&destination); assert(destination.status==AggregateStatus::INVALID_INPUT && destination.failure_simulation_id==clean.failure_simulation_id);
  AggregateResult retained{}; retained.status=AggregateStatus::INVALID_INPUT; retained.failure_simulation_id=3; retained.failure_draw_slot=4;
  merge_aggregate(retained,&destination); assert(destination.failure_simulation_id==3 && destination.failure_draw_slot==4);
  AggregateResult exhausted{}; exhausted.status=AggregateStatus::RNG_REJECTION_EXHAUSTED; exhausted.failure_simulation_id=7; exhausted.failure_draw_slot=4;
  AggregateResult unknown{}; unknown.status=static_cast<AggregateStatus>(99); unknown.completed_trials=99;
  AggregateResult malformed_after=exhausted; merge_aggregate(unknown,&malformed_after);
  AggregateResult malformed_before{}; merge_aggregate(unknown,&malformed_before); merge_aggregate(exhausted,&malformed_before);
  assert(malformed_after.status==AggregateStatus::INVALID_INPUT && malformed_before.status==AggregateStatus::INVALID_INPUT);
  assert(malformed_after.failure_simulation_id==7 && malformed_after.failure_draw_slot==4);
  assert(malformed_before.failure_simulation_id==7 && malformed_before.failure_draw_slot==4);
  AggregateResult overflow{}; overflow.status=AggregateStatus::COUNTER_OVERFLOW; overflow.completed_trials=M; overflow.unique_wins=M; overflow.failure_simulation_id=2; overflow.failure_draw_slot=1;
  merge_aggregate(overflow,&destination); assert(destination.status==AggregateStatus::COUNTER_OVERFLOW && destination.completed_trials==M && destination.failure_simulation_id==2);
  TrialResult row{}; row.status=TrialStatus::OK; row.outcome=TrialOutcome::UNIQUE_WIN; row.equity_share_units=420; row.failure_draw_slot=NO_FAILURE_DRAW_SLOT;
  AggregateResult range{}; reduce_range(&row,M-1,2,&range); assert(range.status==AggregateStatus::INVALID_INPUT && range.completed_trials==0);
}
''')
    subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I", str(INCLUDE), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
