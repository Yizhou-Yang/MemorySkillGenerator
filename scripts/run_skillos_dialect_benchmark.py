#!/usr/bin/env python3
"""
SkillOS Dialect Benchmark — reproduce SkillOS benchmark results using DeepSeek.

Reproduces 3 SkillOS benchmarks adapted for our LLM backend:
1. Math (formal-proof): K_{3,4} spanning trees (answer=432)
2. Physiology (system-dynamics): Mitral valve regurgitation
3. Skill Compression: Apply dialects to SkillForge's induced skills

SkillOS Reference Values (Claude Opus 4.6):
  | Benchmark        | Dialect        | Token Reduction | Quality |
  |------------------|----------------|-----------------|---------|
  | Math K_{3,4}     | formal-proof   | -51.3%          | 90/100  |
  | Physiology       | system-dynamics| -61.1%          | 100/100 |
  | Code Editing     | strict-patch   | -97.5%          | 2/2     |

We validate: same quality with DeepSeek-V3, measure token reduction.
"""
from __future__ import annotations

import json
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

from src.utils.config import load_env
from src.utils.llm import LLMClient
from src.models import Skill, TransformVariant
from src.skillos.dialect_framework import (
    DialectCompiler,
    DialectResult,
    SkillTaxonomy,
    compress_caveman,
    compress_exec_plan,
    compile_formal_proof,
    FORMAL_PROOF_SYSTEM_PROMPT,
    FORMAL_PROOF_RENDERER_PROMPT,
)


# ============================================================
# Benchmark 1: Math — K_{3,4} Spanning Trees
# ============================================================

CORRECT_ANSWER = 432

MATH_PROBLEM = """\
Calculate the exact number of spanning trees in a Complete Bipartite Graph K_{3,4} \
using the Matrix Tree Theorem.

You must:
1. Construct the full 7x7 Laplacian matrix L of K_{3,4} (rows/columns for vertices a1,a2,a3,b1,b2,b3,b4).
2. Find the cofactor L_11 by deleting the first row and first column (yielding a 6x6 matrix).
3. Calculate det(L_11) exactly — this equals the number of spanning trees.
4. State the final answer as a single integer."""

MATH_PLAIN_PROMPT = f"{MATH_PROBLEM}\n\nShow all work. Write your solution with full explanation."

MATH_DIALECT_PROMPT = f"""{MATH_PROBLEM}

Solve using ONLY formal-proof notation. No English prose — only structured derivation.

### Formal-Proof Grammar:
```
GIVEN:
  P1: [premise]
  P2: [premise]
DERIVE:
  D1: [statement] [BY rule]
  D2: [statement] [BY rule]
QED: [conclusion]
```
Rules: definition, matrix_tree_theorem, cofactor_expansion, arithmetic, substitution.
Use exact numeric values at every step. Output ONLY the proof block."""


def verify_math(text: str) -> dict:
    """Verify math answer (adapted from SkillOS benchmark_math.py)."""
    result = {"answer_correct": False, "score": 0, "extracted_answer": None}
    if not text:
        return result

    if re.search(r'\b432\b', text):
        result["answer_correct"] = True
        result["extracted_answer"] = 432
    else:
        for pat in [r'(?:answer|result|QED)[^\d]*?(\d+)', r'(\d+)\s*(?:spanning)', r'=\s*(\d+)\s*$']:
            m = re.search(pat, text, re.MULTILINE | re.IGNORECASE)
            if m:
                result["extracted_answer"] = int(m.group(1))
                if int(m.group(1)) == 432:
                    result["answer_correct"] = True
                break

    score = 0
    text_lower = text.lower()
    if result["answer_correct"]:
        score += 50
    if re.search(r'laplacian|degree\s*matrix|\bL\s*=', text_lower):
        score += 10
    if re.search(r'cofactor|minor|L_\{?11\}?', text_lower):
        score += 10
    if re.search(r'determinant|det\s*\(|\bdet\b', text_lower):
        score += 10
    has_27 = bool(re.search(r'\b27\b', text))
    has_16 = bool(re.search(r'\b16\b', text))
    if has_27 and has_16:
        score += 20
    result["score"] = score
    return result


# ============================================================
# Benchmark 2: Physiology — Mitral Valve Regurgitation
# ============================================================

PHYSIOLOGY_EXPECTED = {"velocity": 500, "volume": 60, "fraction": 60, "severity": "severe"}

PHYSIOLOGY_PROBLEM = """\
A patient has acute mitral valve regurgitation. Echocardiography shows:
- Regurgitant orifice area (ROA) = 0.4 cm^2
- Peak systolic transmitral pressure gradient = 100 mmHg
- Simplified Torricelli's equation: velocity v = 50 * sqrt(delta_P) (in cm/s)
- Systolic ejection time = 0.3 seconds
- Total left ventricular stroke volume (SV) = 100 mL

Calculate:
1. The regurgitant jet velocity (v)
2. The regurgitant flow rate (Q = ROA * v)
3. The Regurgitant Volume (RV = Q * systolic_time)
4. The Regurgitant Fraction (RF = RV / SV)
5. Classify severity: Mild (<30%), Moderate (30-50%), Severe (>50%)"""

PHYSIOLOGY_PLAIN_PROMPT = f"{PHYSIOLOGY_PROBLEM}\n\nShow all calculations and state the final classification."

PHYSIOLOGY_DIALECT_PROMPT = f"""{PHYSIOLOGY_PROBLEM}

Solve using ONLY system-dynamics dialect notation. No English prose — only structured computation.

### System-Dynamics Grammar:
```
[STOCK] name = value (unit)          — accumulated quantity
[FLOW] name: rate_expression         — rate of change
[EXT] name = value (unit)            — external input
[CALC] name = expression = result    — computation step
[EVAL] condition -> classification   — evaluation/threshold
```
Map the heart to a hydraulic circuit. Output ONLY the structured computation block."""


def verify_physiology(text: str) -> dict:
    """Verify physiology answer."""
    result = {"velocity_correct": False, "volume_correct": False,
              "fraction_correct": False, "severity_correct": False, "score": 0}
    if not text:
        return result

    if re.search(r'\b500\b', text):
        result["velocity_correct"] = True
    if re.search(r'\b60\b.*(?:mL|ml|volume)', text, re.IGNORECASE) or re.search(r'(?:volume|RV)\s*=?\s*60', text, re.IGNORECASE):
        result["volume_correct"] = True
    if re.search(r'60\s*%', text) or re.search(r'(?:fraction|RF)\s*=?\s*0?\.?60?', text, re.IGNORECASE):
        result["fraction_correct"] = True
    if re.search(r'severe', text, re.IGNORECASE):
        result["severity_correct"] = True

    score = 0
    if result["velocity_correct"]:
        score += 25
    if result["volume_correct"]:
        score += 25
    if result["fraction_correct"]:
        score += 25
    if result["severity_correct"]:
        score += 25
    result["score"] = score
    return result


# ============================================================
# Benchmark 3: Skill Compression
# ============================================================

def run_skill_compression_benchmark(compiler: DialectCompiler) -> dict:
    """Test dialect compression on SkillForge's skill format."""
    test_skills = [
        Skill(
            name="Multi-hop Question Answering",
            description="Answer questions that require combining information from multiple sources by decomposing into sub-questions",
            procedure=[
                "Decompose the complex question into 2-4 simpler sub-questions",
                "Search for evidence relevant to each sub-question independently",
                "Extract key facts from each evidence source",
                "Verify consistency across extracted facts",
                "Combine verified facts to synthesize the final answer",
            ],
            constraints=["Do not guess when evidence is insufficient", "Verify each reasoning hop before proceeding"],
            facts=["Multi-hop questions typically require 2-4 reasoning steps"],
        ),
        Skill(
            name="Temporal Event Tracking",
            description="Track and reason about events that occur at different times, maintaining chronological accuracy",
            procedure=[
                "Extract all temporal references (dates, times, relative markers)",
                "Build a timeline of events in chronological order",
                "Identify temporal relationships (before, after, during, overlapping)",
                "Resolve ambiguous temporal references using context",
                "Answer the question using the constructed timeline",
            ],
            constraints=["Be precise with dates and times", "Do not assume temporal order without evidence"],
            facts=["Temporal reasoning is one of the hardest QA categories"],
        ),
    ]

    results = {"skills": []}
    for skill in test_skills:
        skill_results = {"name": skill.name, "dialects": {}}
        for dialect in ["formal-proof", "caveman-prose", "exec-plan"]:
            dr = compiler.compile_skill(skill, dialect=dialect)
            skill_results["dialects"][dialect] = {
                "original_tokens": dr.original_tokens,
                "compressed_tokens": dr.compressed_tokens,
                "compression_ratio": dr.compression_ratio,
                "compressed_preview": dr.compressed_text[:200],
            }
        results["skills"].append(skill_results)

    return results


# ============================================================
# Benchmark 4: Taxonomy Token Savings
# ============================================================

def run_taxonomy_benchmark() -> dict:
    """Test hierarchical skill taxonomy token savings."""
    taxonomy = SkillTaxonomy()

    # Register skills in a realistic taxonomy
    domains = {
        "qa": {"multi-hop": 5, "single-hop": 3, "temporal": 4},
        "memory": {"compression": 3, "retrieval": 4, "consolidation": 2},
        "planning": {"decomposition": 3, "execution": 2},
    }

    total_skills = 0
    for domain, families in domains.items():
        for family, count in families.items():
            for i in range(count):
                skill = Skill(name=f"{domain}_{family}_{i}", description=f"Skill {i} in {domain}/{family}")
                taxonomy.register(skill, domain=domain, family=family)
                total_skills += 1

    savings = taxonomy.compute_token_savings(total_skills)
    domain_index = taxonomy.get_domain_index()
    family_index = taxonomy.get_family_index("qa")

    return {
        "total_skills": total_skills,
        "savings": savings,
        "domain_index": domain_index,
        "family_index_qa": family_index,
    }


# ============================================================
# Main
# ============================================================

def main():
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
    log_path = Path("experiments/skillos_dialect_benchmark.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), level="DEBUG")

    logger.info("=" * 70)
    logger.info("SkillOS Dialect Benchmark — Reproducing with DeepSeek-V3")
    logger.info("=" * 70)

    load_env()
    start_time = time.time()
    llm = LLMClient({"temperature": 0.0, "max_tokens": 4096, "timeout": 120})

    # API test
    resp = llm.chat([{"role": "user", "content": "Say OK"}], temperature=0.0, max_tokens=5)
    logger.info(f"API: '{resp.strip()}' ✅")

    compiler = DialectCompiler(llm_client=llm)
    all_results = {}

    # ================================================================
    # Benchmark 1: Math (formal-proof dialect)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK 1: Math — K_{3,4} Spanning Trees (formal-proof)")
    logger.info("=" * 60)
    logger.info(f"Correct answer: {CORRECT_ANSWER}")

    # Run 1a: Plain LLM
    logger.info("\n  [1a] Plain LLM (solve + explain)...")
    plain_resp = llm.chat(
        [{"role": "user", "content": MATH_PLAIN_PROMPT}],
        temperature=0.0, max_tokens=2048,
    )
    plain_tokens = len(plain_resp.split())
    plain_verify = verify_math(plain_resp)
    plain_ans = plain_verify['extracted_answer']
    plain_status = 'CORRECT' if plain_verify['answer_correct'] else f'WRONG ({plain_ans})'
    logger.info(f"    Tokens: {plain_tokens}, Score: {plain_verify['score']}/100, Answer: {plain_status}")

    # Run 1b: Dialect (formal-proof)
    logger.info("\n  [1b] Formal-proof dialect...")
    dialect_resp = llm.chat(
        [{"role": "user", "content": MATH_DIALECT_PROMPT}],
        temperature=0.0, max_tokens=1024,
    )
    dialect_tokens = len(dialect_resp.split())
    dialect_verify = verify_math(dialect_resp)
    dialect_ans = dialect_verify['extracted_answer']
    dialect_status = 'CORRECT' if dialect_verify['answer_correct'] else f'WRONG ({dialect_ans})'
    logger.info(f"    Tokens: {dialect_tokens}, Score: {dialect_verify['score']}/100, Answer: {dialect_status}")

    # Run 1c: Render dialect back to English
    logger.info("\n  [1c] Rendering dialect → English...")
    render_resp = compiler.render_from_dialect(dialect_resp, "formal-proof")
    render_tokens = len(render_resp.split())
    render_verify = verify_math(render_resp)
    logger.info(f"    Render tokens: {render_tokens}, Score: {render_verify['score']}/100")

    token_reduction_math = (plain_tokens - dialect_tokens) / plain_tokens * 100 if plain_tokens > 0 else 0

    all_results["math"] = {
        "plain": {"tokens": plain_tokens, "score": plain_verify["score"], "correct": plain_verify["answer_correct"]},
        "dialect": {"tokens": dialect_tokens, "score": dialect_verify["score"], "correct": dialect_verify["answer_correct"]},
        "render": {"tokens": render_tokens, "score": render_verify["score"]},
        "token_reduction_pct": token_reduction_math,
        "skillos_reference": {"token_reduction": 51.3, "quality": "90/100"},
    }

    logger.info(f"\n  Math Summary: Plain={plain_tokens}tok ({plain_verify['score']}/100), "
                f"Dialect={dialect_tokens}tok ({dialect_verify['score']}/100), "
                f"Reduction={token_reduction_math:.1f}%")
    logger.info(f"  SkillOS reference: -51.3%, 90/100")

    # ================================================================
    # Benchmark 2: Physiology (system-dynamics dialect)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK 2: Physiology — Mitral Valve (system-dynamics)")
    logger.info("=" * 60)

    # Run 2a: Plain LLM
    logger.info("\n  [2a] Plain LLM...")
    plain_resp2 = llm.chat(
        [{"role": "user", "content": PHYSIOLOGY_PLAIN_PROMPT}],
        temperature=0.0, max_tokens=2048,
    )
    plain_tokens2 = len(plain_resp2.split())
    plain_verify2 = verify_physiology(plain_resp2)
    logger.info(f"    Tokens: {plain_tokens2}, Score: {plain_verify2['score']}/100")

    # Run 2b: Dialect (system-dynamics)
    logger.info("\n  [2b] System-dynamics dialect...")
    dialect_resp2 = llm.chat(
        [{"role": "user", "content": PHYSIOLOGY_DIALECT_PROMPT}],
        temperature=0.0, max_tokens=1024,
    )
    dialect_tokens2 = len(dialect_resp2.split())
    dialect_verify2 = verify_physiology(dialect_resp2)
    logger.info(f"    Tokens: {dialect_tokens2}, Score: {dialect_verify2['score']}/100")

    # Run 2c: Render
    logger.info("\n  [2c] Rendering dialect → English...")
    render_resp2 = compiler.render_from_dialect(dialect_resp2, "formal-proof")
    render_tokens2 = len(render_resp2.split())
    render_verify2 = verify_physiology(render_resp2)
    logger.info(f"    Render tokens: {render_tokens2}, Score: {render_verify2['score']}/100")

    token_reduction_phys = (plain_tokens2 - dialect_tokens2) / plain_tokens2 * 100 if plain_tokens2 > 0 else 0

    all_results["physiology"] = {
        "plain": {"tokens": plain_tokens2, "score": plain_verify2["score"]},
        "dialect": {"tokens": dialect_tokens2, "score": dialect_verify2["score"]},
        "render": {"tokens": render_tokens2, "score": render_verify2["score"]},
        "token_reduction_pct": token_reduction_phys,
        "skillos_reference": {"token_reduction": 61.1, "quality": "100/100"},
    }

    logger.info(f"\n  Physiology Summary: Plain={plain_tokens2}tok ({plain_verify2['score']}/100), "
                f"Dialect={dialect_tokens2}tok ({dialect_verify2['score']}/100), "
                f"Reduction={token_reduction_phys:.1f}%")
    logger.info(f"  SkillOS reference: -61.1%, 100/100")

    # ================================================================
    # Benchmark 3: Skill Compression (local, no API)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK 3: Skill Compression (3 dialects × 2 skills)")
    logger.info("=" * 60)

    skill_results = run_skill_compression_benchmark(compiler)
    all_results["skill_compression"] = skill_results

    for sr in skill_results["skills"]:
        logger.info(f"\n  Skill: {sr['name']}")
        for dialect, dr in sr["dialects"].items():
            logger.info(f"    {dialect}: {dr['original_tokens']}→{dr['compressed_tokens']} tokens "
                        f"(-{dr['compression_ratio']*100:.1f}%)")

    # ================================================================
    # Benchmark 4: Taxonomy Token Savings (local, no API)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK 4: Hierarchical Taxonomy Token Savings")
    logger.info("=" * 60)

    taxonomy_results = run_taxonomy_benchmark()
    all_results["taxonomy"] = taxonomy_results

    savings = taxonomy_results["savings"]
    logger.info(f"  Total skills: {taxonomy_results['total_skills']}")
    logger.info(f"  Flat registry: {savings['flat_tokens']} tokens")
    logger.info(f"  Lazy loading: {savings['lazy_tokens']} tokens")
    logger.info(f"  Savings: {savings['savings_pct']:.1f}%")
    logger.info(f"  SkillOS reference: -61% routing-phase tokens")
    logger.info(f"\n  Domain index:\n{taxonomy_results['domain_index']}")

    # ================================================================
    # Benchmark 5: LLM-powered dialect compilation (burns tokens)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK 5: LLM-Powered Dialect Compilation")
    logger.info("=" * 60)

    test_skill = Skill(
        name="Multi-hop QA",
        description="Answer questions requiring information from multiple sources",
        procedure=[
            "Decompose the complex question into sub-questions",
            "Search for evidence for each sub-question",
            "Extract key facts from evidence",
            "Verify consistency across facts",
            "Combine facts to synthesize answer",
        ],
        constraints=["Do not guess", "Verify each hop"],
    )

    # Compile skill to formal-proof using LLM
    logger.info("\n  [5a] LLM-compiled formal-proof...")
    skill_text = compiler._skill_to_text(test_skill)
    llm_compiled = compiler.compile_with_llm(skill_text, "formal-proof")
    llm_tokens = len(llm_compiled.split())
    original_tokens = len(skill_text.split())
    logger.info(f"    Original: {original_tokens} tokens")
    logger.info(f"    Compiled: {llm_tokens} tokens")
    logger.info(f"    Reduction: {(original_tokens - llm_tokens) / original_tokens * 100:.1f}%")
    logger.info(f"    Preview: {llm_compiled[:200]}")

    # Render back
    logger.info("\n  [5b] Rendering back to English...")
    rendered = compiler.render_from_dialect(llm_compiled, "formal-proof")
    render_tokens_5 = len(rendered.split())
    logger.info(f"    Rendered: {render_tokens_5} tokens")
    logger.info(f"    Preview: {rendered[:200]}")

    all_results["llm_compilation"] = {
        "original_tokens": original_tokens,
        "compiled_tokens": llm_tokens,
        "rendered_tokens": render_tokens_5,
        "reduction_pct": (original_tokens - llm_tokens) / original_tokens * 100 if original_tokens > 0 else 0,
    }

    # ================================================================
    # Final Summary
    # ================================================================
    elapsed = time.time() - start_time
    stats = llm.stats

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS: SkillOS Dialect Benchmark")
    logger.info("=" * 70)

    logger.info(f"\n{'Benchmark':<25} {'Ours':<20} {'SkillOS Ref':<20} {'Match?':<10}")
    logger.info("-" * 75)

    # Math
    math_match = "✅" if all_results["math"]["dialect"]["correct"] else "❌"
    logger.info(f"{'Math (formal-proof)':<25} "
                f"{'-' + str(round(all_results['math']['token_reduction_pct'], 1)) + '%':<10} "
                f"{str(all_results['math']['dialect']['score']) + '/100':<10} "
                f"{'-51.3%':<10} {'90/100':<10} {math_match}")

    # Physiology
    phys_match = "✅" if all_results["physiology"]["dialect"]["score"] >= 75 else "❌"
    logger.info(f"{'Physiology (sys-dyn)':<25} "
                f"{'-' + str(round(all_results['physiology']['token_reduction_pct'], 1)) + '%':<10} "
                f"{str(all_results['physiology']['dialect']['score']) + '/100':<10} "
                f"{'-61.1%':<10} {'100/100':<10} {phys_match}")

    # Taxonomy
    tax_match = "✅" if savings["savings_pct"] > 50 else "❌"
    logger.info(f"{'Taxonomy (lazy load)':<25} "
                f"{'-' + str(round(savings['savings_pct'], 1)) + '%':<10} "
                f"{'N/A':<10} "
                f"{'-61%':<10} {'N/A':<10} {tax_match}")

    logger.info(f"\n💰 Token Usage: {stats['total_calls']} calls, {stats['total_tokens']:,} tokens, {elapsed:.1f}s")

    # Checks
    checks = [
        ("Math: formal-proof gets correct answer (432)", all_results["math"]["dialect"]["correct"]),
        ("Math: token reduction > 30%", all_results["math"]["token_reduction_pct"] > 30),
        ("Physiology: system-dynamics score ≥ 75", all_results["physiology"]["dialect"]["score"] >= 75),
        ("Physiology: token reduction > 30%", all_results["physiology"]["token_reduction_pct"] > 30),
        ("Skill compression: all 3 dialects produce output", all(
            len(sr["dialects"]) == 3 for sr in skill_results["skills"]
        )),
        ("Taxonomy: lazy loading saves > 50% tokens", savings["savings_pct"] > 50),
        ("LLM compilation: produces valid output", all_results["llm_compilation"]["compiled_tokens"] > 0),
    ]

    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    logger.info(f"\nChecks: {passed}/{total}")
    for name, ok in checks:
        logger.info(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")

    # Save results
    output_path = Path("experiments/skillos_dialect_benchmark_results.json")
    all_results["meta"] = {
        "elapsed_seconds": elapsed,
        "total_api_calls": stats["total_calls"],
        "total_tokens": stats["total_tokens"],
        "model": llm.model,
        "checks_passed": passed,
        "checks_total": total,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_path.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
