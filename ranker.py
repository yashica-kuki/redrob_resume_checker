#!/usr/bin/env python3
"""
Redrob Hackathon — Candidate Ranker (VS Code / Local Production Hardened)
Produces top-100 ranked CSV from candidates.jsonl per submission_spec.md.

Constraints satisfied: CPU-only, no network, single streaming pass,
well under 5 min / 16GB / 5GB disk for 100K candidates.
"""

import argparse
import csv
import gzip
import json
import os
import re
from datetime import datetime
from pathlib import Path

DEFAULT_INPUT_FILE = "input/candidates.jsonl"
DEFAULT_OUTPUT_FILE = "output/redrob_submission.csv"

TODAY = datetime(2026, 6, 17)

BANNED_SERVICES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "mindtree", "hcl"
}

CORE_RETRIEVAL_KW = [
    "retrieval", "search", "ranking", "rerank", "embedding",
    "vector", "pinecone", "milvus", "weaviate", "faiss",
    "opensearch", "elasticsearch", "qdrant", "bm25",
    "hybrid search", "sentence-transformers", "bge", "e5",
    "ndcg", "mrr", "recommendation"
]

LLM_KW = [
    "llm", "fine-tun", "lora", "qlora", "peft", "rag", "prompt",
    "transformer", "gpt", "langchain"
]

NON_FIT_DOMAIN_KW = [
    "robotics", "computer vision", "yolo", "cnn",
    "image classification", "speech recognition", "ocr",
    "object detection", "gan"
]

TIER1_TITLE_KW = [
    "ai engineer", "ml engineer", "machine learning",
    "nlp engineer", "search engineer", "ranking engineer",
    "recommendation", "applied scientist", "research engineer",
    "data scientist", "ai researcher"
]

LEADERSHIP_ONLY_TITLE_KW = [
    "architect", "tech lead", "engineering manager",
    "director", "vp ", "head of"
]

NON_ENG_TITLE_KW = [
    "marketing", "sales", "accountant", "hr ", "recruiter",
    "operations manager", "business analyst", "customer support",
    "civil engineer", "mechanical engineer", "support engineer"
]


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def open_candidates_file(path):
    p = Path(path)
    if p.suffix == ".gz" or p.name.endswith(".jsonl.gz"):
        return gzip.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


# ---------------------------------------------------------------------
# Honeypot / trap detection
# ---------------------------------------------------------------------
def detect_honeypot(candidate):
    """Returns (is_honeypot, reason) for clearly impossible / synthetic profiles."""
    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    edu = candidate.get("education", []) or []

    expert_zero = sum(
        1
        for s in skills
        if s and s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 8:
        return True, "expert_zero_duration_stuffing"

    for job in history:
        if not job:
            continue
        dm = job.get("duration_months", 0)
        if dm is None:
            dm = 0
        if dm > 600 or dm < 0:
            return True, "impossible_job_duration"

    raw_yoe = profile.get("years_of_experience", 0)
    yoe_val = float(raw_yoe) if raw_yoe is not None else 0.0
    yoe_months = yoe_val * 12

    for s in skills:
        if not s:
            continue
        sdur = s.get("duration_months", 0)
        if sdur is None:
            sdur = 0
        if sdur > yoe_months + 24:
            return True, "skill_duration_exceeds_career"

    total_job_months = sum(
        j.get("duration_months", 0)
        for j in history
        if j and j.get("duration_months") is not None
    )
    if total_job_months > 0 and yoe_months > 0:
        if total_job_months > yoe_months * 1.6 + 24:
            return True, "career_history_exceeds_stated_yoe"

    grad_years = [e.get("end_year") for e in edu if e and e.get("end_year") is not None]
    if grad_years and yoe_val > 0:
        latest_grad = max(grad_years)
        years_since_grad = TODAY.year - latest_grad
        if years_since_grad < yoe_val - 1.5:
            return True, "experience_predates_graduation"

    for job in history:
        if job and job.get("is_current"):
            j_dur = job.get("duration_months", 0)
            if j_dur is None:
                j_dur = 0
            if j_dur > yoe_months + 6:
                return True, "current_job_duration_exceeds_total_experience"

    return False, None


def has_pure_service_background(history):
    companies = [
        j.get("company", "").lower() for j in history if j and j.get("company")
    ]
    if not companies:
        return False
    return all(any(svc in c for svc in BANNED_SERVICES) for c in companies)


def has_recent_product_experience(history):
    return not has_pure_service_background(history)


def keyword_hits(text, kw_list):
    text = (text or "").lower()
    return sum(1 for kw in kw_list if kw in text)


def career_text_blob(candidate):
    profile = candidate.get("profile", {}) or {}
    parts = [profile.get("headline", ""), profile.get("summary", "")]
    for job in candidate.get("career_history", []):
        if job:
            parts.append(job.get("title", ""))
            parts.append(job.get("description", ""))
    return " ".join(parts)


def skill_career_consistency(candidate):
    """Measures alignment between profile skills and historical career descriptions."""
    skills = candidate.get("skills", []) or []
    history = candidate.get("career_history", []) or []

    history_text = " ".join(job.get("description", "") for job in history if job and job.get("description")).lower()

    relevant_skills = 0
    matched_skills = 0

    for skill in skills:
        if not skill:
            continue
        skill_name = skill.get("name", "").lower()

        if len(skill_name) < 4:
            continue

        relevant_skills += 1

        if skill_name in history_text:
            matched_skills += 1

    if relevant_skills == 0:
        return 1.0  

    return matched_skills / relevant_skills


# ---------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------
def evaluate_candidate(candidate):
    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    if signals is None:
        signals = {}

    is_honeypot, honeypot_reason = detect_honeypot(candidate)
    if is_honeypot:
        return {"score": -1.0, "honeypot": True, "honeypot_reason": honeypot_reason}

    blob = career_text_blob(candidate)
    title = profile.get("current_title", "").lower()

    raw_yoe = profile.get("years_of_experience", 0)
    yoe = float(raw_yoe) if raw_yoe is not None else 0.0

    components = {}

    # --- A. Years of experience band ---
    if 5.0 <= yoe <= 9.0:
        components["yoe"] = 20.0
    elif 3.0 <= yoe < 5.0 or 9.0 < yoe <= 12.0:
        components["yoe"] = 8.0
    elif 12.0 < yoe <= 15.0:
        components["yoe"] = 3.0
    else:
        components["yoe"] = 0.0

    # --- B. Title signal ---
    title_score = 0.0
    if any(kw in title for kw in TIER1_TITLE_KW):
        title_score = 18.0
    elif any(kw in title for kw in ["engineer", "developer", "scientist"]) and any(
        kw in blob for kw in CORE_RETRIEVAL_KW + LLM_KW
    ):
        title_score = 10.0
    elif any(kw in title for kw in NON_ENG_TITLE_KW):
        title_score = -15.0

    if any(kw in title for kw in LEADERSHIP_ONLY_TITLE_KW):
        current_job = next((j for j in history if j and j.get("is_current")), None)
        hands_on = current_job and any(
            kw in current_job.get("description", "").lower()
            for kw in ["implement", "built", "wrote", "shipped", "deployed", "coded"]
        )
        title_score += -8.0 if not hands_on else -2.0
    components["title"] = title_score

    # --- C. Career substance ---
    substance_score = 0.0
    core_skill_hits = 0
    stuffed_skill_hits = 0
    deployed_evidence = keyword_hits(
        blob, ["production", "deployed", "shipped", "scale", "real users", "real-time"]
    )

    for s in skills:
        if not s:
            continue
        name = s.get("name", "").lower()
        prof = s.get("proficiency", "beginner")
        duration = s.get("duration_months", 0)
        if duration is None:
            duration = 0

        if any(kw in name for kw in CORE_RETRIEVAL_KW):
            core_skill_hits += 1
            if prof in ("advanced", "expert") and duration >= 12:
                substance_score += 4.0
            elif duration == 0:
                stuffed_skill_hits += 1
            else:
                substance_score += 1.0
        if any(kw in name for kw in LLM_KW):
            if prof in ("advanced", "expert") and duration >= 12:
                substance_score += 1.5
        if any(kw in name for kw in NON_FIT_DOMAIN_KW):
            substance_score -= 0.5

    history_retrieval_hits = keyword_hits(blob, CORE_RETRIEVAL_KW)
    if history_retrieval_hits >= 2 and deployed_evidence >= 1:
        substance_score += 12.0
    elif history_retrieval_hits >= 1:
        substance_score += 5.0

    pre_llm_evidence = any(
        kw in blob
        for kw in [
            "bm25", "tf-idf", "learning to rank", "lucene",
            "solr", "click model", "feature store",
        ]
    )
    if pre_llm_evidence:
        substance_score += 4.0

    # Cross-Verification Structural Logic Adjustment
    consistency = skill_career_consistency(candidate)
    if consistency < 0.4:
        substance_score -= 10.0  

    components["substance"] = min(substance_score, 45.0)
    components["stuffing_penalty"] = -2.0 * stuffed_skill_hits

    # --- D. Pure-services disqualifier ---
    pure_service_penalty = 0.0
    if has_pure_service_background(history):
        pure_service_penalty = -25.0
    components["services_penalty"] = pure_service_penalty

    # --- E. Location / relocation fit ---
    location_score = 0.0
    country = profile.get("country", "")
    location = profile.get("location", "").lower()
    pune_noida = any(c in location for c in ["pune", "noida"])
    tier1_india = any(
        c in location
        for c in [
            "hyderabad", "mumbai", "delhi", "bangalore", "bengaluru",
            "gurugram", "gurgaon", "ncr"
        ]
    )
    if country == "India" and pune_noida:
        location_score = 6.0
    elif country == "India" and tier1_india:
        location_score = 4.0
    elif country == "India":
        location_score = 2.0
    elif country and signals.get("willing_to_relocate"):
        location_score = -3.0
    else:
        location_score = -6.0
    components["location"] = location_score

    # --- F. Notice period ---
    raw_notice = signals.get("notice_period_days")
    notice = int(raw_notice) if raw_notice is not None else 60
    if notice <= 30:
        notice_score = 4.0
    elif notice <= 60:
        notice_score = 0.0
    else:
        notice_score = -3.0
    components["notice"] = notice_score

    # Consistency Alignment Metrics
    components["skill_career_consistency"] = consistency * 10
    if consistency < 0.15:
        components["skill_career_consistency"] -= 5

    base_score = sum(components.values())

    # --- G. Behavioral availability multiplier ---
    open_to_work_mult = 1.15 if signals.get("open_to_work_flag") else 1.0

    last_active = parse_date(signals.get("last_active_date", ""))
    freshness_mult = 1.0
    days_inactive = None
    if last_active:
        days_inactive = (TODAY - last_active).days
        if days_inactive <= 30:
            freshness_mult = 1.2
        elif days_inactive <= 90:
            freshness_mult = 1.0
        elif days_inactive <= 180:
            freshness_mult = 0.75
        else:
            freshness_mult = 0.45

    raw_resp = signals.get("recruiter_response_rate")
    response_rate = float(raw_resp) if raw_resp is not None else 0.0
    response_mult = 0.7 + (response_rate * 0.6)

    verification_mult = 1.0
    if not signals.get("verified_email", True):
        verification_mult -= 0.05
    if signals.get("avg_response_time_hours", 0) > 240:
        verification_mult -= 0.05

    behavioral_mult = (
        open_to_work_mult * freshness_mult * response_mult * verification_mult
    )
    components["behavioral_multiplier"] = round(behavioral_mult, 4)

    final_score = max(0.0, base_score) * behavioral_mult

    return {
        "score": final_score,
        "honeypot": False,
        "components": components,
        "days_inactive": days_inactive,
        "response_rate": response_rate,
        "notice": notice,
        "history_retrieval_hits": history_retrieval_hits,
        "deployed_evidence": deployed_evidence,
        "pure_service": has_pure_service_background(history),
    }


def top_named_skills(candidate, n=2):
    skills = candidate.get("skills", []) or []
    core = [
        s["name"]
        for s in skills
        if s and s.get("name") and any(kw in s.get("name", "").lower() for kw in CORE_RETRIEVAL_KW + LLM_KW)
    ]
    if core:
        return core[:n]
    return [s["name"] for s in skills[:n] if s and s.get("name")]


def generate_reasoning(candidate, eval_result):
    profile = candidate.get("profile", {}) or {}
    yoe = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Engineer")
    company = profile.get("current_company", "their current company")
    signals = candidate.get("redrob_signals", {}) or {}

    raw_resp = signals.get("recruiter_response_rate")
    resp_rate = float(raw_resp) if raw_resp is not None else 0.0

    notice = signals.get("notice_period_days", None)
    days_inactive = eval_result.get("days_inactive")
    skills_str = " and ".join(top_named_skills(candidate)) or "a generalist background"

    positives = []
    concerns = []

    if (
        eval_result["history_retrieval_hits"] >= 2
        and eval_result["deployed_evidence"] >= 1
    ):
        positives.append(
            "career history describes production retrieval/ranking work, not just listed skills"
        )
    elif eval_result["history_retrieval_hits"] >= 1:
        positives.append("career history shows some retrieval/search exposure")

    if eval_result["pure_service"]:
        concerns.append(
            "entire career history is at services firms (TCS/Infosys/Wipro-type), which the JD flags against"
        )

    consistency = skill_career_consistency(candidate)
    if consistency > 0.5:
        positives.append("skills are strongly supported by career history")
    elif consistency < 0.15:
        concerns.append("many listed skills lack supporting career evidence")

    if days_inactive is not None:
        if days_inactive > 180:
            concerns.append(f"inactive on the platform for {days_inactive} days")
        elif days_inactive <= 30:
            positives.append("recently active on the platform")

    if resp_rate < 0.2:
        concerns.append(f"low recruiter response rate ({int(resp_rate*100)}%)")
    elif resp_rate > 0.6:
        positives.append(f"strong recruiter responsiveness ({int(resp_rate*100)}%)")

    if notice is not None and notice > 90:
        concerns.append(f"long notice period ({notice} days)")

    pieces = [f"{yoe} YOE as {title} at {company}, background in {skills_str}."]
    if positives:
        pieces.append("Positives: " + "; ".join(positives) + ".")
    if concerns:
        pieces.append("Concerns: " + "; ".join(concerns) + ".")
    if not positives and not concerns:
        pieces.append(
            "Adjacent fit on experience band but limited direct signal either way."
        )

    return " ".join(pieces)


def run(candidates_path, out_path, top_n=100):
    results = []
    honeypot_count = 0
    total = 0

    print(f"Opening and parsing stream input data line-by-line from: {candidates_path}")
    with open_candidates_file(candidates_path) as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            candidate = json.loads(line)
            cid = candidate.get("candidate_id")
            ev = evaluate_candidate(candidate)
            if ev["honeypot"]:
                honeypot_count += 1
                continue
            results.append((ev["score"], cid, candidate, ev))

    results.sort(key=lambda x: (-x[0], x[1]))
    top = results[:top_n]

    max_score = top[0][0] if top else 1.0
    if max_score <= 0:
        max_score = 1.0

    print(f"Writing monotonic target records to output CSV path: {out_path}")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev_score = None
        for idx, (raw_score, cid, candidate, ev) in enumerate(top):
            rank = idx + 1
            norm_score = round(raw_score / max_score, 4)
            if prev_score is not None and norm_score > prev_score:
                norm_score = prev_score
            prev_score = norm_score
            reasoning = generate_reasoning(candidate, ev)
            writer.writerow([cid, rank, f"{norm_score:.4f}", reasoning])

    print(
        f"Processed {total} candidates. Honeypots excluded: {honeypot_count} "
        f"({honeypot_count/total*100:.2f}%). Wrote top {len(top)} to {out_path}."
    )


def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")

    parser.add_argument("--candidates", default=DEFAULT_INPUT_FILE)
    parser.add_argument("--out", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--top-n", type=int, default=100)

    args = parser.parse_args()

    output_dir = os.path.dirname(args.out)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    run(args.candidates, args.out, args.top_n)


if __name__ == "__main__":
    main()