import json
import os

BANNED_SERVICES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}

def inspect_dataset(file_path):
    print(f"Extracting target profile slices from {file_path}...")
    
    # Storage for our 4 cohorts
    top_scoring_candidates = []
    obvious_honeypots = []
    service_only_candidates = []
    recsys_candidates = []
    
    # We will use a basic version of your previous logic to catch the right groups
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            profile = candidate.get("profile", {})
            history = candidate.get("career_history", [])
            skills = candidate.get("skills", [])
            signals = candidate.get("redrob_signals", {})
            
            # --- Condition Checkers ---
            # 1. Honeypots
            expert_zero_duration = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
            has_glitch_duration = any(job.get("duration_months", 0) > 600 for job in history)
            is_honeypot = (expert_zero_duration >= 10) or has_glitch_duration
            
            # 2. Keywords
            has_retrieval = any(any(kw in s.get("name", "").lower() for kw in ["retrieval", "search", "ranking", "vector"]) for s in skills)
            has_recsys = any("recommendation" in s.get("name", "").lower() or "recommender" in s.get("name", "").lower() for s in skills)
            
            # 3. Service history
            companies = [job.get("company", "").lower() for job in history]
            is_service_only = len(companies) > 0 and all(
                any(service in comp for service in BANNED_SERVICES) for comp in companies if comp
            )
            
            # --- Sorting into Buckets ---
            if is_honeypot and len(obvious_honeypots) < 20:
                obvious_honeypots.append(candidate)
                
            if is_service_only and len(service_only_candidates) < 20:
                service_only_candidates.append(candidate)
                
            if has_recsys and len(recsys_candidates) < 20:
                recsys_candidates.append(candidate)
                
            # Treat active profiles with strong YOE and retrieval skills as placeholder top scorers
            yoe = profile.get("years_of_experience", 0)
            if not is_honeypot and not is_service_only and has_retrieval and (5.0 <= yoe <= 9.0) and len(top_scoring_candidates) < 20:
                top_scoring_candidates.append(candidate)
                
            # Break early if all 4 slices are fully saturated
            if (len(top_scoring_candidates) == 20 and len(obvious_honeypots) == 20 and 
                len(service_only_candidates) == 20 and len(recsys_candidates) == 20):
                break

    # Save outputs to distinct text files for clean inspection
    slices = {
        "inspect_top_scoring.txt": top_scoring_candidates,
        "inspect_honeypots.txt": obvious_honeypots,
        "inspect_service_only.txt": service_only_candidates,
        "inspect_recsys.txt": recsys_candidates
    }
    
    for filename, data in slices.items():
        with open(filename, "w", encoding="utf-8") as out_f:
            out_f.write(f"=== INSPECTION FILE: {filename} ({len(data)} profiles) ===\n\n")
            for idx, cand in enumerate(data, 1):
                out_f.write(f"--- Candidate #{idx} | ID: {cand.get('candidate_id')} ---\n")
                out_f.write(json.dumps(cand, indent=2))
                out_f.write("\n\n" + "="*80 + "\n\n")
        print(f" Saved {filename}")

if __name__ == "__main__":
    target_file = "candidates.jsonl"
    if os.path.exists(target_file):
        inspect_dataset(target_file)
    else:
        print(f"Error: Could not find '{target_file}' in your directory.")