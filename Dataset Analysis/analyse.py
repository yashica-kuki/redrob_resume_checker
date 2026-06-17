import json
import collections
import math
import os

# Define the exact service companies banned by the JD
BANNED_SERVICES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}

def run_analysis(file_path):
    print(f"Starting analysis on {file_path}... (Processing as a stream to save RAM)")
    
    # Counters and aggregators
    total_candidates = 0
    retrieval_ranking_count = 0
    rec_system_count = 0
    service_only_count = 0
    
    yoe_list = []
    title_counts = collections.Counter()
    
    # For correlation: lists to store matching pairs of data
    # We will compute correlation between various signals and 'saved_by_recruiters_30d'
    signal_keys = [
        "profile_completeness_score", 
        "profile_views_received_30d", 
        "applications_submitted_30d", 
        "recruiter_response_rate", 
        "avg_response_time_hours",
        "connection_count",
        "endorsements_received",
        "github_activity_score",
        "search_appearance_30d",
        "interview_completion_rate"
    ]
    
    correlation_data = {key: [] for key in signal_keys}
    recruiter_saves = []

    # Stream read the JSONL file line by line
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            total_candidates += 1
            profile = candidate.get("profile", {})
            history = candidate.get("career_history", [])
            skills = candidate.get("skills", [])
            signals = candidate.get("redrob_signals", {})
            
            # --- 1 & 2. Keyword Search in Skills ---
            has_retrieval_ranking = False
            has_rec_system = False
            
            for s in skills:
                skill_name = s.get("name", "").lower()
                if any(kw in skill_name for kw in ["retrieval", "search", "ranking", "embedding", "vector", "pinecone", "milvus", "weaviate", "faiss", "opensearch", "elasticsearch"]):
                    has_retrieval_ranking = True
                if "recommendation" in skill_name or "recommender" in skill_name:
                    has_rec_system = True
                    
            if has_retrieval_ranking:
                retrieval_ranking_count += 1
            if has_rec_system:
                rec_system_count += 1
                
            # --- 3. Service-Only Check ---
            companies = [job.get("company", "").lower() for job in history]
            # Valid candidate has a history, and every single company belongs to the banned service list
            is_service_only = len(companies) > 0 and all(
                any(service in comp for service in BANNED_SERVICES) for comp in companies if comp
            )
            if is_service_only:
                service_only_count += 1
                
            # --- 4. YOE Tracking ---
            yoe = profile.get("years_of_experience")
            if yoe is not None:
                yoe_list.append(float(yoe))
                
            # --- 5. Title Tracking ---
            current_title = profile.get("current_title")
            if current_title:
                title_counts[current_title.strip()] += 1
                
            # --- 6. Correlation Extraction ---
            saves = signals.get("saved_by_recruiters_30d")
            if saves is not None:
                recruiter_saves.append(float(saves))
                for key in signal_keys:
                    val = signals.get(key, 0.0)
                    # Convert booleans or handle missing data safely
                    correlation_data[key].append(float(val) if val is not None else 0.0)

    # --- Calculations & Math Transformations ---
    
    # YOE Distribution calculations
    yoe_list.sort()
    n_yoe = len(yoe_list)
    mean_yoe = sum(yoe_list) / n_yoe if n_yoe > 0 else 0
    
    def get_percentile(sorted_data, percentile):
        if not sorted_data: return 0
        idx = (len(sorted_data) - 1) * percentile
        floor_idx = int(math.floor(idx))
        ceil_idx = int(math.ceil(idx))
        if floor_idx == ceil_idx:
            return sorted_data[int(idx)]
        return sorted_data[floor_idx] * (ceil_idx - idx) + sorted_data[ceil_idx] * (idx - floor_idx)

    p25 = get_percentile(yoe_list, 0.25)
    p50 = get_percentile(yoe_list, 0.50)  # Median
    p75 = get_percentile(yoe_list, 0.75)
    min_yoe = yoe_list[0] if yoe_list else 0
    max_yoe = yoe_list[-1] if yoe_list else 0

    # Pearson Correlation Coefficient Calculation function
    def calculate_correlation(x, y):
        n = len(x)
        if n == 0: return 0.0
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        den_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        den_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        if den_x == 0 or den_y == 0:
            return 0.0
        return num / math.sqrt(den_x * den_y)

    correlations = {}
    for key in signal_keys:
        correlations[key] = calculate_correlation(correlation_data[key], recruiter_saves)

    # Sort correlations by absolute strength descending
    sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    # --- DISPLAYING COMPREHENSIVE OUTPUT REPORT ---
    print("\n" + "="*50)
    print("      REDROB DATASET INSIGHT ANALYSIS REPORT      ")
    print("="*50)
    print(f"Total Profiles Processed: {total_candidates}\n")
    
    print("1. KEYWORD FOOTPRINTS:")
    print(f"  • Candidates with Retrieval/Ranking keywords:   {retrieval_ranking_count} ({retrieval_ranking_count/total_candidates*100:.2f}%)")
    print(f"  • Candidates with Recommendation keywords:     {rec_system_count} ({rec_system_count/total_candidates*100:.2f}%)")
    print(f"  • Candidates with pure Service-Only history:    {service_only_count} ({service_only_count/total_candidates*100:.2f}%)\n")
    
    print("2. YEARS OF EXPERIENCE (YOE) DISTRIBUTION:")
    print(f"  • Min YOE:    {min_yoe:.1f} years")
    print(f"  • 25th %ile:  {p25:.1f} years")
    print(f"  • Median:     {p50:.1f} years")
    print(f"  • 75th %ile:  {p75:.1f} years")
    print(f"  • Max YOE:    {max_yoe:.1f} years")
    print(f"  • Mean YOE:   {mean_yoe:.1f} years\n")
    
    print("3. TOP 10 MOST COMMON JOB TITLES IN POOL:")
    for rank, (title, count) in enumerate(title_counts.most_common(10), 1):
        print(f"  {rank:2d}. {title:<30} : {count} candidates")
        
    print("\n4. BEHAVIORAL SIGNAL CORRELATIONS WITH RECRUITER SAVES:")
    print("   (Sorted by absolute Pearson Correlation Coefficient |r| )")
    for signal, r_val in sorted_correlations:
        direction = "Positive ⬆" if r_val > 0 else "Negative ⬇"
        if abs(r_val) < 0.1: direction = "Negligible -"
        print(f"  • {signal:<28} : r = {r_val:+6.3f} ({direction})")
    print("="*50)

if __name__ == "__main__":
    # If your file is named slightly differently or unzipped as something else, change this path string!
    target_file = "candidates.jsonl" 
    
    if not os.path.exists(target_file):
        # Alternative loop fallback check for zipped file layout option
        if os.path.exists("candidates.jsonl.gz"):
            print("Detected zipped file format. Please extract it first via terminal: gunzip -k candidates.jsonl.gz")
        else:
            print(f"Error: Could not find '{target_file}' in your current working directory.")
    else:
        run_analysis(target_file)