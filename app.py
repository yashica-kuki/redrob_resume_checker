import streamlit as st
import json
import csv
import io
# Import your exact evaluation engine from your local file
from ranker import evaluate_candidate, generate_reasoning 

st.title("Redrob Candidate Ranker — Sandbox Environment")
st.write("Upload a small candidate sample file (.jsonl) to run the ranking engine end-to-end.")

# Create a clean file uploader widget
uploaded_file = st.file_uploader("Choose a sample candidate file", type=["jsonl", "txt"])

if uploaded_file is not None:
    results = []
    
    # Read the uploaded stream line by line
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    for line in stringio:
        if not line.strip():
            continue
        candidate = json.loads(line)
        cid = candidate.get("candidate_id")
        ev = evaluate_candidate(candidate)
        
        if not ev.get("honeypot", False) and ev.get("score", -1) >= 0:
            results.append((ev["score"], cid, candidate, ev))
            
    # Sort deterministically
    results.sort(key=lambda x: (-x[0], x[1]))
    top_candidates = results[:100]
    
    if top_candidates:
        max_score = top_candidates[0][0] if top_candidates[0][0] > 0 else 1.0
        
        # Build the structured CSV bytes buffer
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        prev_score = None
        for idx, (raw_score, cid, candidate, ev) in enumerate(top_candidates):
            rank = idx + 1
            norm_score = round(raw_score / max_score, 4)
            if prev_score is not None and norm_score > prev_score:
                norm_score = prev_score
            prev_score = norm_score
            reasoning = generate_reasoning(candidate, ev)
            writer.writerow([cid, rank, f"{norm_score:.4f}", reasoning])
            
        st.success(f"Successfully ranked {len(top_candidates)} candidates!")
        
        # Provide an instant download button for the reviewer
        st.download_button(
            label="Download Ranked Submission CSV",
            data=output.getvalue(),
            file_name="sandbox_submission.csv",
            mime="text/csv"
        )
    else:
        st.warning("No valid, non-honeypot candidates found in the uploaded sample slice.")