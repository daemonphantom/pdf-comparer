# Markdown slide comparer using embeddings + Jaccard + Needleman-Wunsch alignment
# - Each item in A/B is a "slide" (e.g., a paragraph or cleaned slide text)
# - Cosine uses real embeddings from sentence transformers
# - Jaccard uses simple token sets (lowercased; add German lemma/compound splitting later)
# - Supports detection of splits/merges between old and new slide versions
"""
Slide comparer using embeddings + Jaccard + Needleman-Wunsch alignment.
Detects changes, splits, and merges between slide versions.
"""
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

def tokenize(text: str) -> List[str]:
    """Simple tokenizer for Jaccard similarity."""
    # swap in spaCy-de + lemma + compound-split later
    return [t for t in ''.join(ch.lower() if ch.isalnum() or ch.isspace() else ' ' for ch in text).split() if t]

def jaccard(a: str, b: str) -> float:
    """Compute Jaccard similarity between two text strings."""
    A, B = set(tokenize(a)), set(tokenize(b))
    if not A and not B: return 1.0
    return len(A & B) / max(1, len(A | B))

class SlideEmbedder:
    """Embedder for slide text using SentenceTransformers"""
    def __init__(self, model_name: str = "distiluse-base-multilingual-cased-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_list(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

def cosine_similarity_matrix(embedder: SlideEmbedder, A: List[str], B: List[str]) -> np.ndarray:
    """Compute cosine similarity matrix between two lists of slide texts."""
    if not A or not B:
        return np.zeros((len(A), len(B)))
    
    embeddings_A = embedder.embed_list(A)
    embeddings_B = embedder.embed_list(B)
    
    # Normalize for cosine similarity
    norm_A = np.linalg.norm(embeddings_A, axis=1, keepdims=True) + 1e-9
    norm_B = np.linalg.norm(embeddings_B, axis=1, keepdims=True) + 1e-9
    embeddings_A_norm = embeddings_A / norm_A
    embeddings_B_norm = embeddings_B / norm_B
    
    return embeddings_A_norm @ embeddings_B_norm.T

def similarity_matrix(embedder: SlideEmbedder, A: List[str], B: List[str], w_cos=0.7, w_jac=0.3) -> np.ndarray:
    """Compute combined similarity matrix using embeddings + Jaccard similarity"""
    cos = cosine_similarity_matrix(embedder, A, B)
    jac = np.zeros_like(cos)
    for i in range(len(A)):
        for j in range(len(B)):
            jac[i,j] = jaccard(A[i], B[j])
    return w_cos * cos + w_jac * jac

def nw_align(S: np.ndarray, gap_penalty: float = -0.3, band: Optional[int]=None) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    Needleman-Wunsch global alignment.
    S[i,j] = match score for A[i] vs B[j] (higher is better).
    Returns path as list of (old_idx or None, new_idx or None).
    """
    m, n = S.shape
    DP = np.full((m+1, n+1), -1e9)
    BT = np.zeros((m+1, n+1), dtype=np.int8)  # 1=diag, 2=up (gap in B), 3=left (gap in A)
    DP[0,0] = 0.0
    for i in range(1, m+1):
        DP[i,0] = DP[i-1,0] + gap_penalty
        BT[i,0] = 2
    for j in range(1, n+1):
        DP[0,j] = DP[0,j-1] + gap_penalty
        BT[0,j] = 3

    for i in range(1, m+1):
        j_start, j_end = 1, n
        if band is not None:
            j_start = max(1, i - band)
            j_end   = min(n, i + band)
        for j in range(j_start, j_end+1):
            score_match = DP[i-1,j-1] + S[i-1,j-1]
            score_up = DP[i-1,j] + gap_penalty
            score_left = DP[i,j-1] + gap_penalty
            best = score_match
            bt = 1
            if score_up > best:
                best = score_up
                bt = 2
            if score_left > best:
                best = score_left
                bt = 3
            DP[i,j] = best
            BT[i,j] = bt

        if band is not None:
            for j in range(1, j_start):
                DP[i,j] = max(DP[i-1,j] + gap_penalty, DP[i,j-1] + gap_penalty)
                BT[i,j] = 2 if DP[i-1,j] >= DP[i,j-1] else 3
            for j in range(j_end+1, n+1):
                DP[i,j] = max(DP[i-1,j] + gap_penalty, DP[i,j-1] + gap_penalty)
                BT[i,j] = 2 if DP[i-1,j] >= DP[i,j-1] else 3
    
    i, j = m, n
    path: List[Tuple[Optional[int], Optional[int]]] = []
    while i > 0 or j > 0:
        bt = BT[i,j]
        if bt == 1:
            path.append((i-1, j-1))
            i -= 1
            j -= 1
        elif bt == 2:
            path.append((i-1, None))
            i -= 1
        else:
            path.append((None, j-1))
            j -= 1
    path.reverse()
    return path

def detect_splits_merges(A: List[str], B: List[str], S: np.ndarray, path, sim_thresh=0.6):
    """
    Detect splits and merges between slide versions.
    Splits: One old slide → multiple new slides
    Merges: Multiple old slides → one new slide
    """
    results = {"pairs": [], "splits": [], "merges": [], "inserts": [], "deletes": []}
    
    aligned = []
    for step in path:
        i, j = step
        if i is not None and j is not None:
            aligned.append((i, j))
        elif i is not None:
            results["deletes"].append(i)
        elif j is not None:
            results["inserts"].append(j)

    results["pairs"] = aligned
    
    splits_found = set()
    merges_found = set()
    
    # Detect splits: one old slide → multiple new slides
    for i in range(len(A)):
        high_sim_new = []
        for j in range(len(B)):
            if S[i, j] >= sim_thresh:
                high_sim_new.append(j)
        
        if len(high_sim_new) >= 2:
            high_sim_new.sort(key=lambda j: S[i, j], reverse=True)
            results["splits"].append((i, high_sim_new))
            splits_found.add(i)
            for j in high_sim_new:
                if j in results["inserts"]:
                    results["inserts"].remove(j)
    
    # Detect merges: multiple old slides → one new slide
    for j in range(len(B)):
        if j in merges_found:
            continue
        
        high_sim_old = []
        for i in range(len(A)):
            if i not in splits_found and S[i, j] >= sim_thresh:
                high_sim_old.append(i)
        
        # If new slide j is similar to 2+ old slides, it's likely a merge
        if len(high_sim_old) >= 2:
            # Sort by similarity (highest first)
            high_sim_old.sort(key=lambda i: S[i, j], reverse=True)
            results["merges"].append((high_sim_old, j))
            merges_found.add(j)
            # Remove these from deletes since they're explained by the merge
            for i in high_sim_old:
                if i in results["deletes"]:
                    results["deletes"].remove(i)
    
    # Clean up pairs: remove splits/merges and weak alignments
    clean_pairs = []
    pair_threshold = sim_thresh * 0.8  # slightly lower than split/merge threshold
    
    for (i, j) in results["pairs"]:
        is_in_split = any(i == split_old for split_old, _ in results["splits"])
        is_in_merge = any(j == merge_new for _, merge_new in results["merges"])
        has_good_similarity = S[i, j] >= pair_threshold
        
        if not is_in_split and not is_in_merge and has_good_similarity:
            clean_pairs.append((i, j))
        elif not is_in_split and not is_in_merge and not has_good_similarity:
            # This was a forced alignment - treat as separate delete + insert
            if i not in results["deletes"]:
                results["deletes"].append(i)
            if j not in results["inserts"]:
                results["inserts"].append(j)
    
    results["pairs"] = clean_pairs
    return results

def compare_slide_versions(old_slides: List[str], new_slides: List[str], 
                         embedder: Optional[SlideEmbedder] = None,
                         w_cos: float = 0.7, w_jac: float = 0.3,
                         gap_penalty: float = -0.3, band: Optional[int] = 3,
                         sim_thresh: float = 0.55) -> dict:
    """
    Compare two versions of slides and detect changes.
    
    Args:
        old_slides: List of slide texts from old version
        new_slides: List of slide texts from new version  
        embedder: SlideEmbedder instance (creates new one if None)
        w_cos: Weight for cosine similarity (0-1)
        w_jac: Weight for Jaccard similarity (0-1)
        gap_penalty: Penalty for alignment gaps (negative)
        band: Band constraint for NW alignment (None for no constraint)
        sim_thresh: Similarity threshold for split/merge detection
        
    Returns:
        Dictionary with alignment results and detected changes
    """
    if embedder is None:
        embedder = SlideEmbedder()
    
    S = similarity_matrix(embedder, old_slides, new_slides, w_cos, w_jac)
    path = nw_align(S, gap_penalty=gap_penalty, band=band)
    results = detect_splits_merges(old_slides, new_slides, S, path, sim_thresh)
    
    results["similarity_matrix"] = S
    results["alignment_path"] = path
    
    return results

# ---------- demo ----------

if __name__ == "__main__":
    print("🚀 Initializing slide comparer with real embeddings...")
    
    # Example slides from old and new lecture versions
    old_slides = [
        "Termine IntroProg: Kickoff 17.10, Test 27.02.",
        "Prüfungsmodalitäten: 13 Programmieraufgaben, 4 Theorieaufgaben, Klausurtermine.",
        "Projektwerkstätte & Codebeispiele",
        "Abgabe der Programmieraufgaben: Regel Freitag 20:00, siehe ISIS."
    ]
    new_slides = [
        "TermineIntroProg: Kickoff 17.10, Test 28.02.",  # minor rename + date tweak
        "Lehr- und Lernkonzept: Vorstellung der Konzepte & Beispiel-Programme.",  # new content
        "Anmeldefristen: Anmeldung 14.10-08.11, Empfehlung: erster Termin 28.02.",  # new content
        "Projektwerkstätte",
        "Codebeispiele",
        "Abgabe der Programmieraufgaben: Regel Freitag 20:00, siehe ISIS."  # same
    ]

    # Run comparison with lower threshold to catch the split
    results = compare_slide_versions(old_slides, new_slides, sim_thresh=0.55)

    print("\n" + "="*50)
    print("📈 SLIDE COMPARISON RESULTS")
    print("="*50)
    print("Similarity matrix:\n", np.round(results["similarity_matrix"], 3))
    print("\nNW alignment (old_idx, new_idx):", results["alignment_path"])
    print("Aligned pairs:", results["pairs"])
    print("Inserted (new):", results["inserts"])
    print("Deleted (old):", results["deletes"])
    print("Splits 1→2:", results["splits"])
    print("Merges 2→1:", results["merges"])
    
    # Interpretation
    print("\n" + "="*50)
    print("� INTERPRETATION")
    print("="*50)
    
    if results["pairs"]:
        print(f"✅ {len(results['pairs'])} slides remained similar between versions")
        for old_idx, new_idx in results["pairs"]:
            print(f"   Old slide {old_idx} ↔ New slide {new_idx}")
    
    if results["inserts"]:
        print(f"➕ {len(results['inserts'])} new slides added")
        for new_idx in results["inserts"]:
            print(f"   New slide {new_idx}: '{new_slides[new_idx][:50]}...'")
    
    if results["deletes"]:
        print(f"➖ {len(results['deletes'])} slides removed")
        for old_idx in results["deletes"]:
            print(f"   Removed slide {old_idx}: '{old_slides[old_idx][:50]}...'")
    
    if results["splits"]:
        print(f"📄 {len(results['splits'])} slides split (1→2)")
        for old_idx, new_indices in results["splits"]:
            print(f"   Old slide {old_idx} → New slides {new_indices}")
    
    if results["merges"]:
        print(f"📑 {len(results['merges'])} slides merged (2→1)")
        for old_indices, new_idx in results["merges"]:
            print(f"   Old slides {old_indices} → New slide {new_idx}")
