"""
Global slide matching algorithm (order-independent).
Uses greedy best-match-first approach
Better for heavily reordered slides.
"""
from typing import List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> List[str]:
    """Simple tokenizer for Jaccard similarity."""
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


def global_best_match(S: np.ndarray, threshold: float = 0.55) -> List[Tuple[int, int]]:
    """
    Greedy best-match-first alignment (order-independent).
    
    Algorithm:
    1. Find the highest similarity score in the matrix
    2. Match those two slides
    3. Remove that row and column from consideration
    4. Repeat until no more matches above threshold
    
    Args:
        S: Similarity matrix [m x n]
        threshold: Minimum similarity to consider a match
        
    Returns:
        List of (old_idx, new_idx) pairs sorted by similarity
    """
    m, n = S.shape
    matches = []
    
    # Create a copy to work with
    S_work = S.copy()
    used_old = set()
    used_new = set()
    
    while True:
        # Find the best remaining match
        max_val = -1
        best_i, best_j = -1, -1
        
        for i in range(m):
            if i in used_old:
                continue
            for j in range(n):
                if j in used_new:
                    continue
                if S_work[i, j] > max_val:
                    max_val = S_work[i, j]
                    best_i, best_j = i, j
        
        # Stop if no match above threshold
        if max_val < threshold:
            break
        
        # Record the match
        matches.append((best_i, best_j, max_val))
        used_old.add(best_i)
        used_new.add(best_j)
    
    # Sort by similarity (highest first) for better presentation
    matches.sort(key=lambda x: x[2], reverse=True)
    
    # Return just the indices
    return [(i, j) for i, j, _ in matches]


def detect_splits_merges_global(A: List[str], B: List[str], S: np.ndarray, 
                                matches: List[Tuple[int, int]], 
                                sim_thresh: float = 0.55):
    """
    Detect splits and merges for global matching.
    
    Args:
        A: Old slides
        B: New slides
        S: Similarity matrix
        matches: List of (old_idx, new_idx) primary matches from global_best_match
        sim_thresh: Threshold for considering additional matches
        
    Returns:
        Dictionary with pairs, splits, merges, inserts, deletes
    """
    results = {"pairs": [], "splits": [], "merges": [], "inserts": [], "deletes": []}
    
    # Convert matches to set for fast lookup
    matched_old = set(i for i, _ in matches)
    matched_new = set(j for _, j in matches)
    
    # Detect splits FIRST: one old slide → multiple new slides
    splits_found = set()
    splits_new_slides = set()
    
    for i in range(len(A)):
        if i not in matched_old:
            continue
            
        # Find all new slides with high similarity to this old slide
        high_sim_new = []
        for j in range(len(B)):
            if S[i, j] >= sim_thresh:
                high_sim_new.append(j)
        
        # If matched to 2+ new slides, it's a split
        if len(high_sim_new) >= 2:
            high_sim_new.sort(key=lambda j: S[i, j], reverse=True)
            results["splits"].append((i, high_sim_new))
            splits_found.add(i)
            splits_new_slides.update(high_sim_new)
    
    # Detect merges: multiple old slides → one new slide
    merges_found = set()
    merges_old_slides = set()
    
    for j in range(len(B)):
        if j not in matched_new:
            continue
            
        # Find all old slides with high similarity to this new slide
        # Exclude old slides already identified as splits
        high_sim_old = []
        for i in range(len(A)):
            if i not in splits_found and S[i, j] >= sim_thresh:
                high_sim_old.append(i)
        
        # If 2+ old slides match this new slide, it's a merge
        if len(high_sim_old) >= 2:
            high_sim_old.sort(key=lambda i: S[i, j], reverse=True)
            results["merges"].append((high_sim_old, j))
            merges_found.add(j)
            merges_old_slides.update(high_sim_old)
    
    # NOW build pairs list: only include matches NOT involved in splits/merges
    for old_idx, new_idx in matches:
        if old_idx not in splits_found and new_idx not in merges_found:
            results["pairs"].append((old_idx, new_idx))
    
    # Find unmatched slides (excluding those in splits/merges)
    for i in range(len(A)):
        if i not in matched_old and i not in splits_found and i not in merges_old_slides:
            results["deletes"].append(i)
    
    for j in range(len(B)):
        if j not in matched_new and j not in merges_found and j not in splits_new_slides:
            results["inserts"].append(j)
    
    return results


def compare_slide_versions_global(old_slides: List[str], new_slides: List[str], 
                                  embedder: Optional[SlideEmbedder] = None,
                                  w_cos: float = 0.7, w_jac: float = 0.3,
                                  sim_thresh: float = 0.55) -> dict:
    """
    Compare two versions of slides using global best-match algorithm.
    Order-independent - works well for heavily reordered slides.
    
    Args:
        old_slides: List of slide texts from old version
        new_slides: List of slide texts from new version  
        embedder: SlideEmbedder instance (creates new one if None)
        w_cos: Weight for cosine similarity (0-1)
        w_jac: Weight for Jaccard similarity (0-1)
        sim_thresh: Similarity threshold for matching
        
    Returns:
        Dictionary with alignment results and detected changes
    """
    if embedder is None:
        embedder = SlideEmbedder()
    
    # Compute similarity matrix
    S = similarity_matrix(embedder, old_slides, new_slides, w_cos, w_jac)
    
    # Find best matches globally (order-independent)
    matches = global_best_match(S, threshold=sim_thresh)
    
    # Detect splits and merges
    results = detect_splits_merges_global(old_slides, new_slides, S, matches, sim_thresh)
    
    results["similarity_matrix"] = S
    results["match_method"] = "global_best_match"
    
    return results


# ---------- demo ----------

if __name__ == "__main__":
    print("🚀 Testing global matching with split detection...")
    
    # Example that triggers the bug: old slide 0 matches both new slides 0 and 1 (split)
    old_slides = [
        "Lernziele: Datenstrukturen, Listen, Queue, Stack, Heap, Bäume, Algorithmen, Suchen, Sortieren, Aufwandsabschätzung, Korrektheit",
        "Slide B: Other content",
        "Slide C: More content"
    ]
    
    # New version: old slide 0 was split into two slides
    new_slides = [
        "Lernziele: Datenstrukturen, Listen, Queue (Warteschlange), Stack (Stapel), Heap (Haufen), Bäume",  # First half
        "Lernziele: Elementare Algorithmen, Suchen, Sortieren, Aufwandsabschätzung (Komplexität), Korrektheit",  # Second half
        "Slide B: Other content",
        "Slide C: More content"
    ]
    
    # Run global comparison
    results = compare_slide_versions_global(old_slides, new_slides, sim_thresh=0.55)
    
    print("\n" + "="*60)
    print("📈 GLOBAL MATCHING RESULTS")
    print("="*60)
    print("Similarity matrix:")
    print(np.round(results["similarity_matrix"], 2))
    
    print(f"\n📋 Matched pairs (should NOT include old[0] if it's a split):")
    for old_idx, new_idx in results["pairs"]:
        sim_score = results["similarity_matrix"][old_idx, new_idx]
        print(f"  Old[{old_idx}] ↔ New[{new_idx}] (sim={sim_score:.3f})")
    
    print(f"\n📄 Splits detected:")
    for old_idx, new_indices in results["splits"]:
        print(f"  Old[{old_idx}] → New{new_indices}")
        for new_idx in new_indices:
            sim_score = results["similarity_matrix"][old_idx, new_idx]
            print(f"    - New[{new_idx}]: '{new_slides[new_idx][:60]}...' (sim={sim_score:.3f})")
    
    print(f"\n➕ New slides added:")
    for new_idx in results["inserts"]:
        print(f"  New[{new_idx}]: '{new_slides[new_idx]}'")
    
    print(f"\n➖ Slides removed:")
    for old_idx in results["deletes"]:
        print(f"  Old[{old_idx}]: '{old_slides[old_idx]}'")
    
    # Verify no duplicates
    print("\n" + "="*60)
    print("✅ VERIFICATION")
    print("="*60)
    
    # Check old slide 0
    old_0_in_pairs = any(old_idx == 0 for old_idx, _ in results["pairs"])
    old_0_in_splits = any(old_idx == 0 for old_idx, _ in results["splits"])
    
    if old_0_in_pairs and old_0_in_splits:
        print("❌ BUG: Old[0] appears in BOTH pairs and splits!")
    elif old_0_in_splits:
        print("✅ Old[0] correctly appears only in splits (not in pairs)")
    
    # Check new slides 0 and 1
    new_0_in_pairs = any(new_idx == 0 for _, new_idx in results["pairs"])
    new_1_in_pairs = any(new_idx == 1 for _, new_idx in results["pairs"])
    new_0_in_inserts = 0 in results["inserts"]
    new_1_in_inserts = 1 in results["inserts"]
    
    if (new_0_in_pairs or new_1_in_pairs) and (new_0_in_inserts or new_1_in_inserts):
        print("❌ BUG: Split slides appear in BOTH pairs/inserts!")
    elif not new_0_in_pairs and not new_1_in_pairs and not new_0_in_inserts and not new_1_in_inserts:
        print("✅ New[0] and New[1] correctly appear only in splits (not in pairs or inserts)")
    else:
        print(f"⚠️  New[0] in pairs: {new_0_in_pairs}, in inserts: {new_0_in_inserts}")
        print(f"⚠️  New[1] in pairs: {new_1_in_pairs}, in inserts: {new_1_in_inserts}")
