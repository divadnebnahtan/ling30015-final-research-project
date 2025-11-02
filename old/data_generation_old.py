from collections import Counter, defaultdict, deque
import os
import re
import time
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from conllu import parse_incr, parse
from sentence_transformers import SentenceTransformer
from fastcoref import FCoref
import hanlp

from nltk.corpus import wordnet as wn
from nltk.corpus import verbnet as vn
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

from gensim import corpora, models, matutils
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess

logging.getLogger("gensim").setLevel(logging.ERROR)
logging.getLogger("hanlp").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*pynvml package is deprecated.*"
)

CONLLU_DIRECTORY = "C:/Users/Nathan/Downloads/Universal Dependencies 2.16/ud-treebanks-v2.16/ud-treebanks-v2.16/UD_English-GUM/"
CONLLU_FILES = [
    "en_gum-ud-test.conllu",
    "en_gum-ud-dev.conllu",
    "en_gum-ud-train.conllu"
]

VALID_VERB_XPOS = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]
STATIVE_VERBS = {"be", "seem", "know", "believe", "want", "contain", "resemble"}
REPORTING_VERBS = {"say", "report", "state", "claim", "announce", "describe"}
HEDGE_MARKERS = {"it", "was", "decided", "appears", "seems", "may"}

MAX_SENTENCES: int = -1

def get_conllu_paths():
    return [os.path.join(CONLLU_DIRECTORY, filename) for filename in CONLLU_FILES]


def load_corpus_sentences(conllu_paths, max_sentences=None):
    count = 0
    for path in conllu_paths:
        with open(path, encoding="utf-8") as f:
            for sentence in parse_incr(f):
                yield sentence
                count += 1
                if max_sentences is not None and count >= max_sentences:
                    return


def load_corpus_documents(conllu_paths, max_sentences=None):
    count = 0
    for path in conllu_paths:
        with open(path, encoding="utf-8") as f:
            current_doc = []
            current_sent_lines = []

            for line in f:
                line = line.strip()

                if not line:
                    if current_sent_lines:
                        sentence = parse("\n".join(current_sent_lines))[0]
                        current_doc.append(sentence)
                        current_sent_lines = []
                        count += 1
                        if max_sentences is not None and count >= max_sentences:
                            if current_doc:
                                yield current_doc
                            return
                    continue

                if line.startswith("# newdoc id"):
                    if current_doc:
                        yield current_doc
                        current_doc = []
                    continue

                current_sent_lines.append(line)

            if current_sent_lines:
                sentence = parse("\n".join(current_sent_lines))[0]
                current_doc.append(sentence)
                current_sent_lines = []
                count += 1
                if max_sentences is not None and count >= max_sentences:
                    if current_doc:
                        yield current_doc
                    return
            if current_doc:
                yield current_doc


def convert_documents_to_text(documents):
    return [" ".join(t["form"] for s in doc for t in s) for doc in documents]

def compute_lemma_statistics(sentences):
    lemma_stats = defaultdict(lambda: {
        "freq": 0,
        "passive_count": 0,
        "transitive_count": 0,
        "total_occurrences": 0
    })

    for sentence in sentences:
        for token in sentence:
            if token["upos"] != "VERB":
                continue
                
            lemma = token.get("lemma", "").lower()
            lemma_stats[lemma]["freq"] += 1
            lemma_stats[lemma]["total_occurrences"] += 1

            feats = token.get("feats", {}) or {}
            if feats.get("Voice") == "Pass" or any(
                t.get("deprel") == "aux:pass" and t.get("head") == token["id"]
                for t in sentence
            ):
                lemma_stats[lemma]["passive_count"] += 1

            if any(
                t.get("head") == token["id"] and t.get("deprel") in ("obj", "iobj")
                for t in sentence
            ):
                lemma_stats[lemma]["transitive_count"] += 1

    total_freq = sum(s["freq"] for s in lemma_stats.values())
    for lemma, stats in lemma_stats.items():
        total = stats["total_occurrences"]
        stats["passive_rate"] = stats["passive_count"] / total if total else 0.0
        stats["transitivity"] = stats["transitive_count"] / total if total else 0.0
        stats["freq_per_million"] = stats["freq"] / total_freq * 1_000_000 if total_freq else 0.0

    return lemma_stats


def compute_stopwords():
    english_stopwords = set(stopwords.words('english'))
    
    gum_extra_stops = {
        'uh', 'um', 'yeah', 'okay', 'hmm', 'uhh', 'uhm',  # spoken fillers
        'gon', 'na', 'wan', 'ta', 'im', 'aint',  # contractions
        'section', 'figure', 'example', 'table', 'chapter',  # academic markers
        'interviewer', 'interviewee', 'speaker', 'paragraph', 'sentence'
    }
    
    return list(english_stopwords.union(gum_extra_stops))

def load_nlp_models():
    print("Loading SRL model")
    srl_model = hanlp.load(
        hanlp.pretrained.mtl.EN_TOK_LEM_POS_NER_SRL_UDEP_SDP_CON_MODERNBERT_BASE
    )
    
    print("Loading sentence transformer model")
    sentence_model = SentenceTransformer("all-mpnet-base-v2")
    
    return srl_model, sentence_model


def load_coreference_model(doc_texts):
    print("Loading coreference model")
    coref_model = FCoref(device="cpu")
    
    print("Predicting coreference")
    coref_output = coref_model.predict(texts=doc_texts, max_tokens_in_batch=20000)
    
    return coref_model, coref_output


def load_lda_model(doc_texts, stop_words, 
                   num_topics: int = 15, iterations: int = 200):
    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=stop_words,
        token_pattern=r'\b[a-zA-Z]{3,}\b',
        # min_df=3,
        # max_df=0.8
    )
    
    print(f"Loading LDA model with {num_topics} topics and {iterations} iterations")
    doc_corpus_bow = vectorizer.fit_transform(doc_texts)
    gensim_corpus = matutils.Sparse2Corpus(doc_corpus_bow, documents_columns=False)
    id2word = dict((v, k) for k, v in vectorizer.vocabulary_.items())
    
    lda_model = models.LdaModel(
        corpus=gensim_corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=42,
        passes=20,
        iterations=iterations,
        alpha='auto',
        eta='auto',
        eval_every=None,
    )

    print("Computing coherence score")
    texts = [simple_preprocess(doc) for doc in doc_texts]
    coherence_id2word = corpora.Dictionary(texts)
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=coherence_id2word,
        coherence='c_v',
        processes=1
    )
    coherence_score = coherence_model.get_coherence()
    
    return vectorizer, lda_model, coherence_model, coherence_score


def compute_srl_and_embeddings(sentences, srl_model, sentence_model):
    srl_results = {}
    sent_vecs = {}
    
    time_srl = []
    time_embed = []
    
    for idx, sentence in enumerate(sentences):
        sentence_text = sentence.metadata.get("text")
        sent_id = sentence.metadata.get("sent_id")
        
        # SRL
        t1 = time.time()
        srl_result = srl_model(sentence_text)
        t2 = time.time()
        srl_results[sent_id] = srl_result
        
        # Sentence embedding
        t3 = time.time()
        sent_vec = sentence_model.encode(sentence_text, convert_to_numpy=True)
        t4 = time.time()
        sent_vecs[sent_id] = sent_vec
        
        time_srl.append(t2 - t1)
        time_embed.append(t4 - t3)
        
        if idx % 100 == 0:
            print(f"Processed {idx + 1} / {len(sentences)} sentences. "
                  f"Avg time: {np.mean(time_srl):.2f}s (SRL), "
                  f"{np.mean(time_embed):.2f}s (Embeddings)")
    
    return srl_results, sent_vecs

def build_dependents_index_conllu(sentence):
    deps = defaultdict(list)
    for token in sentence:
        head = token.get("head", 0)
        deps[head].append(token)
    return deps


def get_subtree_word_ids_conllu(token_id, sentence):
    """Get all word IDs in the dependency subtree rooted at token_id."""
    deps = build_dependents_index_conllu(sentence)
    ids = set()
    stack = [token_id]
    
    while stack:
        cur = stack.pop()
        if cur not in ids:
            ids.add(cur)
            for child in deps.get(cur, []):
                stack.append(child.get("id"))
    
    return ids


def shortest_tree_distance_conllu(sentence, src_id, tgt_id):
    adj = defaultdict(list)
    
    for token in sentence:
        tid = token.get("id")
        head = token.get("head", 0)
        if head and head != 0:
            adj[tid].append(head)
            adj[head].append(tid)
    
    # BFS
    queue = deque([(src_id, 0)])
    seen = {src_id}
    
    while queue:
        node, dist = queue.popleft()
        if node == tgt_id:
            return dist
        for neighbor in adj.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, dist + 1))
    
    return max(len(sentence), abs(src_id - tgt_id))


def is_animate_wordnet(lemma):
    try:
        synsets = wn.synsets(lemma, pos=wn.NOUN)
    except Exception:
        return False
    
    for synset in synsets:
        closure = set(h.name().split('.')[0] for h in synset.closure(lambda z: z.hypernyms()))
        if 'person' in closure or any('person' in str(h).lower() for h in closure):
            return True
    
    return False

def classify_voice(token, sentence):
    xpos = token.get("xpos")
    
    if xpos not in VALID_VERB_XPOS:
        return "other"
    
    feats = token.get("feats", {}) or {}
    
    if feats.get("Voice", "") == "Pass":
        return "passive"
    
    if token.get("deprel") == "aux:pass":
        return "passive"
    
    for t in sentence:
        if t.get("head") == token.get("id"):
            if t.get("deprel") in ("nsubj:pass", "csubj:pass"):
                return "passive"
    
    has_nsubj = any(
        t.get("head") == token.get("id") and t.get("deprel") == "nsubj"
        for t in sentence
    )
    has_obj = any(
        t.get("head") == token.get("id") and t.get("deprel") == "obj"
        for t in sentence
    )
    
    if has_nsubj or has_obj:
        return "active"
    
    return "other"


def extract_predicates(documents):
    predicates = []
    voice_counts = Counter()
    
    for document_id, document in enumerate(documents):
        for sentence in document:
            for token in sentence:
                if token.get("xpos") not in VALID_VERB_XPOS:
                    continue
                
                voice = classify_voice(token, sentence)
                
                token_id = token.get("id")
                if isinstance(token_id, tuple):
                    token_id = int(token_id[0])
                token["id"] = token_id
                
                pred = {
                    "token": token,
                    "sentence": sentence,
                    "document": document,
                    "document_id": document_id,
                    "voice": voice,
                }
                
                predicates.append(pred)
                voice_counts[voice] += 1
    
    return predicates, voice_counts

def extract_morphological_features(predicate):
    token = predicate["token"]
    sentence = predicate["sentence"]
    
    feats = token.get("feats", {}) or {}
    token_id = token.get("id", 0)
    
    def get_children(token_id, deprels=None):
        results = []
        for t in sentence:
            if t.get("head") == token_id:
                if deprels is None or t.get("deprel") in deprels:
                    results.append(t)
        return results
    
    def has_negation(token_id):
        for t in sentence:
            t_feats = t.get("feats", {}) or {}
            if (t.get("head") == token_id and 
                (t.get("deprel") == "neg" or t_feats.get("Polarity") == "Neg")):
                return True
        return False
    
    aux_children = get_children(token_id, deprels=["aux", "aux:pass", "cop"])
    aux_lemmas = sorted({
        (a.get("lemma") or "").lower() for a in aux_children if a.get("lemma")
    })
    
    form = token.get("form") or ""
    lower_form = form.lower()
    morph_marked = (
        bool(re.search(r"(ed|en)$", lower_form)) or
        feats.get("VerbForm") == "Part"
    )
    
    verbform = feats.get("VerbForm")
    
    return {
        "verb_form": form,
        "verb_lemma": (token.get("lemma") or "").lower(),
        "voice": predicate.get("voice"),
        "xpos": token.get("xpos", ""),
        "tense": feats.get("Tense"),
        "aspect": feats.get("Aspect"),
        "mood": feats.get("Mood"),
        "person": feats.get("Person"),
        "number": feats.get("Number"),
        "aux_present": len(aux_lemmas) > 0,
        "aux_lemmas": aux_lemmas,
        "aux_be": any(a in {"be", "am", "is", "are", "was", "were", "been", "being"} 
                     for a in aux_lemmas),
        "aux_get": any(a in {"get", "got", "getting", "gotten"} for a in aux_lemmas),
        "aux_have": any(a in {"have", "has", "had"} for a in aux_lemmas),
        "aux_modal": any(a in {"will", "would", "can", "could", "shall", "should", 
                               "may", "might", "must"} for a in aux_lemmas),
        "has_aux_pass": any(a.get("deprel") == "aux:pass" for a in aux_children),
        "verbform": verbform,
        "finite": verbform == "Fin",
        "nonfinite": verbform in {"Inf", "Part", "Ger"},
        "participial": verbform == "Part",
        "negated": has_negation(token_id),
        "morph_marked": morph_marked,
        "clitic_present": bool(re.search(r"('s|n't|'d|'ll|'re|'ve|'m)$", lower_form)),
    }

def extract_argument_properties(arg_token, sentence):
    feats = arg_token.get("feats", {}) or {}
    subtree_ids = get_subtree_word_ids_conllu(arg_token.get("id"), sentence)
    subtree_words = [w for w in sentence if w.get("id") in subtree_ids]
    
    head_lemma = (arg_token.get("lemma") or "").lower()
    upos = arg_token.get("upos")
    pron_type = feats.get("PronType")
    
    is_pronoun = (upos == "PRON") or (pron_type == "Prs")
    is_proper = (upos == "PROPN")
    
    if is_pronoun or is_proper:
        animacy = "animate"
    else:
        try:
            animacy = "animate" if head_lemma and is_animate_wordnet(head_lemma) else "unknown"
        except Exception:
            animacy = "unknown"
    
    dets = [t for t in subtree_words if t.get("deprel") == "det"]
    definite = None
    if dets:
        det_lemmas = [(d.get("lemma") or "").lower() for d in dets]
        if any(d == "the" for d in det_lemmas):
            definite = True
        elif any(d in {"a", "an"} for d in det_lemmas):
            definite = False
    
    return {
        "id": arg_token.get("id"),
        "lemma": head_lemma,
        "upos": upos,
        "number": feats.get("Number"),
        "person": feats.get("Person"),
        "is_pronoun": is_pronoun,
        "is_proper": is_proper,
        "definite": definite,
        "animacy": animacy,
        "subtree_word_ids": sorted(list(subtree_ids)),
        "linear_position": arg_token.get("id"),
    }


def check_verb_passive_evidence_conllu(vtoken, dependents_by_head):
    v_feats = vtoken.get("feats", {}) or {}
    v_dependents = dependents_by_head.get(vtoken.get("id"), [])
    
    has_nsubjpass = any((ch.get("deprel") or "") == "nsubj:pass" for ch in v_dependents)
    has_auxpass = any((ch.get("deprel") or "") == "aux:pass" for ch in v_dependents)
    voice_feat = v_feats.get("Voice") == "Pass"
    morph_part = v_feats.get("VerbForm") == "Part"
    
    return has_nsubjpass or has_auxpass or voice_feat or (morph_part and has_auxpass)


def extract_syntactic_features(predicate):
    sentence = predicate["sentence"]
    token = predicate["token"]
    word_id = token.get("id", 0)
    
    dependents_by_head = build_dependents_index_conllu(sentence)
    dependents_for_verb = dependents_by_head.get(word_id, [])
    
    subj_nodes = [t for t in dependents_for_verb 
                  if (t.get("deprel") or "").startswith("nsubj")]
    obj_children = [t for t in dependents_for_verb 
                   if (t.get("deprel") or "") in {"obj", "iobj", "obl", "dobj"}]
    clausal_children = [t for t in dependents_for_verb 
                       if (t.get("deprel") or "") in {"ccomp", "xcomp", "csubj", "csubj:pass"}]
    
    agent_nodes = [t for t in dependents_for_verb 
                  if (t.get("deprel") or "") == "obl:agent"]
    
    def has_by_preposition(obl_token):
        for child in dependents_by_head.get(obl_token.get("id"), []):
            if (child.get("deprel") or "") == "case" and (child.get("lemma") or "").lower() == "by":
                return True
        return False
    
    has_agent_by_pp = any(has_by_preposition(w) for w in dependents_for_verb 
                         if (w.get("deprel") or "") == "obl")
    
    subj_info = [extract_argument_properties(s, sentence) for s in subj_nodes]
    obj_info = [extract_argument_properties(o, sentence) for o in obj_children]
    
    subj_linear_distance = None
    subj_tree_distance = None
    order_subject_before_verb = None
    if subj_nodes:
        s = subj_nodes[0]
        subj_linear_distance = abs(word_id - s.get("id"))
        subj_tree_distance = shortest_tree_distance_conllu(sentence, word_id, s.get("id"))
        order_subject_before_verb = (s.get("id") < word_id)
    
    obj_linear_distance = None
    obj_tree_distance = None
    order_object_before_verb = None
    if obj_children:
        o = obj_children[0]
        obj_linear_distance = abs(word_id - o.get("id"))
        obj_tree_distance = shortest_tree_distance_conllu(sentence, word_id, o.get("id"))
        order_object_before_verb = (o.get("id") < word_id)
    
    order_type = None
    if subj_nodes and obj_children:
        s_pos = subj_nodes[0].get("id")
        o_pos = obj_children[0].get("id")
        v_pos = word_id
        seq = sorted([(s_pos, "S"), (v_pos, "V"), (o_pos, "O")], key=lambda x: x[0])
        order_type = "".join(ch for _, ch in seq)
    elif subj_nodes:
        order_type = "SV" if subj_nodes[0].get("id") < word_id else "VS"
    
    deprel_counts = defaultdict(int)
    for w in dependents_for_verb:
        deprel_counts[w.get("deprel")] += 1
    
    frame_parts = []
    n_obj = deprel_counts.get("obj", deprel_counts.get("dobj", 0))
    n_iobj = deprel_counts.get("iobj", 0)
    
    if n_obj > 0:
        frame_parts.append("NP")
    if n_iobj > 0:
        frame_parts.append("NP[iobj]")
    
    pp_preps = []
    for w in dependents_for_verb:
        if (w.get("deprel") or "") == "obl":
            for child in dependents_by_head.get(w.get("id"), []):
                if (child.get("deprel") or "") == "case" and child.get("lemma"):
                    pp_preps.append((child.get("lemma") or "").lower())
    if pp_preps:
        frame_parts.append("PP[" + ",".join(sorted(set(pp_preps))) + "]")
    
    if deprel_counts.get("ccomp", 0) > 0:
        frame_parts.append("ccomp")
    if deprel_counts.get("xcomp", 0) > 0:
        frame_parts.append("xcomp")
    if len(clausal_children) > 0:
        frame_parts.append("other_clause")
    
    subcat_frame = "+".join(frame_parts) if frame_parts else "intransitive_or_no_complements"
    
    token_feats = token.get("feats", {}) or {}
    verbform = token_feats.get("VerbForm")
    clause_type = "main" if token.get("head", 0) == 0 else "subordinate"
    clause_finiteness = "finite" if verbform == "Fin" else (
        "nonfinite" if verbform in {"Inf", "Part", "Ger"} else "unknown"
    )
    
    is_relcl = (
        (token.get("deprel") == "acl:relcl") or 
        any(t.get("deprel") == "acl:relcl" for t in dependents_for_verb) or 
        (token.get("deprel") == "acl")
    )
    
    is_participial_clause = (
        verbform == "Part" and 
        ((token.get("deprel") or "") in {"acl", "acl:relcl", "xcomp", "advcl"} or 
         any(ch.get("deprel") == "acl" for ch in dependents_for_verb))
    )
    
    is_conjunct = (token.get("deprel") == "conj")
    conj_children = [w for w in dependents_for_verb if (w.get("deprel") or "") == "conj"]
    
    conj_siblings = []
    if is_conjunct:
        head_conj_id = token.get("head")
        conj_siblings = [w for w in dependents_by_head.get(head_conj_id, []) 
                        if (w.get("deprel") or "") == "conj" and w.get("id") != word_id]
        head_word = next((w for w in sentence if w.get("id") == head_conj_id), None)
        if head_word is not None:
            conj_siblings = [head_word] + conj_siblings
    
    conj_passive_evidence = None
    harmony = None
    if is_conjunct or conj_children:
        siblings = []
        if is_conjunct:
            headword = next((w for w in sentence if w.get("id") == token.get("head")), None)
            if headword:
                siblings.append(headword)
            siblings.extend(conj_siblings)
        siblings.extend(conj_children)
        
        seen = {}
        for w in siblings:
            if w is not None:
                seen[w.get("id")] = w
        siblings = list(seen.values())
        
        conj_passive_evidence = {
            w.get("id"): check_verb_passive_evidence_conllu(w, dependents_by_head) 
            for w in siblings
        }
        
        this_passive_evidence = check_verb_passive_evidence_conllu(token, dependents_by_head)
        sibling_values = list(conj_passive_evidence.values())
        if sibling_values:
            harmony = "harmonic" if all(val == this_passive_evidence for val in sibling_values) else "mixed"
    
    return {
        "has_nsubj": any(t.get("deprel") == "nsubj" for t in subj_nodes),
        "has_nsubj_pass": any(t.get("deprel") == "nsubj:pass" for t in subj_nodes),
        "n_subj_nodes": len(subj_nodes),
        "subject_info": subj_info,
        "n_objs": len([w for w in obj_children if (w.get("deprel") or "") in {"obj", "dobj"}]),
        "n_iobjs": len([w for w in obj_children if (w.get("deprel") or "") == "iobj"]),
        "obj_info": obj_info,
        "n_clausal_complements": len(clausal_children),
        "has_agent_obl_agent": len(agent_nodes) > 0,
        "has_agent_by_pp": has_agent_by_pp,
        "has_agent_phrase": len(agent_nodes) > 0 or has_agent_by_pp,
        "subj_linear_distance": subj_linear_distance,
        "subj_tree_distance": subj_tree_distance,
        "order_subject_before_verb": order_subject_before_verb,
        "obj_linear_distance": obj_linear_distance,
        "obj_tree_distance": obj_tree_distance,
        "order_object_before_verb": order_object_before_verb,
        "order_type": order_type,
        "deprel_counts": dict(deprel_counts),
        "n_subj": deprel_counts.get("nsubj", 0),
        "n_subj_pass": deprel_counts.get("nsubj:pass", 0),
        "n_obj": n_obj,
        "n_iobj": n_iobj,
        "n_obl": deprel_counts.get("obl", 0),
        "n_advmod": deprel_counts.get("advmod", 0),
        "n_compound": deprel_counts.get("compound", 0),
        "n_xcomp": deprel_counts.get("xcomp", 0),
        "n_ccomp": deprel_counts.get("ccomp", 0),
        "n_csubj": deprel_counts.get("csubj", 0),
        "subcat_frame": subcat_frame,
        "clause_type": clause_type,
        "clause_finiteness": clause_finiteness,
        "is_relcl_or_acl": is_relcl,
        "is_participial_clause": is_participial_clause,
        "is_conjunct": is_conjunct,
        "conj_children_ids": [w.get("id") for w in conj_children],
        "conj_sibling_ids": [w.get("id") for w in conj_siblings],
        "conj_passive_evidence": conj_passive_evidence,
        "this_passive_evidence": check_verb_passive_evidence_conllu(token, dependents_by_head),
        "coordination_harmony": harmony,
    }

def extract_lexical_semantic_features(predicate, corpus_stats, srl_results, sent_vecs):
    token = predicate["token"]
    sentence = predicate["sentence"]
    
    sentence_id = sentence.metadata.get("sent_id")
    lemma = token.get("lemma").lower()
    token_id = token.get("id")
    
    features = {}
    
    stats = corpus_stats[lemma]
    features["lemma_freq"] = stats.get("freq")
    features["lemma_freq_per_million"] = stats.get("freq_per_million")
    features["lemma_passive_rate"] = stats.get("passive_rate", np.nan)
    features["lemma_transitivity"] = stats.get("transitivity", np.nan)
    
    verb_classes = set()
    for syn in wn.synsets(lemma, pos=wn.VERB):
        verb_classes.add(syn.lexname())
    features["verb_semantic_classes"] = list(verb_classes)
    
    try:
        features["verbnet_classes"] = vn.classids(lemma)
    except Exception:
        features["verbnet_classes"] = []
    
    features["is_stative"] = lemma in STATIVE_VERBS
    
    srl_result = srl_results.get(sentence_id, {"srl": []})
    features["srl_predicate"] = None
    features["has_ARG0"] = False
    features["has_ARG1"] = False
    
    for frame in srl_result.get("srl", []):
        pred_token = None
        for entry in frame:
            span_text, role, start, end = entry
            if role == "PRED":
                pred_token = span_text.lower()
                break
        
        if pred_token and lemma in pred_token:
            features["srl_predicate"] = pred_token
            for entry in frame:
                span_text, role, start, end = entry
                if role == "ARG0":
                    features["has_ARG0"] = True
                if role == "ARG1":
                    features["has_ARG1"] = True
            break
    
    subj_tokens = [t for t in sentence 
                  if t.get("deprel") in ("nsubj", "nsubj:pass") and t.get("head") == token_id]
    obj_tokens = [t for t in sentence 
                 if t.get("deprel") in ("obj", "iobj", "dobj") and t.get("head") == token_id]
    
    def get_simple_arg_features(tok):
        is_pron = tok.get("upos") == "PRON"
        is_proper = tok.get("upos") == "PROPN"
        
        deps = build_dependents_index_conllu(sentence)
        subtree_ids = get_subtree_word_ids_conllu(tok.get("id"), sentence)
        dets = [t for t in sentence if t.get("head") in subtree_ids and t.get("deprel") == "det"]
        definite = any(((d.get("lemma") or "").lower() == "the") for d in dets) if dets else None
        
        head_lemma = (tok.get("lemma") or "").lower()
        if is_pron or is_proper:
            animate = True
        else:
            try:
                animate = is_animate_wordnet(head_lemma) if head_lemma else False
            except Exception:
                animate = None
        
        return {"is_pronoun": is_pron, "is_proper_noun": is_proper, 
                "definite": definite, "animate": animate}
    
    if subj_tokens:
        features.update({f"subj_{k}": v for k, v in get_simple_arg_features(subj_tokens[0]).items()})
    else:
        features.update({f"subj_{k}": np.nan for k in ["is_pronoun", "is_proper_noun", "definite", "animate"]})
    
    if obj_tokens:
        features.update({f"obj_{k}": v for k, v in get_simple_arg_features(obj_tokens[0]).items()})
    else:
        features.update({f"obj_{k}": np.nan for k in ["is_pronoun", "is_proper_noun", "definite", "animate"]})
    
    sent_vec = sent_vecs.get(sentence_id)
    features["verb_embedding"] = sent_vec.tolist() if sent_vec is not None else []
    
    window = 3
    idx = token_id - 1
    left = sentence[max(0, idx - window):idx]
    right = sentence[idx + 1: idx + 1 + window]
    collocates = [t["lemma"].lower() for t in left + right if t.get("lemma")]
    features["collocates"] = collocates
    features["has_by_in_window"] = "by" in collocates
    
    prep_lemmas = []
    for t in sentence:
        if t.get("head") == token_id and t.get("deprel", "").startswith("obl"):
            for c in sentence:
                if c.get("head") == t["id"] and c.get("deprel") == "case":
                    if c.get("lemma"):
                        prep_lemmas.append(c["lemma"].lower())
    features["prep_complements"] = prep_lemmas
    
    features["has_hedge_marker"] = any(w in HEDGE_MARKERS for w in collocates)
    features["genre"] = predicate.get("genre", np.nan)
    features["formality_score"] = np.nan
    
    return features

def extract_discourse_features(predicate, sentences, doc_texts, vectorizer, lda_model, coref_output):
    token = predicate["token"]
    sentence = predicate["sentence"]
    doc_id = predicate["document_id"]
    doc_text = doc_texts[doc_id]
    
    features = {}
    
    sent_idx = next((i for i, s in enumerate(sentences) if s is sentence), 0)
    features["sentence_index"] = sent_idx
    features["paragraph_index"] = sent_idx // 5
    
    total_sentences = len(sentences)
    total_paragraphs = max(1, (len(sentences) + 4) // 5)
    
    features["sentence_pos_normalized"] = sent_idx / max(1, total_sentences - 1)
    features["paragraph_pos_normalized"] = features["paragraph_index"] / max(1, total_paragraphs - 1)
    
    if features["sentence_pos_normalized"] < 0.33:
        features["sentence_position_cat"] = "beginning"
    elif features["sentence_pos_normalized"] < 0.66:
        features["sentence_position_cat"] = "middle"
    else:
        features["sentence_position_cat"] = "end"
    
    token_text = token.get("form", "")
    features["subject_coref_count"] = 0
    features["agent_coref_recent"] = False
    
    for cluster in coref_output[doc_id].get_clusters():
        for mention in cluster:
            mention_text = str(mention).lower()
            if token_text.lower() in mention_text:
                features["subject_coref_count"] += 1
                mention_start = doc_text.lower().find(mention_text)
                sentence_start = sum(len(" ".join(t.get("form", "") for t in s)) + 1 
                                    for s in sentences[:sent_idx])
                if mention_start < sentence_start:
                    features["agent_coref_recent"] = True
    
    features["subject_given"] = features["subject_coref_count"] > 0
    
    sentence_text = sentence.metadata.get("text")
    features["reporting_context"] = any(rv in sentence_text.lower() for rv in REPORTING_VERBS)
    features["inside_quotes"] = bool(re.search(r"[\"“”‘’']", sentence_text))
    
    pos_tags = [t.get("upos", "") for t in sentence]
    content_words = sum(1 for pos in pos_tags if pos in {"NOUN", "PROPN", "ADJ", "ADP"})
    features["formality_score"] = content_words / max(1, len(sentence))
    
    sentence_bow = vectorizer.transform([sentence_text])
    sentence_corpus = matutils.Sparse2Corpus(sentence_bow, documents_columns=False)
    sentence_bow_gensim = next(iter(sentence_corpus))
    
    topic_dist = lda_model.get_document_topics(sentence_bow_gensim, minimum_probability=0)
    topic_vec = np.array([prob for _, prob in topic_dist])
    features["topic_vector"] = topic_vec

    # used to avoid 0 errors in log
    eps = 1e-12
    entropy = -np.sum(topic_vec * np.log(topic_vec + eps))
    features["topic_entropy"] = float(entropy)
    features["topic_max_prob"] = float(np.max(topic_vec) if topic_vec.size else np.nan)
    
    features["genre"] = None
    features["discourse_relation"] = None
    
    return features

def extract_all_features(predicate, corpus_stats, srl_results, sent_vecs, 
                        sentences, doc_texts, vectorizer, lda_model, coref_output):
    features = {}
    
    features.update(extract_morphological_features(predicate))
    features.update(extract_syntactic_features(predicate))
    features.update(extract_lexical_semantic_features(predicate, corpus_stats, srl_results, sent_vecs))
    features.update(extract_discourse_features(predicate, sentences, doc_texts, 
                                              vectorizer, lda_model, coref_output))
    
    return features


def run_feature_extraction(predicates, corpus_stats, srl_results, sent_vecs,
                          sentences, doc_texts, vectorizer, lda_model, coref_output):
    all_features = []
    
    for idx, predicate in enumerate(predicates):
        features = extract_all_features(
            predicate, corpus_stats, srl_results, sent_vecs,
            sentences, doc_texts, vectorizer, lda_model, coref_output
        )
        
        if idx % 5000 == 0:
            voice = predicate.get("voice")
            print(f"Processed predicate {idx+1}/{len(predicates)}: voice={voice}")
        
        all_features.append(features)
    
    return pd.DataFrame(all_features)

def _safe_colname(text):
    if text is None:
        return "none"
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(text))
    name = re.sub(r"_+", "_", name).strip("_")
    return name.lower() or "none"


def _expand_list_one_hot(df, col, prefix):
    if col not in df.columns:
        return df

    series = df[col].apply(
        lambda x: x if isinstance(x, (list, tuple, set, np.ndarray)) else ([] if pd.isna(x) else [x])
    )

    exploded = series.explode()
    if exploded.shape[0] == 0:
        return df.drop(columns=[col])

    try:
        dummies = pd.get_dummies(exploded, prefix='', prefix_sep='')
    except TypeError:
        exploded = exploded.astype(str)
        dummies = pd.get_dummies(exploded, prefix='', prefix_sep='')

    wide = dummies.groupby(level=0).max().reindex(df.index, fill_value=0)

    new_cols = {c: f"{prefix}{_safe_colname(c)}" for c in wide.columns}
    wide.rename(columns=new_cols, inplace=True)
    try:
        wide = wide.astype('int8')
    except Exception:
        wide = wide.astype(int)

    df_out = pd.concat([df.drop(columns=[col]), wide], axis=1)
    return df_out


def prepare_correlation_ready_features(df):
    df = df.copy()

    df = df[df["voice"].isin(["active", "passive"])].reset_index(drop=True)
    df["is_passive"] = (df["voice"] == "passive").astype(int)

    if "verb_lemma" in df.columns:
        grp = df.groupby("verb_lemma")
        total = grp["is_passive"].transform("size")
        passive_sum = grp["is_passive"].transform("sum")
        df["lemma_passive_rate_loo"] = np.where(
            total > 1,
            (passive_sum - df["is_passive"]) / (total - 1),
            np.nan,
        )

    if "lemma_passive_rate" in df.columns:
        df = df.drop(columns=["lemma_passive_rate"]) 
    if "topic_vector" in df.columns:
        df = df.drop(columns=["topic_vector"])  # too high-dimensional for correlation

    categoricals = [
        "tense", "aspect", "mood", "person", "number",
        "verbform", "clause_type", "clause_finiteness", "order_type",
        "coordination_harmony", "sentence_position_cat", "subcat_frame"
    ]
    present_cats = [c for c in categoricals if c in df.columns]
    if present_cats:
        df = pd.get_dummies(df, columns=present_cats, prefix=present_cats, dummy_na=True)

    if "verb_semantic_classes" in df.columns:
        df = _expand_list_one_hot(df, "verb_semantic_classes", prefix="wncls_")
    if "verbnet_classes" in df.columns:
        df = _expand_list_one_hot(df, "verbnet_classes", prefix="vncls_")

    drop_cols = [
        "verb_form", "verb_lemma", "xpos", "aux_lemmas", "deprel_counts",
        "subject_info", "obj_info", "conj_children_ids", "conj_sibling_ids",
        "conj_passive_evidence", "srl_predicate", "verb_embedding", "collocates",
        "prep_complements", "genre", "discourse_relation"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    tri_bool_cols = []
    for c in df.columns:
        if df[c].dtype == object:
            non_na = df[c].dropna()
            if non_na.map(lambda x: isinstance(x, (list, tuple, set, dict, np.ndarray))).any():
                continue
            try:
                uniques = set(non_na.unique().tolist())
            except TypeError:
                continue
            if uniques.issubset({True, False}):
                tri_bool_cols.append(c)
    for c in tri_bool_cols:
        df[c] = df[c].map({True: 1, False: 0})

    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    return df

def analyze_lda_topics(doc_texts, stop_words, num_topics_list=[5, 10, 15, 20, 25, 30]):
    tokenized_texts = [simple_preprocess(doc) for doc in doc_texts]
    id2word = corpora.Dictionary(tokenized_texts)
    gensim_corpus = [id2word.doc2bow(text) for text in tokenized_texts]
    
    results = []
    
    for num_topics in num_topics_list:
        print(f"\n{'='*60}")
        print(f"Testing LDA with {num_topics} topics")
        print(f"{'='*60}")
        
        vectorizer, lda_model, coherence_model, coherence_score = load_lda_model(
            doc_texts, stop_words, num_topics=num_topics
        )
        
        top_topics = lda_model.top_topics(
            corpus=gensim_corpus,
            texts=tokenized_texts,
            dictionary=id2word,
            coherence='c_v'
        )
        
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        print(f'Average topic coherence: {avg_topic_coherence:.4f}')
        
        for topic_idx, topic in enumerate(top_topics):
            print(f"\nTopic {topic_idx}:")
            for weight, term in topic[0][:5]:
                print(f"  {term}: {float(weight):.4f}")
            print(f"  Coherence: {float(topic[1]):.4f}")
        
        print(f"\nOverall Coherence Score: {float(avg_topic_coherence):.4f}")
        
        results.append({
            'num_topics': num_topics,
            'coherence': avg_topic_coherence,
            'vectorizer': vectorizer,
            'lda_model': lda_model
        })
    
    return results


def extract_topic_vectors_for_all_models(predicates, doc_texts, stop_words, 
                                        sentences, coref_output, num_topics_list=[5, 10, 15, 20, 25, 30]):
    for num_topics in num_topics_list:
        print(f"\n{'='*60}")
        print(f"Extracting topic vectors with {num_topics} topics")
        print(f"{'='*60}")
        
        vectorizer, lda_model, _, _ = load_lda_model(
            doc_texts, stop_words, num_topics=num_topics
        )
        
        topic_vectors = []
        
        for predicate in predicates:
            discourse_features = extract_discourse_features(
                predicate, sentences, doc_texts, vectorizer, lda_model, coref_output
            )
            topic_vectors.append(discourse_features['topic_vector'])
        
        topic_vectors_array = np.array(topic_vectors)
        topic_vectors_df = pd.DataFrame(
            topic_vectors_array,
            columns=[f"topic_{i}" for i in range(topic_vectors_array.shape[1])]
        )
        
        output_path = f"predicate_topic_vectors_{num_topics}_topics.csv"
        topic_vectors_df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")
        
        visualize_topic_distribution(topic_vectors_array, num_topics)


def visualize_topic_distribution(topic_probs, num_topics):
    time_steps = topic_probs.shape[0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(time_steps)
    
    polys = ax.stackplot(
        x,
        topic_probs.T,
        labels=[f"Topic {i}" for i in range(num_topics)],
        alpha=0.8
    )
    
    colors = [p.get_facecolor()[0] for p in polys]
    hsv_colors = [mcolors.rgb_to_hsv(mcolors.to_rgb(c)) for c in colors]
    sorted_indices = np.argsort([h[0] for h in hsv_colors])
    
    sorted_labels = [f"Topic {i}" for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    
    legend_handles = [
        plt.Line2D([0], [0], color=sorted_colors[i], lw=8, label=sorted_labels[i])
        for i in range(num_topics)
    ]
    ax.legend(handles=legend_handles, loc="upper right", title="Topics (Hue Sorted)")
    
    ax.set_xlabel("Time step")
    ax.set_ylabel("Probability")
    ax.set_title("Topic Distributions Over Time")
    
    output_path = f"topic_distributions_over_time_{num_topics}_topics.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()

def main():
    print("VOICE FEATURE EXTRACTION PIPELINE")

    print("\nLoading corpus data...")
    max_sentences = None if MAX_SENTENCES == -1 else MAX_SENTENCES
    if max_sentences is not None:
        print(f"  Limiting to the first {max_sentences} sentences across all files")
    conllu_paths = get_conllu_paths()
    sentences = list(load_corpus_sentences(conllu_paths, max_sentences=max_sentences))
    documents = list(load_corpus_documents(conllu_paths, max_sentences=max_sentences))
    doc_texts = convert_documents_to_text(documents)

    print(f"  Loaded {len(sentences)} sentences")
    print(f"  Loaded {len(documents)} documents")
    print(f"  Total tokens: {sum(len(s) for s in sentences)}")

    print("\nComputing corpus statistics...")
    corpus_stats = compute_lemma_statistics(sentences)
    stop_words = compute_stopwords()
    print(f"  Computed statistics for {len(corpus_stats)} verb lemmas")

    print("\nLoading NLP models...")
    srl_model, sentence_model = load_nlp_models()
    coref_model, coref_output = load_coreference_model(doc_texts)
    vectorizer, lda_model, coherence_model, coherence_score = load_lda_model(
        doc_texts, stop_words
    )
    print(f"  LDA coherence score: {coherence_score:.4f}")

    print("\nComputing SRL and sentence embeddings...")
    srl_results, sent_vecs = compute_srl_and_embeddings(sentences, srl_model, sentence_model)

    print("\nExtracting predicates...")
    predicates, voice_counts = extract_predicates(documents)
    print(f"  Found {len(predicates)} predicates")
    print(f"  Voice distribution: {dict(voice_counts)}")

    print("\nExtracting features...")
    features_df = run_feature_extraction(
        predicates, corpus_stats, srl_results, sent_vecs,
        sentences, doc_texts, vectorizer, lda_model, coref_output
    )

    print("\nSaving main feature set...")
    output_path = "predicate_features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"  Saved {output_path}")
    print(f"  Shape: {features_df.shape}")

    print("\nPreparing correlation-ready features...")
    features_corr_df = prepare_correlation_ready_features(features_df)
    output_corr_path = "predicate_features_correlation_ready.csv"
    features_corr_df.to_csv(output_corr_path, index=False)
    print(f"  Saved {output_corr_path}")
    print(f"  Shape: {features_corr_df.shape}")

    print("\nAnalyzing LDA topic models...")
    lda_results = analyze_lda_topics(doc_texts, stop_words)

    print("\nExtracting topic vectors for all LDA models...")
    extract_topic_vectors_for_all_models(
        predicates, doc_texts, stop_words, sentences, coref_output
    )

    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass
    main()