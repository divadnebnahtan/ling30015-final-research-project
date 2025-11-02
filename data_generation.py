import pandas as pd
from conllu import parse_incr
from collections import Counter, defaultdict
from tqdm import tqdm

# uncomment if wordnet is not already downloaded
# import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def read_conllu(file_path):
    """Generator yielding sentences as lists of token dicts with normalized doc metadata."""
    current_doc_id = None
    current_genre = None

    with open(file_path, "r", encoding="utf-8") as f:
        for sentence in parse_incr(f):
            md = sentence.metadata or {}

            # Update current doc id if a header line is present
            # reset genre so it can be recomputed per document
            header_doc_id = md.get('doc_id') or md.get('document_id') or md.get('newdoc id') or md.get('newdoc')
            if header_doc_id and header_doc_id != current_doc_id:
                current_doc_id = header_doc_id
                current_genre = None

            sid = md.get('sent_id')
            if sid and '-' in sid:
                doc_from_sid = sid.rsplit('-', 1)[0]
                if (current_doc_id is None) or (doc_from_sid != current_doc_id and not header_doc_id):
                    current_doc_id = doc_from_sid
                    current_genre = None

            # determine genre for this sentence
            new_genre = md.get('genre')
            if new_genre:
                current_genre = new_genre

            if (not current_genre) and current_doc_id and current_doc_id.startswith('GUM_'):
                parts = current_doc_id.split('_')
                if len(parts) >= 2:
                    current_genre = parts[1]

            # normalise onto every sentence
            md['doc_id'] = current_doc_id or 'UNKNOWN_DOC'
            md['genre'] = current_genre or 'UNKNOWN_GENRE'
            sentence.metadata = md

            yield sentence

def label_voice(token, sentence):
    if not isinstance(token.get('id'), int):
        return 'other'
    if token.get('xpos') not in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "MD"]:
        return 'other'
    # verbform = token['feats'].get('VerbForm') if token['feats'] else None
    # if verbform in {'Inf', 'Part', 'Ger'}:
    #     return 'other'

    feats = token.get('feats') or {}
    if feats.get('Voice') == 'Pass' or token.get('deprel') == 'aux:pass':
        return 'passive'
    children = [
        t for t in sentence
        if isinstance(t.get('head'), int)
        and t['head'] == token['id']
        and isinstance(t.get('id'), int)
    ]
    for child in children:
        if child.get('deprel') == 'nsubj:pass' or child.get('deprel') == 'aux:pass':
            return 'passive'

    has_nsubj = any((c.get('deprel') == 'nsubj') for c in children)
    has_obj = any((c.get('deprel') in {'obj', 'iobj'}) for c in children)
    if has_nsubj or has_obj:
        return 'active'

    return 'other'

# Light weight lexical semantics helpers
# based on WordNet via NLTK
# returns None if WordNet not available
def _wn_pos_from_upos(upos):
    if wn is None:
        return None
    return {
        'VERB': wn.VERB,
        'NOUN': wn.NOUN,
        'ADJ': wn.ADJ,
        'ADV': wn.ADV,
    }.get(upos)

def wn_lexname(lemma, upos):
    if wn is None or not lemma or not upos:
        return None
    pos = _wn_pos_from_upos(upos)
    if not pos:
        return None
    try:
        syns = wn.synsets(lemma, pos=pos)
        return syns[0].lexname() if syns else None
    except Exception:
        return None

def wn_polysemy(lemma, upos):
    if wn is None or not lemma or not upos:
        return None
    pos = _wn_pos_from_upos(upos)
    if not pos:
        return None
    try:
        return len(wn.synsets(lemma, pos=pos))
    except Exception:
        return None

# english doesnt have a long list of pronouns
# hard-coded animate list keeps it simple for speed
_PRON_ANIMATE = {
    'i','you','he','she','we','they','who','someone','somebody','anyone','anybody','everyone','everybody'
}

def wn_is_animate(lemma, upos):
    # t/f if determinable, else None
    if not lemma:
        return None
    low = str(lemma).lower()
    if upos == 'PRON':
        return low in _PRON_ANIMATE
    if wn is None:
        return None
    try:
        syns = wn.synsets(lemma, pos=wn.NOUN)
        if not syns:
            return None
        for s in syns[:3]:  # check a few senses
            ln = s.lexname()
            if ln in ('noun.person', 'noun.animal'):
                return True
        return False
    except Exception:
        return None

def compute_document_passive_rates(sentences):
    doc_verb_counts = defaultdict(int)
    doc_passive_counts = defaultdict(int)
    for sent in sentences:
        doc_id = sent.metadata.get('doc_id', 'UNKNOWN_DOC')
        for token in sent:
                voice = label_voice(token, sent)
                if voice == 'other':
                    continue
                doc_verb_counts[doc_id] += 1
                if voice == 'passive':
                    doc_passive_counts[doc_id] += 1
    doc_passive_rate = {doc: doc_passive_counts[doc]/doc_verb_counts[doc] for doc in doc_verb_counts}
    return doc_passive_rate

def compute_genre_passive_rates(sentences):
    genre_verb_counts = defaultdict(int)
    genre_passive_counts = defaultdict(int)
    for sent in sentences:
        genre = sent.metadata.get('genre', 'UNKNOWN_GENRE')
        for token in sent:
            voice = label_voice(token, sent)
            if voice == 'other':
                continue
            genre_verb_counts[genre] += 1
            if voice == 'passive':
                genre_passive_counts[genre] += 1
    return {g: (genre_passive_counts[g]/genre_verb_counts[g]) for g in genre_verb_counts}

def compute_features(sentences, doc_passive_rate, genre_passive_rate):
    # lemma statistics
    lemma_counts = Counter()
    lemma_obj_counts = Counter()
    lemma_docs = defaultdict(set)
    lemma_particle_counts = Counter()
    lemma_tense_counts = defaultdict(Counter)
    doc_sentence_counts = Counter()
    doc_token_counts = Counter()
    for sent in sentences:
        doc_id_pre = sent.metadata.get('doc_id', 'UNKNOWN_DOC')
        doc_sentence_counts[doc_id_pre] += 1
        # count integer-id tokens only
        doc_token_counts[doc_id_pre] += sum(1 for t in sent if isinstance(t.get('id'), int))
        for token in sent:
            voice = label_voice(token, sent)
            if voice == 'other':
                continue
            lemma = token['lemma']
            lemma_counts[lemma] += 1
            children = [
                t for t in sent
                if isinstance(t.get('head'), int)
                and t['head'] == token['id']
                and isinstance(t.get('id'), int)
            ]
            has_obj = any(c['deprel'] in {'obj', 'iobj'} for c in children)
            if has_obj:
                lemma_obj_counts[lemma] += 1
            # particle count
            if any(c.get('deprel') == 'compound:prt' for c in children):
                lemma_particle_counts[lemma] += 1
            # tense counts
            feats_local = token.get('feats') or {}
            if feats_local.get('Tense'):
                lemma_tense_counts[lemma][feats_local.get('Tense')] += 1
            # doc frequency for lemma
            lemma_docs[lemma].add(doc_id_pre)

    rows = []
    prev_voice = None
    # majority voice per sentence for nearby voice features
    sent_majority_voice = []
    for s in sentences:
        finite_verbs_in_s = [t for t in s if label_voice(t, s) != 'other']
        if finite_verbs_in_s:
            voices = [label_voice(t, s) for t in finite_verbs_in_s]
            sent_majority_voice.append(Counter(voices).most_common(1)[0][0])
        else:
            sent_majority_voice.append(None)

    for sent_idx, sent in enumerate(tqdm(sentences, desc="Processing sentences")):
        sent_len = len(sent)
        doc_id = sent.metadata.get('doc_id', 'UNKNOWN_DOC')
        doc_genre = sent.metadata.get('genre', 'UNKNOWN_GENRE')

        id_to_token = {t['id']: t for t in sent if isinstance(t.get('id'), int)}
        children_map = defaultdict(list)
        for t in sent:
            if isinstance(t.get('id'), int) and isinstance(t.get('head'), int):
                children_map[t['head']].append(t)

        # sentence level cues
        forms = [t.get('form', '') for t in sent]
        forms_set = set(forms)
        sentence_has_question_mark = any(f == '?' for f in forms)
        sentence_has_exclamation = any(f == '!' for f in forms)
        sentence_has_semicolon = any(f == ';' for f in forms)
        quotes = {'"', "''", '``', "’", "‘", "“", "”", "'"}
        sentence_has_quotes = any(f in quotes for f in forms_set)
        for token in sent:
            if not isinstance(token.get('id'), int):
                continue
            if label_voice(token, sent) == 'other':
                continue
            lemma = token['lemma']
            
            # direct dependents of the verb
            children = children_map.get(token['id'], [])

            # subject info
            subj_tokens = [c for c in children if c['deprel'] in {'nsubj', 'nsubj:pass'}]
            if subj_tokens:
                subj = subj_tokens[0]
                subj_pos = subj['upostag']
            else:
                subj = None
                subj_pos = 'MISSING'

            # object info
            obj_tokens = [c for c in children if c['deprel'] in {'obj', 'iobj'}]
            obj_present = len(obj_tokens) > 0
            obj_type = 'none'
            obj_pos_type = 'MISSING'
            if obj_present:
                obj_type = obj_tokens[0]['deprel']
                obj_pos_type = obj_tokens[0]['upostag']

            # distances
            subj_distance = ((token['id'] - subj['id']) / sent_len) if subj else None
            obj_distance = ((obj_tokens[0]['id'] - token['id']) / sent_len) if obj_present else None
            verb_position = token['id'] / sent_len if sent_len else None

            # voice label
            voice_label = label_voice(token, sent)

            # feats
            feats = token['feats'] or {}
            verb_tense = feats.get('Tense')
            verb_form = feats.get('VerbForm')
            verb_form_participle = verb_form == 'Part'
            verb_aspect = feats.get('Aspect')
            verb_mood = feats.get('Mood')
            verb_voice_morph = feats.get('Voice')
            verb_number = feats.get('Number')
            verb_person = feats.get('Person')
            is_modal = (token.get('xpos') == 'MD')
            is_aux_verb = (token.get('upostag') == 'AUX')

            # lexical-semantic
            verb_lexname = wn_lexname(lemma, token.get('upostag'))
            verb_polysemy = wn_polysemy(lemma, token.get('upostag'))

            # negation
            # for some reason negation is not always marked explicitly
            # so we also check for common negation words among dependents
            negation_words = {'not', 'never', 'no', "n't"}  # common English negators
            negation = any(
                (c['deprel'] == 'neg') or (c['form'].lower() in negation_words)
                for c in children
            )

            # dep depth
            dep_depth = 0
            head_id = token['head']
            while head_id not in (0, None):
                dep_depth += 1
                head_token = id_to_token.get(head_id)
                if head_token is None:
                    break
                head_id = head_token['head']
            is_root = token.get('head') in (0, None) or token.get('deprel') == 'root'

            # subtree size and depth
            def _subtree_size(node_id):
                stack = [node_id]
                count = 0
                while stack:
                    nid = stack.pop()
                    count += 1
                    for ch in children_map.get(nid, []):
                        stack.append(ch['id'])
                return count

            def _subtree_max_depth(node_id):
                # depth of deepest descendant path starting at node
                maxd = 0
                stack = [(node_id, 0)]
                while stack:
                    nid, d = stack.pop()
                    maxd = max(maxd, d)
                    for ch in children_map.get(nid, []):
                        stack.append((ch['id'], d + 1))
                return maxd

            subtree_size = _subtree_size(token['id'])
            subtree_depth = _subtree_max_depth(token['id'])

            # clause type
            clause_type = token['deprel']
            is_relative_clause = isinstance(clause_type, str) and clause_type.startswith('acl:relcl')

            # lemma statistics
            lemma_freq = lemma_counts[lemma]
            lemma_transitivity = lemma_obj_counts[lemma] / lemma_freq if lemma_freq > 0 else 0

            # previous sentence voice
            previous_sentence_voice = prev_voice

            # auxiliaries and complementation
            aux_children = [c for c in children if c['deprel'].startswith('aux')]
            aux_count = len(aux_children)
            has_aux_pass = any(c['deprel'] == 'aux:pass' for c in aux_children) or (verb_voice_morph == 'Pass')
            aux_lemmas = [c.get('lemma', '').lower() for c in aux_children]
            has_aux_have = 'have' in aux_lemmas
            has_aux_be = 'be' in aux_lemmas
            has_aux_do = 'do' in aux_lemmas
            has_get_passive = has_aux_pass and ('get' in aux_lemmas)
            
            # aspectual cues
            # robust beyond FEATS when missing
            has_perfect = has_aux_have or (verb_aspect == 'Perf')
            has_progressive = has_aux_be or (verb_aspect == 'Prog') or (token.get('xpos') == 'VBG')
            agent_present = any(c['deprel'] in {'obl:agent', 'agent'} for c in children)
            nmod_present = any(c['deprel'].startswith('nmod') for c in children)
            obl_present = any(c['deprel'].startswith('obl') for c in children)
            advmod_present = any(c['deprel'] == 'advmod' for c in children)
            conj_present = any(c['deprel'] == 'conj' for c in children)
            cc_present = any(c['deprel'] == 'cc' for c in children)
            xcomp_present = any(c['deprel'] == 'xcomp' for c in children)
            ccomp_present = any(c['deprel'] == 'ccomp' for c in children)
            mark_present = any(c['deprel'] == 'mark' for c in children)
            particle_present = any(c['deprel'] == 'compound:prt' for c in children)
            has_expletive_subject = any(c['deprel'] == 'expl' for c in children)
            has_nsubj = any(c['deprel'] == 'nsubj' for c in children)
            has_nsubj_pass = any(c['deprel'] == 'nsubj:pass' for c in children)
            child_count = len(children)

            # preposition lexical cues on oblique/nmod dependents
            preps = set()
            for dep in children:
                if isinstance(dep.get('id'), int) and (dep.get('deprel','').startswith('obl') or dep.get('deprel','').startswith('nmod')):
                    dep_children = children_map.get(dep['id'], [])
                    for ch in dep_children:
                        if ch.get('deprel') == 'case':
                            pf = (ch.get('lemma') or ch.get('form') or '').lower()
                            if pf:
                                preps.add(pf)
            has_prep_by = 'by' in preps
            has_prep_with = 'with' in preps
            has_prep_for = 'for' in preps
            has_prep_to = 'to' in preps
            has_prep_from = 'from' in preps
            has_prep_of = 'of' in preps
            has_prep_on = 'on' in preps
            has_prep_in = 'in' in preps
            has_prep_at = 'at' in preps
            has_prep_as = 'as' in preps

            # subject object morphological features
            if subj is not None:
                s_feats = subj.get('feats') or {}
                subj_number = s_feats.get('Number')
                subj_person = s_feats.get('Person')
                subj_definite = s_feats.get('Definite')
                subj_pron_type = s_feats.get('PronType')
                subj_is_pronoun = subj.get('upostag') == 'PRON'
                subj_is_proper = subj.get('upostag') == 'PROPN'
                subject_lemma = subj.get('lemma')
                subject_lexname = wn_lexname(subject_lemma, subj.get('upostag'))
                subject_polysemy = wn_polysemy(subject_lemma, subj.get('upostag'))
                subject_is_animate_wn = wn_is_animate(subject_lemma, subj.get('upostag'))
                # subject subtree measures
                subject_subtree_size = _subtree_size(subj['id'])
                subject_subtree_depth = _subtree_max_depth(subj['id'])
            else:
                subj_number = None
                subj_person = None
                subj_definite = None
                subj_pron_type = None
                subj_is_pronoun = False
                subj_is_proper = False
                subject_lemma = None
                subject_lexname = None
                subject_polysemy = None
                subject_is_animate_wn = None
                subject_subtree_size = None
                subject_subtree_depth = None

            if obj_present:
                obj = obj_tokens[0]
                o_feats = obj.get('feats') or {}
                obj_number = o_feats.get('Number')
                obj_person = o_feats.get('Person')
                obj_definite = o_feats.get('Definite')
                obj_pron_type = o_feats.get('PronType')
                obj_is_pronoun = obj.get('upostag') == 'PRON'
                obj_is_proper = obj.get('upostag') == 'PROPN'
                object_lemma = obj.get('lemma')
                object_lexname = wn_lexname(object_lemma, obj.get('upostag'))
                object_polysemy = wn_polysemy(object_lemma, obj.get('upostag'))
                object_is_animate_wn = wn_is_animate(object_lemma, obj.get('upostag'))
                object_subtree_size = _subtree_size(obj['id'])
                object_subtree_depth = _subtree_max_depth(obj['id'])
            else:
                obj_number = None
                obj_person = None
                obj_definite = None
                obj_pron_type = None
                obj_is_pronoun = False
                obj_is_proper = False
                object_lemma = None
                object_lexname = None
                object_polysemy = None
                object_is_animate_wn = None
                object_subtree_size = None
                object_subtree_depth = None

            # argument prepositional specifics for complements
            has_xcomp_to = False
            if xcomp_present:
                for dep in children:
                    if dep.get('deprel') == 'xcomp':
                        for ch in children_map.get(dep['id'], []):
                            if ch.get('deprel') == 'mark' and (ch.get('lemma') or ch.get('form') or '').lower() == 'to':
                                has_xcomp_to = True
                                break

            has_ccomp_that = False
            if ccomp_present:
                for dep in children:
                    if dep.get('deprel') == 'ccomp':
                        for ch in children_map.get(dep['id'], []):
                            if ch.get('deprel') == 'mark' and (ch.get('lemma') or ch.get('form') or '').lower() in {'that','whether','if'}:
                                has_ccomp_that = True
                                break

            # sentence context
            # nearby token POS around the verb
            prev_upos = id_to_token.get(token['id'] - 1, {}).get('upostag') if (token['id'] - 1) in id_to_token else None
            next_upos = id_to_token.get(token['id'] + 1, {}).get('upostag') if (token['id'] + 1) in id_to_token else None

            # doc and corpus level enrichments
            lemma_doc_freq = len(lemma_docs.get(lemma, set()))
            lemma_particle_rate = (lemma_particle_counts[lemma] / lemma_counts[lemma]) if lemma_counts[lemma] else 0.0
            tense_ctr = lemma_tense_counts.get(lemma, {})
            lemma_past_rate = (tense_ctr.get('Past', 0) / lemma_counts[lemma]) if lemma_counts[lemma] else 0.0
            lemma_pres_rate = (tense_ctr.get('Pres', 0) / lemma_counts[lemma]) if lemma_counts[lemma] else 0.0

            # info status (giveness) within document up to this point
            if '___doc_mentions___' not in locals():
                ___doc_mentions___ = defaultdict(Counter)
            dm = ___doc_mentions___[doc_id]
            verb_lemma_seen_before_in_doc = dm.get(lemma, 0) > 0
            verb_lemma_mention_count_before = dm.get(lemma, 0)
            subject_seen_before_in_doc = dm.get(subject_lemma, 0) > 0 if subject_lemma else None
            subject_mention_count_before = dm.get(subject_lemma, 0) if subject_lemma else None
            object_seen_before_in_doc = dm.get(object_lemma, 0) > 0 if object_lemma else None
            object_mention_count_before = dm.get(object_lemma, 0) if object_lemma else None

            # update mention counts after measuring before
            dm[lemma] += 1
            if subject_lemma:
                dm[subject_lemma] += 1
            if object_lemma:
                dm[object_lemma] += 1

            # frame signature
            # blunt but helpful representation of argument structure
            frame_bits = []
            if has_nsubj:
                frame_bits.append('nsubj')
            if has_nsubj_pass:
                frame_bits.append('nsubj:pass')
            if obj_present:
                frame_bits.append(obj_type)
            if xcomp_present:
                frame_bits.append('xcomp')
            if ccomp_present:
                frame_bits.append('ccomp')
            if obl_present:
                frame_bits.append('obl')
            if agent_present:
                frame_bits.append('agent')
            frame_signature = '+'.join(sorted(frame_bits)) if frame_bits else 'none'

            # sentence and doc normalized index and doc sizes
            doc_sent_count = doc_sentence_counts.get(doc_id, None)
            sentence_index_normalized = (sent_idx / doc_sent_count) if doc_sent_count else None
            doc_token_count = doc_token_counts.get(doc_id, None)

            row = {
                'voice_label': voice_label,
                'lemma': lemma,
                'lemma_freq': lemma_freq,
                'lemma_transitivity': lemma_transitivity,
                'verb_tense': verb_tense,
                'verb_form': verb_form,
                'verb_form_participle': verb_form_participle,
                'verb_aspect': verb_aspect,
                'verb_mood': verb_mood,
                'verb_voice_morph': verb_voice_morph,
                'verb_number': verb_number,
                'verb_person': verb_person,
                'verb_lexname': verb_lexname,
                'verb_polysemy': verb_polysemy,
                'is_modal': is_modal,
                'is_aux_verb': is_aux_verb,
                'subj_present': subj is not None,
                'subj_pos_type': subj_pos,
                'subj_number': subj_number,
                'subj_person': subj_person,
                'subj_definite': subj_definite,
                'subj_pron_type': subj_pron_type,
                'subj_is_pronoun': subj_is_pronoun,
                'subj_is_proper': subj_is_proper,
                'subject_lemma': subject_lemma,
                'subject_lexname': subject_lexname,
                'subject_polysemy': subject_polysemy,
                'subject_is_animate_wn': subject_is_animate_wn,
                'subject_subtree_size': subject_subtree_size,
                'subject_subtree_depth': subject_subtree_depth,
                'obj_present': obj_present,
                'obj_type': obj_type,
                'obj_pos_type': obj_pos_type,
                'obj_number': obj_number,
                'obj_person': obj_person,
                'obj_definite': obj_definite,
                'obj_pron_type': obj_pron_type,
                'obj_is_pronoun': obj_is_pronoun,
                'obj_is_proper': obj_is_proper,
                'object_lemma': object_lemma,
                'object_lexname': object_lexname,
                'object_polysemy': object_polysemy,
                'object_is_animate_wn': object_is_animate_wn,
                'object_subtree_size': object_subtree_size,
                'object_subtree_depth': object_subtree_depth,
                'subj_distance': subj_distance,
                'obj_distance': obj_distance,
                'verb_position': verb_position,
                'prev_upos': prev_upos,
                'next_upos': next_upos,
                'negation': negation,
                'sentence_length': sent_len,
                'dep_depth': dep_depth,
                'is_root': is_root,
                'child_count': child_count,
                'subtree_size': subtree_size,
                'subtree_depth': subtree_depth,
                'clause_type': clause_type,
                'is_relative_clause': is_relative_clause,
                'sentence_index_in_doc': sent_idx,
                'sentence_index_in_doc_normalized': sentence_index_normalized,
                'doc_id': doc_id,
                'doc_genre': doc_genre,
                'doc_passive_rate': doc_passive_rate.get(doc_id, 0.0),
                'genre_passive_rate': genre_passive_rate.get(doc_genre, 0.0),
                'doc_token_count': doc_token_count,
                'previous_sentence_voice': previous_sentence_voice,
                'next_sentence_voice': sent_majority_voice[sent_idx + 1] if sent_idx + 1 < len(sentences) else None,
                'aux_count': aux_count,
                'has_aux_pass': has_aux_pass,
                'has_aux_have': has_aux_have,
                'has_aux_be': has_aux_be,
                'has_aux_do': has_aux_do,
                'has_get_passive': has_get_passive,
                'has_perfect': has_perfect,
                'has_progressive': has_progressive,
                'agent_present': agent_present,
                'nmod_present': nmod_present,
                'obl_present': obl_present,
                'advmod_present': advmod_present,
                'conj_present': conj_present,
                'cc_present': cc_present,
                'xcomp_present': xcomp_present,
                'ccomp_present': ccomp_present,
                'has_xcomp_to': has_xcomp_to,
                'has_ccomp_that': has_ccomp_that,
                'mark_present': mark_present,
                'particle_present': particle_present,
                'has_expletive_subject': has_expletive_subject,
                'has_nsubj': has_nsubj,
                'has_nsubj_pass': has_nsubj_pass,
                'has_prep_by': has_prep_by,
                'has_prep_with': has_prep_with,
                'has_prep_for': has_prep_for,
                'has_prep_to': has_prep_to,
                'has_prep_from': has_prep_from,
                'has_prep_of': has_prep_of,
                'has_prep_on': has_prep_on,
                'has_prep_in': has_prep_in,
                'has_prep_at': has_prep_at,
                'has_prep_as': has_prep_as,
                'sentence_has_question_mark': sentence_has_question_mark,
                'sentence_has_exclamation': sentence_has_exclamation,
                'sentence_has_semicolon': sentence_has_semicolon,
                'sentence_has_quotes': sentence_has_quotes,
                'lemma_doc_freq': lemma_doc_freq,
                'lemma_particle_rate': lemma_particle_rate,
                'lemma_past_rate': lemma_past_rate,
                'lemma_pres_rate': lemma_pres_rate,
                'verb_lemma_seen_before_in_doc': verb_lemma_seen_before_in_doc,
                'verb_lemma_mention_count_before': verb_lemma_mention_count_before,
                'subject_seen_before_in_doc': subject_seen_before_in_doc,
                'subject_mention_count_before': subject_mention_count_before,
                'object_seen_before_in_doc': object_seen_before_in_doc,
                'object_mention_count_before': object_mention_count_before,
                'frame_signature': frame_signature,
            }
            rows.append(row)

        finite_verbs_in_sent = [t for t in sent if label_voice(t, sent) != 'other']
        if finite_verbs_in_sent:
            # majority voice
            voices = [label_voice(t, sent) for t in finite_verbs_in_sent]
            prev_voice = Counter(voices).most_common(1)[0][0]

    df = pd.DataFrame(rows)
    return df

def main(conllu_path, output_csv):
    sentences = list(read_conllu(conllu_path))
    print(f"Loaded {len(sentences)} sentences")

    doc_passive_rate = compute_document_passive_rates(sentences)
    genre_passive_rate = compute_genre_passive_rates(sentences)

    df = compute_features(sentences, doc_passive_rate, genre_passive_rate)
    
    # print distribution of all features
    # for debugging purposes
    for col in df.columns:
        print(f"Column: {col}")
        print(df[col].value_counts(dropna=False))
        print("\n")
    
    df.to_csv(output_csv, index=False)
    print(f"Saved features to {output_csv}")

if __name__ == "__main__":
    conllu_path = "en_gum-ud-train.conllu"  # replace with your CoNLL-U path
    output_csv = "GUM_verb_features_full.csv"
    main(conllu_path, output_csv)
