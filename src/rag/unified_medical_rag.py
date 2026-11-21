import json
import re
import requests
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen3.5:2b-q4_K_M"
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

WHO_DIR = ROOT / "data/processed/rag"
MEDQUAD_DIR = ROOT / "data/processed/medquad_v4/retrieval"
DAILYMED_DIR = ROOT / "data/processed/dailymed/retrieval"

DOMAIN_TERMS = {
    "health", "medical", "medicine", "disease", "disorder",
    "syndrome", "symptom", "symptoms", "treatment", "therapy",
    "drug", "medication", "diagnosis", "diagnostic", "doctor",
    "hospital", "patient", "clinical", "healthcare", "prevention",
    "heart", "blood", "brain", "lung", "lungs", "kidney",
    "kidneys", "liver", "skin", "stomach", "pancreas",
    "nerve", "nerves", "immune", "respiratory", "cardiovascular",
    "cancer", "diabetes", "asthma", "hypertension",
    "cholesterol", "infection", "allergy", "allergic",
    "migraine", "arthritis", "anemia", "stroke",
    "epilepsy", "depression", "anxiety", "obesity",
    "pregnancy", "pregnant",

    # Nutrition and metabolic health
    "nutrition", "nutritional", "diet", "dietary",
    "food", "foods", "glycemic", "glycaemic",
    "glucose", "insulin", "sugar", "carbohydrate",
    "carbohydrates", "calorie", "calories",
    "metabolism", "metabolic", "prediabetes",
    "protein", "proteins", "fat", "fats",
    "fiber", "fibre", "vitamin", "vitamins",
    "mineral", "minerals", "electrolyte",
    "electrolytes", "bloodstream",

    "fever", "pain", "cough", "coughing", "wheezing",
    "breathing", "breathlessness", "headache", "dizziness",
    "fatigue", "vomiting", "nausea", "diarrhea",
    "bleeding", "swelling", "rash", "itching",

    "hypoglycemia", "hypoglycemic",
    "hypoglycaemia", "hypoglycaemic",
    "hyperglycemia", "hyperglycemic",
    "hyperglycaemia", "hyperglycaemic",
    "hypotension", "pulmonary"
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_store(name, directory, embedding_file, metadata_file, content_file):
    embeddings = np.load(directory / embedding_file)
    metadata = read_jsonl(directory / metadata_file)
    records = read_jsonl(directory / content_file)

    if embeddings.ndim != 2:
        raise ValueError(f"{name}: embeddings must be a 2D matrix")

    if not np.isfinite(embeddings).all():
        raise ValueError(f"{name}: embeddings contain invalid values")

    if len(embeddings) != len(metadata) or len(metadata) != len(records):
        raise ValueError(f"{name}: embedding/metadata/content counts differ")

    if embeddings.shape[1] != 384:
        raise ValueError(f"{name}: unexpected embedding dimension")

    return {
        "name": name,
        "embeddings": embeddings,
        "metadata": metadata,
        "records": records
    }


def load_stores():
    who = load_store(
        "WHO",
        WHO_DIR,
        "medical_embeddings.npy",
        "embedding_metadata.jsonl",
        "medical_chunks.jsonl"
    )

    medquad = load_store(
        "MedQuAD V4",
        MEDQUAD_DIR,
        "medquad_embeddings.npy",
        "medquad_metadata.jsonl",
        "medquad_metadata.jsonl"
    )

    # DailyMed is an additional medication-label source. Keep the existing
    # WHO and MedQuAD stores unchanged; load DailyMed only when its complete
    # retrieval files are present so older checkouts still start safely.
    dailymed_files = (
        DAILYMED_DIR / "dailymed_embeddings.npy",
        DAILYMED_DIR / "dailymed_metadata.jsonl",
        DAILYMED_DIR / "dailymed_records.jsonl",
    )
    stores = [who, medquad]
    if all(path.is_file() for path in dailymed_files):
        dailymed = load_store(
            "DailyMed",
            DAILYMED_DIR,
            "dailymed_embeddings.npy",
            "dailymed_metadata.jsonl",
            "dailymed_records.jsonl"
        )
        stores.append(dailymed)
    else:
        missing = [str(path) for path in dailymed_files if not path.is_file()]
        print("WARNING: DailyMed retrieval store not loaded; missing:", ", ".join(missing))

    return stores


# Common medication names and medication-question cues. Keep this list
# conservative: it helps recognize medicine questions without treating every
# general question as medical.
MEDICATION_TERMS = {
    "metformin", "glimepiride", "glipizide", "gliclazide", "glibenclamide",
    "insulin", "sitagliptin", "linagliptin", "saxagliptin", "vildagliptin",
    "empagliflozin", "dapagliflozin", "canagliflozin", "semaglutide",
    "liraglutide", "tirzepatide", "atorvastatin", "rosuvastatin",
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen", "amoxicillin",
    "azithromycin", "omeprazole", "pantoprazole", "amlodipine", "losartan",
    "telmisartan", "ramipril", "metoprolol", "levothyroxine", "warfarin",
    "apixaban", "clopidogrel", "prednisolone", "cetirizine", "montelukast"
}

MEDICATION_QUESTION_CUES = {
    "side effect", "side effects", "adverse effect", "adverse effects",
    "drug interaction", "drug interactions", "medicine interaction",
    "medicine interactions", "interact", "interacts", "interaction",
    "interactions", "dosage", "dose", "dosages", "overdose",
    "taken together", "take together", "taking together", "can i take",
    "can they be taken", "contraindication", "contraindications",
    "prescription", "tablet", "tablets", "capsule", "capsules"
}

def is_medical_question(question):
    normalized = re.sub(r"[^a-z0-9]+", " ", question.lower()).strip()
    words = set(re.findall(r"[a-z]+", normalized))

    # Explicit health vocabulary or a recognized medication name.
    if words & DOMAIN_TERMS or words & MEDICATION_TERMS:
        return True

    # Medication-safety phrasing is medical even when the drug name is
    # unfamiliar to the local vocabulary list.
    return any(cue in normalized for cue in MEDICATION_QUESTION_CUES)


def retrieve(question, model, stores, top_k=6):
    """
    Fast hybrid retrieval:
    1. Encode the question once.
    2. Use NumPy to shortlist semantic candidates per store.
    3. Run the more expensive lexical/intent checks only on that shortlist.
    """
    # Add focused retrieval wording for narrow questions. These are search
    # expansions only: the answer generator must still use retrieved evidence.
    normalized_for_expansion = question.lower()
    retrieval_queries = [question]
    if "metformin" in normalized_for_expansion and any(
        phrase in normalized_for_expansion
        for phrase in ("side effect", "adverse effect", "adverse reaction")
    ):
        retrieval_queries.append(
            "Metformin section 6 adverse reactions common gastrointestinal "
            "effects nausea diarrhea vomiting flatulence indigestion abdominal discomfort"
        )
    if "asthma" in normalized_for_expansion and any(
        term in normalized_for_expansion
        for term in ("severe", "attack", "emergency", "warning")
    ):
        retrieval_queries.append(
            "Severe asthma attack emergency danger signs worsening breathing "
            "unable to speak full sentences blue lips confusion drowsiness "
            "exhaustion reliever inhaler not helping"
        )

    query_vectors = model.encode(
        retrieval_queries,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32, copy=False)

    def words(text):
        text = str(text).lower()
        text = re.sub(r"\bhigh\s+blood\s+pressure\b", "hypertension", text)
        text = re.sub(r"\bhypertensive\b", "hypertension", text)
        return set(re.findall(r"[a-z]+", text))

    stopwords = {
        "what", "when", "where", "which", "who", "whom", "how", "why",
        "does", "do", "is", "are", "the", "a", "an", "of", "for", "to",
        "in", "on", "and", "or", "with", "can", "could", "should", "my",
        "me", "i", "you", "it", "this", "that", "about", "tell",
        "common", "commonly", "usual", "usually"
    }

    intent_groups = {
        "symptom": {"symptom", "symptoms", "sign", "signs"},
        "cause": {"cause", "causes", "reason", "reasons"},
        "treatment": {
            "treatment", "treatments", "treat", "treated", "therapy",
            "management", "manage", "managed", "medication", "medications"
        },
        "prevention": {"prevent", "prevention"},
        "diagnosis": {"diagnosis", "diagnose", "diagnostic"},
        "risk": {"risk", "risks"},
        "complication": {"complication", "complications"},
        "test": {"test", "tests", "examination", "screening"},
        "prognosis": {"outlook", "prognosis"},
        "side_effect": {"side", "sides", "effect", "effects", "adverse", "reaction", "reactions"},
        "interaction": {"interaction", "interactions", "interact", "interacts", "together", "combine", "combined"},
        "comparison": {"difference", "differences", "compare", "compared", "comparison", "versus", "vs"},
        "medication": {"medicine", "medicines", "medication", "medications", "drug", "drugs", "tablet", "tablets", "pill", "pills", "dose", "dosage", "dosages"}
    }

    all_intent_words = set().union(*intent_groups.values())

    def get_intent(tokens):
        return {
            group for group, variants in intent_groups.items()
            if tokens & variants
        }

    query_words = words(question) - stopwords
    query_intents = get_intent(query_words)
    topic_words = query_words - all_intent_words

    # High-value anchors prevent a broad disease match from being mistaken
    # for an answer to a narrow question (for example, diabetes prevention
    # text returned for a question about fruits).
    normalized_question = question.lower()
    fruit_terms = {"fruit", "fruits", "apple", "apples", "orange", "oranges",
                   "berries", "berry", "guava", "papaya", "pear", "pears",
                   "grapes", "banana", "bananas", "mango", "mangoes"}
    medication_names = {
        name for name in ("metformin", "glimepiride", "insulin", "topiramate",
                          "sitagliptin", "empagliflozin", "glipizide")
        if name in normalized_question
    }
    asks_fruit = bool(query_words & fruit_terms) or "fruit" in normalized_question
    asks_side_effects = bool(query_intents & {"side_effect"})
    asks_interaction = bool(query_intents & {"interaction"})
    results = []

    # 250 candidates per store is a speed/recall compromise. The full
    # embedding matrix is still searched with one vectorized operation.
    semantic_candidates_per_store = 250

    for store in stores:
        matrix = np.asarray(store["embeddings"], dtype=np.float32)
        # Max similarity across original question and focused expansions.
        scores = (matrix @ query_vectors.T).max(axis=1)
        candidate_count = min(semantic_candidates_per_store, len(scores))
        if candidate_count == 0:
            continue

        if candidate_count < len(scores):
            candidate_indices = np.argpartition(
                scores, -candidate_count
            )[-candidate_count:]
        else:
            candidate_indices = np.arange(len(scores))

        candidate_indices = candidate_indices[
            np.argsort(scores[candidate_indices])[::-1]
        ]
        is_who = store["name"] == "WHO"
        is_dailymed = store["name"] == "DailyMed"

        for index in candidate_indices:
            semantic_score = float(scores[index])
            metadata = store["metadata"][int(index)]
            record = store["records"][int(index)]

            if metadata.get("needs_review", False):
                continue

            if is_who:
                content = (
                    record.get("text")
                    or metadata.get("text")
                    or record.get("page_text")
                    or ""
                )
            elif is_dailymed:
                content = (
                    record.get("text")
                    or record.get("content")
                    or record.get("answer")
                    or metadata.get("text")
                    or metadata.get("content")
                    or ""
                )
            else:
                content = metadata.get("answer") or record.get("answer") or ""

            if not isinstance(content, str) or not content.strip():
                continue

            candidate_question = str(metadata.get("question") or "").strip()
            focus = str(metadata.get("focus") or "").strip()
            drug_name = str(metadata.get("drug_name") or "").strip()
            label_title = str(metadata.get("label_title") or "").strip()
            section = str(metadata.get("section") or "").strip()

            # MedQuAD records need a meaningful question. DailyMed records are
            # label sections, so they are identified by drug/section metadata.
            if not is_who and not is_dailymed and len(words(candidate_question)) < 2:
                continue
            if is_dailymed and not (drug_name or label_title or section):
                continue

            candidate_question_words = words(candidate_question)
            focus_words = words(focus) - stopwords
            dailymed_topic_words = words(" ".join((drug_name, label_title, section)))
            candidate_intents = get_intent(candidate_question_words)
            candidate_all_text = " ".join((
                candidate_question, focus, drug_name, label_title, section, content[:5000]
            )).lower()
            candidate_all_words = words(candidate_all_text)

            # Require the requested subject, not merely the broad disease.
            if asks_fruit and not (candidate_all_words & fruit_terms):
                continue

            # Medication questions must retrieve evidence about the named
            # medicine. This blocks unrelated diabetes-label passages.
            if medication_names and not any(
                name in candidate_all_text for name in medication_names
            ):
                continue

            # For a combination question, evidence must mention BOTH drugs.
            # Otherwise the system must abstain instead of inferring safety.
            if asks_interaction and len(medication_names) >= 2:
                if not all(name in candidate_all_text for name in medication_names):
                    continue

            # A side-effect question should not be answered with boxed-warning
            # or dosage sections unless the record is actually about adverse
            # reactions / side effects.
            if asks_side_effects and is_dailymed:
                # Common side effects must come from the label's adverse-
                # reactions section, not boxed warnings or general precautions.
                section_key_for_filter = re.sub(
                    r"[^a-z0-9]+", "_", section.lower()
                ).strip("_")
                section_key_for_filter = re.sub(
                    r"^\d+(?:_\d+)*_", "", section_key_for_filter
                )
                adverse_reaction_section = section_key_for_filter in {
                    "adverse_reactions", "adverse_effects",
                    "adverse_reaction", "side_effects",
                    # DailyMed commonly stores frequency data under 6.1
                    # Clinical Studies Experience, not section 6 itself.
                    "clinical_studies_experience"
                }
                if not adverse_reaction_section:
                    continue
            if is_dailymed:
                section_intent_map = {
                    "adverse_reactions": "side_effect",
                    "adverse_reaction": "side_effect",
                    "adverse_effects": "side_effect",
                    "side_effects": "side_effect",
                    "warnings": "risk",
                    "boxed_warning": "risk",
                    "drug_interactions": "interaction",
                    "contraindications": "medication",
                    "dosage_and_administration": "medication",
                    "overdosage": "medication",
                    "indications_and_usage": "treatment",
                }
                section_key = re.sub(
                    r"[^a-z0-9]+", "_", section.lower()
                ).strip("_")

                # Remove numbered DailyMed section prefixes:
                # "7 DRUG INTERACTIONS" -> "drug_interactions"
                # "2.2 Recommended Dosage" -> "recommended_dosage"
                section_key = re.sub(r"^\d+(?:_\d+)*_", "", section_key)

                section_intent_map.update({
                    "recommended_dosage": "medication",
                    "dosage": "medication",
                    "drug_interaction": "interaction",
                    "adverse_effects": "side_effect",
                    "precautions": "risk",
                    "use_in_specific_populations": "risk",
                    "clinical_pharmacology": "medication",
                })

                mapped_intent = section_intent_map.get(section_key)
                if mapped_intent:
                    candidate_intents.add(mapped_intent)

            if (
                query_intents
                and candidate_intents
                and not query_intents.intersection(candidate_intents)
            ):
                continue

            content_words = words(content[:2500])
            candidate_topic = (
                candidate_question_words | focus_words | dailymed_topic_words | content_words
            ) - stopwords - all_intent_words

            if topic_words and not topic_words.intersection(candidate_topic):
                continue

            matched_words = query_words & (
                candidate_question_words | focus_words | dailymed_topic_words | content_words
            )
            lexical_score = len(matched_words) / len(query_words) if query_words else 0.0
            question_overlap = (
                len(query_words & candidate_question_words) / len(query_words)
                if query_words else 0.0
            )

            if is_dailymed:
                # DailyMed has structured label sections, not Q&A questions.
                # Do not penalize it for having no `question` field.
                score = 0.65 * semantic_score + 0.35 * lexical_score
            else:
                score = (
                    0.55 * semantic_score
                    + 0.20 * lexical_score
                    + 0.25 * question_overlap
                )

            if score < (0.25 if is_dailymed else 0.30):
                continue
            if is_who and score < 0.72:
                continue

            results.append({
                "score": score,
                "store": store["name"],
                "content": content,
                "metadata": metadata
            })

    results.sort(key=lambda item: item["score"], reverse=True)

    selected = []
    seen = set()
    dailymed_label_counts = {}
    dailymed_section_counts = {}

    for item in results:
        metadata = item["metadata"]

        if item["store"] == "WHO":
            identity = (
                "WHO",
                metadata.get("document_name", ""),
                metadata.get("page_number", ""),
                re.sub(r"\s+", " ", item["content"].lower()).strip()
            )
        elif item["store"] == "DailyMed":
            normalized_content = re.sub(r"\s+", " ", item["content"].lower()).strip()
            identity = (
                "DailyMed",
                str(metadata.get("setid") or metadata.get("label_title") or ""),
                re.sub(r"[^a-z0-9]+", "_", str(metadata.get("section") or "").lower()),
                normalized_content[:1200]
            )
        else:
            # Multiple MedQuAD Q&A rows can point to the same source page.
            # Keep the highest-ranked row per document/page to avoid duplicate
            # references in the UI while retaining distinct source documents.
            document_identity = (
                metadata.get("document_id")
                or metadata.get("source_url")
            )
            if document_identity:
                identity = ("MedQuAD-document", str(document_identity).strip().lower())
            else:
                normalized_question = re.sub(
                    r"[^a-z0-9 ]", "",
                    str(metadata.get("question", "")).lower()
                )
                identity = ("MedQuAD-question", normalized_question)

        if identity in seen:
            continue

        # Avoid filling the answer with near-duplicate chunks from one
        # DailyMed label or repeated copies of the same label section.
        if item["store"] == "DailyMed":
            label_id = str(metadata.get("setid") or metadata.get("label_title") or "")
            section_name = re.sub(
                r"[^a-z0-9]+", "_",
                str(metadata.get("section") or "").lower()
            ).strip("_")
            section_name = re.sub(r"^\d+(?:_\d+)*_", "", section_name)

            if dailymed_label_counts.get(label_id, 0) >= 2:
                continue
            if dailymed_section_counts.get(section_name, 0) >= 2:
                continue

            dailymed_label_counts[label_id] = dailymed_label_counts.get(label_id, 0) + 1
            dailymed_section_counts[section_name] = dailymed_section_counts.get(section_name, 0) + 1

        seen.add(identity)
        item["source_id"] = f"S{len(selected) + 1}"
        selected.append(item)

        if len(selected) >= top_k:
            break

    return selected


def get_valid_source_url(url):
    if not url:
        return None

    url = str(url).strip()

    # NIH Senior Health is a retired website.
    if "nihseniorhealth.gov" in url.lower():
        return None

    return url


def make_source(item):
    metadata = item["metadata"]

    if item["store"] == "WHO":
        return {
            "id": item["source_id"],
            "source": metadata.get("document_name", "WHO document"),
            "page": metadata.get("page_number"),
            "source_url": get_valid_source_url(metadata.get("source_url")),
            "record_id": metadata.get("record_id"),
            "source_type": "WHO document"
        }

    if item["store"] == "DailyMed":
        return {
            "id": item["source_id"],
            "source": "DailyMed",
            "drug_name": metadata.get("drug_name"),
            "label_title": metadata.get("label_title"),
            "section": metadata.get("section"),
            "published_date": metadata.get("published_date"),
            "country_scope": metadata.get("country_scope", "United States"),
            "setid": metadata.get("setid"),
            "source_url": get_valid_source_url(metadata.get("source_url")),
            "source_type": "US medication label (DailyMed; not India-specific prescribing guidance)"
        }

    return {
        "id": item["source_id"],
        "source": metadata.get("source", "MedQuAD"),
        "document_id": metadata.get("document_id"),
        "record_id": metadata.get("record_id") or metadata.get("id"),
        "original_question": metadata.get("question"),
        "focus": metadata.get("focus"),
        "source_url": get_valid_source_url(metadata.get("source_url")),
        "source_type": "Medical Q&A dataset"
    }


def answer_question(question, model, stores):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    # Handle basic conversation and assistant introduction before domain filtering.
    normalized = question.lower().strip().rstrip(" .!?")

    greeting_responses = {
        "hi": "Hello! Welcome to MedIntel-AI. I'm your medical information and learning assistant. What would you like to learn about today?",
        "hello": "Hello! Welcome to MedIntel-AI. I'm here to help you understand medical and health-related topics.",
        "hey": "Hey! Welcome to MedIntel-AI. Ask me any medical or health-related question.",
        "how are you": "I'm doing well and ready to help you learn about medical and health-related topics. What would you like to explore?",
        "good morning": "Good morning! Welcome to MedIntel-AI. What medical topic would you like to learn about today?",
        "good afternoon": "Good afternoon! Welcome to MedIntel-AI. How can I help with your medical learning today?",
        "good evening": "Good evening! Welcome to MedIntel-AI. What health-related question can I help you understand?"
    }

    capability_questions = {
        "how can you help me",
        "how do you help me",
        "what can you do",
        "what is your use",
        "what is the use of you",
        "what is your purpose",
        "who are you",
        "introduce yourself",
        "tell me about yourself",
        "what is medintel ai",
        "what is medintel-ai"
    }

    if normalized in greeting_responses or normalized in capability_questions:
        if normalized in greeting_responses:
            response = greeting_responses[normalized]
        else:
            response = (
                "I'm MedIntel-AI, an evidence-grounded medical information "
                "and learning assistant. I can help you:\n"
                "- Understand diseases such as diabetes, asthma and heart conditions.\n"
                "- Learn about medicines, their uses and documented side effects.\n"
                "- Understand symptoms, risk factors and prevention.\n"
                "- Explore nutrition, healthy lifestyle and medical concepts.\n"
                "- Clarify medical doubts using information from available "
                "medical references.\n\n"
                "Ask me a medical question or tell me which topic you want "
                "to learn. I'll explain it in clear, simple English. "
                "I provide educational information, not a personal diagnosis "
                "or replacement for a healthcare professional."
            )

        return {
            "question": question,
            "question_type": "conversation",
            "status": "answered",
            "answer": response,
            "quick_answer": response,
            "sections": [
                {
                    "heading": "MedIntel-AI",
                    "content": response,
                    "citations": []
                }
            ],
            "sources": [],
            "evidence": [],
            "safety": {}
        }

    if not is_medical_question(question):
        return {
            "question": question,
            "question_type": "out_of_domain",
            "status": "rejected",
            "quick_answer": (
                "This question is outside the medical knowledge "
                "domain of MedIntel-AI."
            ),
            "sections": [],
            "sources": [],
            "evidence": [],
            "safety": {
                "message": "Please ask a medical or health-related question."
            }
        }

    evidence = retrieve(question, model, stores, top_k=4)

    if not evidence:
        return {
            "question": question,
            "question_type": "medical",
            "status": "insufficient_evidence",
            "quick_answer": (
                "There is insufficient supporting evidence in the "
                "available medical sources to answer this question reliably."
            ),
            "sections": [],
            "sources": [],
            "evidence": [],
            "safety": {
                "message": "Consult a qualified healthcare professional."
            }
        }

    # Preserve the identity and original wording of every evidence record.
    context_parts = []

    for item in evidence:
        metadata = item["metadata"]

        original_question = metadata.get("question", "")
        focus = metadata.get("focus", "")
        record_id = (
            metadata.get("record_id")
            or metadata.get("id")
            or metadata.get("setid")
            or "Not available"
        )

        if item["store"] == "DailyMed":
            source_details = (
                f"Drug: {metadata.get('drug_name', 'Not specified')}\n"
                f"US label title: {metadata.get('label_title', 'Not specified')}\n"
                f"Label section: {metadata.get('section', 'Not specified')}\n"
                f"Published date: {metadata.get('published_date', 'Not specified')}\n"
                f"Country scope: {metadata.get('country_scope', 'United States')}\n"
                f"Source URL: {metadata.get('source_url', 'Not available')}"
            )
        else:
            source_details = (
                f"Record ID: {record_id}\n"
                f"Original question: {original_question}\n"
                f"Focus: {focus}"
            )

        context_parts.append(
            f"[{item['source_id']}]\n"
            f"Source type: {item['store']}\n"
            f"Publisher: {metadata.get('source', item['store'])}\n"
            f"{source_details}\n"
            f"Evidence text:\n{item['content'][:1100]}"
        )

    context = "\n\n--- EVIDENCE RECORD ---\n\n".join(context_parts)

    system_prompt = """
You are MedIntel-AI, an evidence-grounded medical information assistant.

Use ONLY the supplied evidence. Do not add medical facts from memory.
Do not invent doses, diagnoses, statistics, or treatment instructions.
Do not give personalized treatment advice. Never tell a patient to change
or stop prescribed medication independently.
If the evidence does not directly answer the question, use
"insufficient_evidence" and explain what is missing.
Every factual medical statement must cite one or more supplied IDs
such as S1. Never invent citation IDs. Use simple, clear English.
Do not repeat evidence or include irrelevant sections.
For questions asking for COMMON side effects, report only effects explicitly
identified as common/frequent in the evidence. Do not label boxed warnings,
rare life-threatening events, overdose effects, or monitoring risks as common
side effects. If the evidence does not identify common effects, say so.
For severe asthma attack questions, distinguish ordinary asthma symptoms from
emergency danger signs. Include a danger sign only if the supplied evidence
supports it; otherwise state that the retrieved evidence does not fully list
emergency warning signs.

Return ONLY valid JSON in this exact compact structure:
{
  "question_type": "short category",
  "status": "answered or insufficient_evidence",
  "quick_answer": "direct answer in 1-3 sentences",
  "sections": [
    {
      "heading": "short heading",
      "content": "concise explanation",
      "citations": ["S1"]
    }
  ],
  "evidence": [
    {
      "claim": "one directly supported claim",
      "citations": ["S1"]
    }
  ],
  "safety": {
    "message": "relevant safety note",
    "uncertainty_note": "limitations or empty string"
  }
}

Use at most 2 sections and 3 evidence claims.
If evidence is insufficient, do not make unsupported claims.
For DailyMed, identify it as a US medication label, not India-specific
prescribing guidance.
"""

    import time
    generation_started = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "format": "json",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Available evidence records:\n\n{context}"
                    )
                }
            ],
            "options": {
                "temperature": 0,
                "num_ctx": 3072,
                "num_predict": 600
            }
        },
        timeout=180
    )

    response.raise_for_status()


    response_data = response.json()
    print(f"RAG + Ollama request time: {time.perf_counter() - generation_started:.2f}s")
    message = response_data.get("message") or {}
    content = message.get("content") or ""

    print("Ollama finish reason:", response_data.get("done_reason"))
    print("Ollama response length:", len(content))

    valid_ids = {item["source_id"] for item in evidence}
    sources = [make_source(item) for item in evidence]

    def fallback_response(reason):
        # Show retrieved source passages instead of inventing an answer.
        sections = []
        claims = []

        # Do not display raw, potentially irrelevant source passages as if
        # they answered the user's question. Return a clean abstention instead.

        return {
            "question": question,
            "question_type": "medical",
            "status": "insufficient_evidence",
            "quick_answer": (
                "I retrieved medical source material, but could not validate "
                "a complete answer from it. Please consult a qualified "
                "healthcare professional for medical decisions."
            ),
            "sections": sections,
            "sources": [],
            "evidence": [],
            "safety": {
                "message": "For personal medical decisions, consult a qualified healthcare professional.",
                "uncertainty_note": str(reason)[:300]
            }
        }

    # Parse the model response safely.
    try:
        generated = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        print("Invalid JSON from Ollama. Using evidence fallback.")
        return fallback_response("The model returned incomplete or invalid JSON.")

    if not isinstance(generated, dict):
        return fallback_response("The model response was not a JSON object.")

    # Some model responses use {"answer": "..."} instead of our schema.
    # Do not present an uncited answer as an evidence-grounded answer.
    required = {
        "question_type", "status", "quick_answer",
        "sections", "evidence", "safety"
    }

    if not required.issubset(generated):
        print("Ollama returned an unexpected JSON structure.")
        return fallback_response("The model response did not match the required schema.")

    if not isinstance(generated.get("quick_answer"), str):
        return fallback_response("The model did not provide a valid quick answer.")

    if generated.get("status") not in {"answered", "insufficient_evidence"}:
        return fallback_response("The model returned an invalid answer status.")

    if not isinstance(generated.get("sections"), list):
        return fallback_response("The model sections were invalid.")

    if not isinstance(generated.get("evidence"), list):
        return fallback_response("The model evidence was invalid.")

    def validate_entries(entries):
        validated = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            citations = entry.get("citations", [])
            if not isinstance(citations, list):
                continue

            valid_citations = list(dict.fromkeys(
                citation for citation in citations
                if isinstance(citation, str) and citation in valid_ids
            ))

            if not valid_citations:
                continue

            entry["citations"] = valid_citations
            validated.append(entry)

        return validated

    generated["sections"] = validate_entries(generated["sections"])
    generated["evidence"] = validate_entries(generated["evidence"])

        # Deterministic, frequency-aware handling of common adverse reactions.
    normalized_question = question.lower()
    asks_common_side_effects = (
        bool(re.search(r"\b(common|commonly|most common|frequent)\b", normalized_question))
        and bool(re.search(r"\b(side effects?|adverse effects?|reactions?)\b", normalized_question))
    )

    target_drug = None
    if re.search(r"\bmetformin\b", normalized_question):
        target_drug = "metformin"
    elif re.search(r"\bglimepiride\b", normalized_question):
        target_drug = "glimepiride"

    if asks_common_side_effects and target_drug:
        relevant_items = [
            item for item in evidence
            if item["store"] == "DailyMed"
            and str(item["metadata"].get("drug_name", "")).lower() == target_drug
        ]

        sections = []
        claims = []
        all_citations = []

        if target_drug == "metformin":
            effect_patterns = {
                "diarrhea": r"\bdiarrh(?:e|o)a\b",
                "nausea/vomiting": r"\bnausea\s*/\s*vomiting\b",
                "flatulence": r"\bflatulence\b",
                "asthenia": r"\basthenia\b",
                "indigestion": r"\bindigestion\b",
                "abdominal discomfort": r"\babdominal discomfort\b",
                "headache": r"\bheadache\b",
            }

            groups = {}

            for item in relevant_items:
                metadata = item["metadata"]
                section = str(metadata.get("section", "")).lower()

                if "clinical studies experience" not in section:
                    continue

                label_title = str(metadata.get("label_title") or "Metformin label")
                if re.search(r"extended[\s-]*release", label_title, re.I):
                    formulation = "Extended-release"
                else:
                    formulation = "Tablet label (not identified as extended-release)"

                group = groups.setdefault(formulation, {
                    "citations": [],
                    "rates": {},
                    "other_effects": []
                })

                source_id = item["source_id"]
                if source_id not in group["citations"]:
                    group["citations"].append(source_id)

                # Reconstruct the complete clinical-trial section from
                # all stored chunks belonging to this exact DailyMed label.
                # Retrieval may return a chunk beginning midway through a table.
                content_parts = []

                for daily_store in stores:
                    if daily_store["name"] != "DailyMed":
                        continue

                    for meta, record in zip(
                        daily_store["metadata"],
                        daily_store["records"]
                    ):
                        if str(meta.get("setid", "")) != str(metadata.get("setid", "")):
                            continue

                        if "clinical studies experience" not in str(
                            meta.get("section", "")
                        ).lower():
                            continue

                        chunk = (
                            record.get("text")
                            or record.get("content")
                            or record.get("answer")
                            or meta.get("text")
                            or meta.get("content")
                            or ""
                        )

                        if isinstance(chunk, str) and chunk.strip():
                            content_parts.append(chunk.strip())

                content = "\\n".join(content_parts) if content_parts else item["content"]

                # Extract actual metformin/placebo percentage pairs.
                for effect, pattern in effect_patterns.items():
                    match = re.search(
                        pattern + r"\s+(\d+(?:\.\d+)?)%\s+(\d+(?:\.\d+)?)%",
                        content,
                        re.I
                    )
                    if match:
                        group["rates"][effect] = (
                            match.group(1) + "%",
                            match.group(2) + "%"
                        )

                # Capture the separately reported 1–5% reaction list.
                additional_match = re.search(
                    r"Additionally,.*?(?:1\.0%\s*to\s*5\.0%|"
                    r"≥\s*1\.0?%?\s*to\s*≤\s*5\.0?%).*?:\s*(.*?)(?:\.|$)",
                    content,
                    re.I | re.S
                )

                if additional_match:
                    raw_list = additional_match.group(1)
                    raw_list = re.sub(r"\s+", " ", raw_list).strip(" ;")
                    if raw_list and raw_list not in group["other_effects"]:
                        group["other_effects"].append(raw_list)

            for formulation, group in groups.items():
                lines = []

                if group["rates"]:
                    lines.append(
                        "Clinical-trial rates (metformin vs placebo): "
                        + "; ".join(
                            f"{effect}: {rates[0]} vs {rates[1]}"
                            for effect, rates in group["rates"].items()
                        )
                        + "."
                    )

                for other in group["other_effects"]:
                    lines.append(
                        "Additional reactions reported in the label's "
                        "1–5% range and more often than placebo: "
                        + other + "."
                    )

                if not lines:
                    continue

                citations = group["citations"]
                all_citations.extend(citations)

                content = " ".join(lines)
                sections.append({
                    "heading": f"Metformin — {formulation}",
                    "content": content,
                    "citations": citations
                })
                claims.append({
                    "claim": f"{formulation} Metformin frequency findings: {content}",
                    "citations": citations
                })

        elif target_drug == "glimepiride":
            common_reactions = [
                "hypoglycemia", "dizziness", "asthenia", "headache", "nausea"
            ]

            for item in relevant_items:
                content = item["content"]
                if not re.search(
                    r"most common adverse reactions with glimepiride were",
                    content,
                    re.I
                ):
                    continue

                found = [
                    reaction for reaction in common_reactions
                    if re.search(r"\b" + re.escape(reaction) + r"\b", content, re.I)
                ]

                if len(found) == 5:
                    source_id = item["source_id"]
                    all_citations.append(source_id)
                    sections.append({
                        "heading": "Glimepiride — most common adverse reactions",
                        "content": (
                            "The DailyMed label explicitly identifies these "
                            "five reactions as the most common: "
                            + ", ".join(found[:-1]) + " and " + found[-1] + "."
                        ),
                        "citations": [source_id]
                    })
                    claims.append({
                        "claim": (
                            "The label identifies hypoglycemia, dizziness, "
                            "asthenia, headache and nausea as the most common "
                            "glimepiride adverse reactions."
                        ),
                        "citations": [source_id]
                    })
                    break

        if sections:
            all_citations = list(dict.fromkeys(all_citations))
            generated["status"] = "answered"
            generated["quick_answer"] = " ".join(
                section["heading"] + ": " + section["content"]
                for section in sections
            )
            generated["sections"] = sections
            generated["evidence"] = claims
            generated.setdefault("safety", {})["uncertainty_note"] = (
                "Evidence is from retrieved US DailyMed labels. Clinical-trial "
                "rates are specific to the formulation and study described "
                "in each label and may not represent rates in all patients."
            )
        else:
            generated["status"] = "insufficient_evidence"
            generated["quick_answer"] = (
                "The retrieved DailyMed evidence does not contain enough "
                "frequency-specific information to identify common adverse "
                "reactions reliably for this medicine."
            )
            generated["sections"] = []
            generated["evidence"] = []
            generated.setdefault("safety", {})["uncertainty_note"] = (
                "No suitable frequency-qualified adverse-reaction evidence "
                "was found in the retrieved passages."
            )

    # Keep the answer field populated for the frontend/API.
    if asks_common_side_effects and target_drug:
        if generated.get("quick_answer"):
            generated["answer"] = generated["quick_answer"]

# If the model did not provide properly cited claims, use retrieved
    # passages rather than returning a fabricated or unsupported answer.
    if (
        generated["status"] == "answered"
        and (not generated["sections"] or not generated["evidence"])
    ):
        return fallback_response("The generated answer did not contain valid evidence citations.")

    generated["question"] = question

    # Display only sources cited by validated answer sections or evidence.
    cited_ids = set()
    for entry in generated.get("sections", []):
        cited_ids.update(entry.get("citations", []))
    for entry in generated.get("evidence", []):
        cited_ids.update(entry.get("citations", []))

    generated["sources"] = [
        source for source in sources
        if source.get("id") in cited_ids
    ]

    if not isinstance(generated.get("safety"), dict):
        generated["safety"] = {
            "message": "Consult a qualified healthcare professional.",
            "uncertainty_note": "The answer may have limitations."
        }

    return generated


def main():
    print("\nMEDINTEL-AI — UNIFIED MEDICAL RAG")

    print("\nLoading medical evidence stores...")
    stores = load_stores()

    for store in stores:
        print(
            f"{store['name']}: {len(store['records'])} records, "
            f"{store['embeddings'].shape[1]} dimensions"
        )

    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    question = input("\nEnter your medical question: ").strip()

    result = answer_question(question, model, stores)

    print("\n" + json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
