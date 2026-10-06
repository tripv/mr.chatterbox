"""
gen_teacher_source4.py — Source 4 of the teacher model instruction-tuning pipeline.

Samples diverse passages from the Victorian pretraining corpus, then uses Claude
Haiku to extract structured metadata and generate a natural user question for each.

Sources (in priority order):
  - NCSE   : Victorian periodical prose (6 .txt files, paragraph-split)
  - BL     : British Library decade JSONL.gz files (28 K books, windowed passages)
  - Gutenberg : pretraining parquet shards

Output: instruction-tuning JSONL for the teacher model, where each record contains:
  - instruction  : the teacher task description
  - topic        : extracted subject (3–5 words)
  - content_type : expository / narrative / instructional / argumentative / descriptive
  - source       : which corpus the passage came from (NCSE / BL / Gutenberg)
  - sub_corpus   : routed subject bucket used for weighted extraction
  - passage      : verbatim Victorian text
  - pair         : [{"role":"user",...}, {"role":"assistant",...}] — usable directly as SFT data

Run:
    python gen_teacher_source4.py --count 1000
    python gen_teacher_source4.py --count 1000 --preview   # sample only, no API calls
    python gen_teacher_source4.py --count 200  --concurrency 10  # smaller test run
    python gen_teacher_source4.py --count 1000 --subcorpus-weighting
"""

import argparse
import asyncio
import csv
import gzip
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent / "parsers"))
from normalize_sft_provenance import NCSE_MAP, parse_ia_id
from passage_utils import (
    extract_passages_from_text,
    extract_from_paragraph,
    starts_cleanly,
    ends_cleanly,
    is_good_passage,
    MIN_WORDS,
    MAX_WORDS,
)
from pg_nonfiction import build_nonfiction_epubs, extract_epub_text, clean_epub_text
from ia_text_parser import process_file as ia_process_file, _iter_txt_files as ia_iter_files

# ---------------------------------------------------------------------------
# Source locations and sampling weights
# ---------------------------------------------------------------------------

NCSE_DIR          = Path("raw_book_data/NCSE Data/ncse_text")
BL_DATA_DIR       = Path("raw_book_data/BL Data")
PG_DATA_DIR       = Path("raw_book_data/PG Data")
PG_CATALOG        = Path("raw_book_data/pg_catalog.csv")
PG_VICTORIAN_LIST = Path("raw_book_data/victorian_epub_final.txt")
IA_RAW_DIR        = Path("raw_book_data/ia_raw")
EB_QA_DIR         = Path("training_data/dialogue/eb_qa/eb_qa_apr 21")
IB_SHARDS_DIR     = Path("training_data/pretraining/institutional_british_shards")
HATHI_SHARDS_DIR    = Path("training_data/pretraining/hathi_shards_dedup")
WELLCOME_SHARDS_DIR = Path("training_data/pretraining/wellcome_shards")
HATHI_META          = Path("raw_book_data/hathi_trust/Victorian_workset-trip.csv")  # htid -> title/author

OUTPUT_FILE    = Path("training_data/teacher/source4_pairs.jsonl")

# Weights for weighted sampling across source types
SOURCE_WEIGHTS = {
    "NCSE":         2.0,   # periodical prose — high Q/A density, short paragraphs
    "BL":           1.0,   # book prose — broad subject coverage
    "PG_Nonfiction": 3.0,  # catalog-filtered nonfiction: history, science, travel, etc.
    "IA":           2.0,   # IA periodicals + books — broad Victorian coverage
    #"EB":           2.0,   # Encyclopaedia Britannica — pre-formed QA, encyclopedic_factual
    "IB":           2.0,   # Institutional Books (HathiTrust) — Victorian academic library
    "HATHI":        2.0,   # HathiTrust deduped — on-theme (mind/machine) book pre-filter
    "HATHI_BROAD":  2.0,   # HathiTrust deduped — broad, unfiltered sample (largest single source)
    "WELLCOME":     2.0,   # Wellcome medical — mental physiology / mind-body
}

# ---------------------------------------------------------------------------
# Sub-corpus weighting + topical routing
# ---------------------------------------------------------------------------

DEFAULT_DISPOSITION_MAP = Path(
    "tests_and_assessments/usage data and anaylsis/use_data_analysis/"
    "cluster_disposition_map_2026-05-05.csv"
)

DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-14B-Instruct-4bit"

SOURCE_SUBCORPORA = {
    "encyclopedic_factual",
    "history_biography",
    "travel_geography",
    "science_technology",
    "conduct_propriety",
    "religion_morality",
    "political_contemporary",
    "fraught_social",
    "literary_criticism",
    "thinking_machines",
    "period_dictionary",
    "period_commerce",
    "period_communication",
}

_SUBCORPUS_ALIAS = {
    "period_telegraph": "period_communication",
    "late-period_telephone": "period_communication",
}

# Lexical (in-text) patterns — period vocabulary only. The corpus is
# 1837–1899; modern terms (`internet`, `python`, `quantum`, `aluminum`)
# will not appear and were removed. For sub-corpora whose user-side topic
# is itself anachronistic (thinking_machines, period_communication), the
# regex targets the actual Victorian source vocabulary that we want to
# pull (Babbage / Lovelace / "calculating engine" / Erewhon, etc.).
#
# `encyclopedic_factual` deliberately has no lexical pattern — that bucket
# is filled directly from the EB QA corpus rather than routed.
SUBCORPUS_PATTERNS = {
    "science_technology": [
        r"\b(darwin|huxley|lyell|tyndall|faraday|maxwell|kelvin|mendel|pasteur|"
        r"origin of species|natural selection|natural history|natural philosophy|"
        r"evolution|species|geological|geology|fossil|astronom(?:y|ical|er)|"
        r"telescope|microscope|electric(?:al|ity)?|galvanic|magnetic|magnetism|"
        r"chemistry|chemical|chemist|locomotive|steam[- ]engine|railway|"
        r"locomotion|atom(?:ic|s)?|molecul|optic|mechanic|mechanism|invention|"
        r"inventor|surgeon|surgery|medicine|anatomy|physiolog|physic|disease|"
        r"fever|cholera|consumption|vivisection|laborator|specimen|naturalist)\b",
    ],
    "political_contemporary": [
        r"\b(parliament(?:ary)?|the house of commons|the house of lords|"
        r"prime minister|cabinet|gladstone|disraeli|peel|palmerston|"
        r"whig|tory|liberal party|conservative party|chartist|chartism|"
        r"reform act|reform bill|home rule|fenian|empire|imperial|"
        r"colon(?:y|ies|ial)|revolution|democra(?:cy|tic)|monarch(?:y|ist)|"
        r"republic|communis(?:m|t)|socialis(?:m|t)|trade[s ]?union|"
        r"legislation|statute|electorate|franchise|suffrage|election|elector|"
        r"despatch|polit(?:ic|ician)|despot|tyrant|emperor|tsar|kaiser|"
        r"bonaparte|premier|opposition|hansard|prussia|russia|"
        r"germany|france|austria|ottoman|turkey|america(?:n|ns)?|"
        r"the united states|president of the)\b",
    ],
    "travel_geography": [
        r"\b(london|edinburgh|dublin|paris|oxford|cambridge|manchester|"
        r"birmingham|liverpool|york|bath|brighton|new south wales|australia|"
        r"china|peking|nanking|shanghai|japan|yokohama|africa|cape colony|"
        r"cairo|alexandria|spain|brazil|rio|bombay|calcutta|delhi|colombo|"
        r"hong[- ]kong|gibraltar|the continent|continental|gazetteer|"
        r"geograph(?:y|ical)|colon(?:y|ial)|emigrant|emigration|tour(?:ist)?|"
        r"travel(?:ler)?|voyage|excursion|journey|coast|harbour|orient|levant)\b",
    ],
    "fraught_social": [
        # Abolition / race axis (period vocabulary)
        r"\b(negro(?:es)?|race|racial|races|the coloured races|"
        r"slave(?:s|ry|holder)?|abolition(?:ist)?|emancipation|manumission|"
        r"freedman|freedmen|plantation|chattel|the negro question|the slavery question|"
        r"africans?|the black race|miscegenation|mulatto)\b",
        # Women / suffrage axis
        r"\b(the woman question|equality of the sexes|female emancipation|"
        r"womens? suffrage|votes for women|women['']s rights|"
        r"the rights of women)\b",
        # Sexuality axis (period clinical / literary vocabulary)
        r"\b(sodomy|sodomite|invert(?:s|ed)?|inversion|sapphic|sapphism|"
        r"uranian|the third sex|sexual inversion)\b",
        # Jewish question axis
        r"\b(jew(?:s|ish)?|hebrew|israelite|judaism|the jewish question|"
        r"antisemit|anti-semit)\b",
        # Irish / colonial-subjects axis
        r"\b(the irish question|home rule|fenian(?:ism)?|"
        r"the great famine|the irish famine|"
        r"native peoples|the natives|the aboriginal)\b",
    ],
    "conduct_propriety": [
        r"\b(gentleman|gentlemen|the lady|ladies|good breeding|conduct|"
        r"deportment|etiquette|attire|garment|courtship|matrimony|husband|"
        r"the wife|tea|dinner|breakfast|luncheon|claret|sherry|drawing[- ]?room|"
        r"parlou?r|servant|housekeeper|governess|nursery|propriety|impropriety|"
        r"polite(?:ness)?|table manners|hostess|chaperon(?:e)?|visiting card|"
        r"introduction|domestic|household|dress(?:ing)?|fashion(?:able)?)\b",
    ],
    "religion_morality": [
        r"\b(god|the almighty|the creator|providence|christ|jesus|saviour|"
        r"the cross|salvation|redemption|damnation|sermon|preach(?:er)?|"
        r"theolog(?:y|ical)|divinity|the church|christianity|christian|"
        r"catholic|romish|romanism|popish|popery|protestant|nonconformist|"
        r"dissenter|chapel|baptist|methodist|wesleyan|anglican|episcopal|"
        r"atheist|infidel|sceptic|free[- ]?thought|virtue|vice|moral(?:s|ity)?|"
        r"conscience|sin(?:ner|ful)?|repentance|piet(?:y|ous)|pious|grace|"
        r"divine|angel|heaven|hell|purgator|the soul|the spirit|spiritual|"
        r"prayer|psalm|hymn|scripture|bible|biblical|gospel|evangelical)\b",
    ],
    "history_biography": [
        # Specific named figures and events — strong signals even in passing.
        # Dropped: "the king", "the queen", "victoria" (standalone), "the throne",
        # "history of", "reign of", "life of" — all too common in general prose.
        r"\b(napoleon(?:ic)?|wellington|admiral nelson|sir robert peel|palmerston|"
        r"queen victoria|prince albert|prince of wales|the regency|the regent|"
        r"cromwell|marlborough|julius caesar|"
        r"french revolution|crimean war|waterloo|trafalgar|battle of |"
        r"the treaty of |the peace of |"
        r"coronation|jubilee|royal dynasty)\b",
        # Genre markers: biograph/memoir are reasonably genre-specific; drop
        # "history", "historical", "historian" (too broad in Victorian prose).
        r"\b(biograph(?:y|ical)|autobiograph(?:y|ical)|memoir)\b",
    ],
    "literary_criticism": [
        r"\b(dickens|sir walter scott|scott|austen|thackeray|george eliot|"
        r"tennyson|browning|byron|shelley|keats|wordsworth|coleridge|"
        r"shakespeare|milton|carlyle|ruskin|matthew arnold|hazlitt|charles lamb|"
        r"macaulay|hardy|stevenson|the spectator|punch|household words|"
        r"all the year round|cornhill|edinburgh review|quarterly review|"
        r"blackwood['']?s|review of|essay on|criticism|critique|"
        r"the poet|poetic|poetry|prose|romance|verse|stanza|elegy|sonnet|"
        r"drama|tragedy|comedy|periodical)\b",
    ],
    "thinking_machines": [
        r"\b(babbage|ada lovelace|countess of lovelace|byron['']?s daughter|"
        r"analytical engine|difference engine|calculating engine|"
        r"calculating machine|the automaton|automata|automatic|"
        r"mechanical philosopher|self[- ]acting|jacquard loom|jacquard|"
        r"punched cards?|signs of the times|erewhon|book of the machines|"
        r"samuel butler|machinery|the machine|the mechanical age|"
        r"spinning jenny|simulacrum|menabrea|"
        r"george boole|boole|laws of thought|logical piano|"
        r"jevons|arithmometer|arithmetical machine|computing machine|"
        r"tabulating machine|mechanical calculation|mechanical arithmetic|"
        r"mechanical reason|mechanical intellect|mechanical genius|"
        r"can machines think|thinking machine|artificial reason|"
        r"logic machine|the calculator|steam intellect)\b",
        # Cognitive / philosophy-of-mind concepts. Broadens the bucket beyond
        # calculating-engine vocabulary to the mind/consciousness register from the
        # usage-data gap. Multi-word phrases score +2 (clear the floor-of-2 threshold
        # alone); bare words score +1, so they must RECUR to tag — an incidental
        # single "consciousness" won't route, a passage about it will. The Stage-1
        # passage gate and Stage-2 pair judge remain the precision gate.
        r"\b(consciousness|self[- ]?conscious(?:ness)?|unconscious(?:ness)?|"
        r"introspection|sentience|the sentient|volition|cognition|cognitive|"
        r"percipient|human mind|mind of man|mental philosophy|mental science|"
        r"philosophy of mind|science of mind|mental physiology|physiology of mind|"
        r"psycholog(?:y|ical|ist)|association of ideas|laws of mind|"
        r"operations? of the mind|mental operations?|mental process(?:es)?|"
        r"faculty of (?:reason|thought|mind)|reasoning faculty|reflective faculty|"
        r"intellectual faculties|animal automatism|reflex action|automatic action|"
        r"seat of (?:thought|consciousness|the mind)|organ of thought|"
        r"mind and (?:body|brain|matter)|(?:body|brain) and mind|"
        r"states? of consciousness|stream of (?:thought|consciousness)|"
        r"sense[- ]perception|the immaterial|the nervous system)\b",
    ],
    "period_communication": [
        r"\b(electric telegraph|telegraph(?:ic|y)?|the cable|atlantic cable|"
        r"despatch|telegram|the wire|telephone(?:s|ic)?|edison|the post|"
        r"post office|postage|penny post|royal mail|mail coach|messenger|"
        r"signal(?:ling)?|semaphore|carrier pigeon|wireless|marconi)\b",
    ],
    "period_dictionary": [
        r"\b(dictionary|dictionaries|lexicon|lexicograph|thesaurus|"
        r"webster['']?s|johnson['']?s dictionary|roget|brande['']?s|"
        r"defin(?:e|ed|es|ition)|signif(?:y|ies|ication)|denotes|"
        r"etymolog|the term|the word)\b",
    ],
    "period_commerce": [
        r"\b(pound sterling|shilling|pence|farthing|guinea|sovereign|"
        r"halfcrown|merchant|tradesman|commerce|commercial|the exchange|"
        r"the market|interest|usury|bank(?:er|ing)|book[- ]?keep|ledger|"
        r"invoice|account[- ]book|reckoner|arithmetic|computation|tally|"
        r"bill of exchange|consols|stock exchange|joint stock|"
        r"partnership|debenture)\b",
    ],
}

# Strong-signal source allowlists. A passage gets +5 to a sub-corpus's
# score for each author/title term it matches — well above the +1 we give
# per lexical hit. This lets us tag a Babbage memoir as thinking_machines
# even when the prose itself doesn't repeat "calculating engine" every
# page, and keeps Carlyle / Ruskin / Newman / Mill etc. flowing into the
# right buckets without depending on lucky keyword matches.
SUBCORPUS_AUTHOR_TERMS = {
    "thinking_machines": [
        "babbage", "ada lovelace", "lovelace", "samuel butler",
        "edgar allan poe", "george boole", "boole", "jevons",
        "william stanley jevons", "menabrea",
        # Mind-science authors (cognitive broadening). Distinctive surnames only —
        # the router substring-matches against title+author, so short/ambiguous
        # names (bare "bain", "sully") use full forms to avoid e.g. "chamberlain".
        "alexander bain", "henry maudsley", "maudsley",
        "george henry lewes", "g. h. lewes", "lewes",
        "w. k. clifford", "clifford", "george romanes", "romanes",
        "james sully", "charles mercier", "mercier", "ribot",
    ],
    "religion_morality": [
        "carlyle", "ruskin", "newman", "manning", "spurgeon",
        "frederick denison maurice", "kierkegaard",
    ],
    "literary_criticism": [
        "matthew arnold", "thomas carlyle", "william hazlitt", "hazlitt",
        "charles lamb", "macaulay", "leslie stephen", "saintsbury",
        "george saintsbury",
    ],
    "political_contemporary": [
        "john stuart mill", "harriet martineau", "edmund burke",
        "matthew arnold", "walter bagehot",
    ],
    "history_biography": [
        "macaulay", "froude", "carlyle",
    ],
    # fraught_social is typically sparse in random corpus pulls;
    # author-keyed routing is the primary way to surface this material.
    # Authors span all axes deliberately: abolitionist / pro-empire /
    # feminist / conservative — the corpus needs multiple perspectives.
    "fraught_social": [
        # Abolition
        "wilberforce", "thomas clarkson", "clarkson", "olaudah equiano",
        "equiano", "frederick douglass", "douglass",
        "harriet beecher stowe", "beecher stowe",
        # Women / suffrage
        "mary wollstonecraft", "wollstonecraft",
        "olive schreiner", "schreiner",
        "frances power cobbe", "josephine butler",
        "lydia becker", "harriet martineau",
        "john stuart mill",  # Subjection of Women
        # Irish question
        "michael davitt", "parnell", "charles stewart parnell",
        # Sexuality (period scholarship / literature)
        "edward carpenter", "havelock ellis", "john addington symonds",
        "symonds",
        # Race (including pro-empire / conservative voices for period balance)
        "james anthony froude",  # The English in the West Indies
        "anthony trollope",      # The West Indies and the Spanish Main
    ],
}

# Title-pattern allowlists. Same +5 weight as author terms. Useful for
# work-keyed routing: a Roget thesaurus is period_dictionary regardless
# of its prose content, and Carlyle's "Sartor Resartus" hits both
# religion_morality and literary_criticism via title rather than guessing
# from text fragments.
SUBCORPUS_TITLE_PATTERNS = {
    "thinking_machines": [
        r"\berewhon\b",
        r"signs of the times",
        r"passages from the life of a philosopher",
        r"\banalytical engine\b",
        r"book of the machines",
    ],
    "period_dictionary": [
        r"\bdictionary\b", r"\blexicon\b", r"\bthesaurus\b",
        r"\bwebster\b", r"\bjohnson['']?s\b", r"\broget\b", r"\bbrande\b",
    ],
    "period_commerce": [
        r"ready[- ]reckoner",
        r"book[- ]?keeping",
        r"\barithmetic\b",
        r"merchant.{0,30}(?:handbook|guide|manual|companion)",
        r"counting[- ]house",
    ],
    "religion_morality": [
        r"\bsermon", r"\btheology", r"\bmeditations\b", r"\bapologia\b",
        r"sartor resartus", r"signs of the times",
    ],
    "literary_criticism": [
        r"essays in criticism", r"on heroes",
        r"sartor resartus",
        r"\bessay on\b", r"\bcritic(?:ism|al)?\b",
        r"\breview of\b",
    ],
    "history_biography": [
        # Title-level "history of" / "life of" etc. are reliable genre signals
        # in a book title even though they're too broad in prose.
        r"\bhistory of\b", r"\blife of\b", r"\bmemoirs of\b",
        r"\breign of\b", r"french revolution",
        r"\bannals of\b", r"\bchronicles of\b",
    ],
    "fraught_social": [
        r"negro question", r"slavery",
        r"subjection of women", r"woman question",
        r"vindication of the rights of",  # Wollstonecraft
        r"abolition", r"emancipation",
        r"the irish question", r"the jewish question",
        r"sexual inversion",  # Havelock Ellis
        r"the third sex",
        r"english in the west indies",  # Froude (pro-empire)
        r"west indies and the spanish main",  # Trollope
    ],
    "conduct_propriety": [
        r"\bmanners\b", r"\betiquette\b",
        r"young lad(?:y|ies)", r"young gentleman",
        r"household management", r"domestic economy",
        r"\bservant\b", r"\bgoverness\b",
    ],
}

_COMPILED_SUBCORPUS_PATTERNS = {
    sub: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for sub, pats in SUBCORPUS_PATTERNS.items()
}

_COMPILED_SUBCORPUS_TITLE_PATTERNS = {
    sub: [re.compile(pat, re.IGNORECASE) for pat in pats]
    for sub, pats in SUBCORPUS_TITLE_PATTERNS.items()
}


def _canonical_subcorpus(label: str) -> str:
    key = label.strip()
    if not key:
        return key
    return _SUBCORPUS_ALIAS.get(key, key)


def _author_from_title(title: str) -> str:
    """
    Best-effort author extraction from title strings like:
      "Some Book [By Jane Doe.]"
    """
    if not title:
        return "unknown"
    m = re.search(r"\[\s*By\s+([^\]]+)\]", title, flags=re.IGNORECASE)
    if not m:
        return "unknown"
    author = m.group(1).strip().strip(".;,:")
    return author or "unknown"


def derive_subcorpus_weights(
    disposition_map: Path,
    mixed_source_ratio: float,
) -> dict[str, float]:
    """
    Build normalized sub-corpus weights from cluster disposition map.

    Uses:
      - A clusters at full weight
      - F clusters at (size * mixed_source_ratio)
      - non-source labels ignored
    """
    if not disposition_map.exists():
        print(f"  WARNING: disposition map not found: {disposition_map}")
        return {}

    raw = Counter()
    with disposition_map.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disp = row.get("disposition", "").strip()
            try:
                size = float(row.get("size", "0") or 0)
            except ValueError:
                continue
            if size <= 0:
                continue
            if disp == "A":
                base_weight = size
            elif disp == "F":
                base_weight = size * mixed_source_ratio
            else:
                continue

            subs = []
            for part in (row.get("sub_corpus") or "").split("|"):
                sub = _canonical_subcorpus(part)
                if sub in SOURCE_SUBCORPORA:
                    subs.append(sub)
            if not subs:
                continue
            share = base_weight / len(subs)
            for sub in subs:
                raw[sub] += share

    total = sum(raw.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in raw.items()}


def route_passage_to_subcorpus(
    text: str,
    title: str = "",
    author: str = "",
) -> list[tuple[str, int]]:
    """
    Route a passage to one or more sub-corpora.

    Two signals contribute, with author/title getting much higher weight:
      - Author-term match against title+author metadata: +5 per term hit
      - Title-pattern match against title+author metadata: +5 per pattern hit
      - Lexical pattern match against the text body: +1 per hit

    Returns sub-corpora sorted by score descending, keeping every sub-corpus
    that scored within roughly a third of the leader. A passage may end up
    tagged with multiple sub-corpora (e.g. a Carlyle essay on machinery
    legitimately lands in thinking_machines + religion_morality + literary
    criticism). Returns [] if no signal — caller treats as "unassigned".
    """
    text_lc = text.lower()
    meta_lc = f"{title} {author}".lower()
    scores: dict[str, int] = defaultdict(int)

    for sub, terms in SUBCORPUS_AUTHOR_TERMS.items():
        for term in terms:
            if term in meta_lc:
                scores[sub] += 5

    for sub, patterns in _COMPILED_SUBCORPUS_TITLE_PATTERNS.items():
        for pat in patterns:
            if pat.search(meta_lc):
                scores[sub] += 5

    for sub, patterns in _COMPILED_SUBCORPUS_PATTERNS.items():
        for pat in patterns:
            hits = pat.findall(text_lc)
            for hit in hits:
                # Phrase matches ("the great famine", "home rule") are more
                # specific than bare single-word hits — weight them at 2 so
                # a single phrase hit clears the minimum threshold of 2.
                scores[sub] += 2 if " " in hit else 1

    if not scores:
        return []

    primary_score = max(scores.values())
    # Multi-route threshold: keep secondaries only if they're within a
    # third of the leader's score, with a floor of 2 so a single weak hit
    # can't tag a passage. Tweak if the routed mix gets too noisy.
    threshold = max(2, primary_score // 3)
    return sorted(
        ((sub, sc) for sub, sc in scores.items() if sc >= threshold),
        key=lambda kv: (-kv[1], kv[0]),
    )


def _weighted_pick_without_replacement(
    items: list[dict],
    k: int,
    rng: random.Random,
    seen: set[str],
) -> list[dict]:
    """
    Weighted sampling (by SOURCE_WEIGHTS) without replacement.
    Deduplicates by passage prefix key.
    """
    if k <= 0 or not items:
        return []
    pool = list(items)
    chosen = []
    while pool and len(chosen) < k:
        wts = [SOURCE_WEIGHTS.get(p["source"], 1.0) for p in pool]
        idx = rng.choices(range(len(pool)), weights=wts, k=1)[0]
        pick = pool.pop(idx)
        key = pick["text"][:80]
        if key in seen:
            continue
        seen.add(key)
        chosen.append(pick)
    return chosen


def _sample_with_subcorpus_targets(
    pool: list[dict],
    n: int,
    rng: random.Random,
    targets: dict[str, float],
) -> list[dict]:
    """
    Sample passages to hit sub-corpus targets while preserving source diversity.
    Falls back to pool-wide weighted sampling for any shortfall.
    """
    by_sub = defaultdict(list)
    for p in pool:
        # Index a passage under every sub-corpus it routes to, not just
        # the primary. Dedup-by-text-prefix in the picker prevents the
        # same passage from being selected twice across buckets.
        subs = p.get("sub_corpus_all") or [p.get("sub_corpus", "unassigned")]
        for sub in subs:
            by_sub[sub].append(p)

    # Allocate per-sub quotas with availability constraints.
    eligible = {
        sub: wt for sub, wt in targets.items()
        if sub in by_sub and by_sub[sub]
    }
    if not eligible:
        return []

    norm = sum(eligible.values())
    eligible = {k: v / norm for k, v in eligible.items()}
    quota = Counter()
    for _ in range(n):
        choices = [s for s in eligible if quota[s] < len(by_sub[s])]
        if not choices:
            break
        weights = [eligible[s] for s in choices]
        chosen_sub = rng.choices(choices, weights=weights, k=1)[0]
        quota[chosen_sub] += 1

    seen = set()
    selected = []
    for sub, q in quota.items():
        picked = _weighted_pick_without_replacement(by_sub[sub], q, rng, seen)
        selected.extend(picked)

    # Backfill from the full pool if any quotas were unattainable.
    if len(selected) < n:
        extra = _weighted_pick_without_replacement(pool, n - len(selected), rng, seen)
        selected.extend(extra)

    return selected

# ---------------------------------------------------------------------------
# NCSE loader — split Victorian periodical .txt files on double-newline
# ---------------------------------------------------------------------------

def load_ncse_passages(rng: random.Random, cap_per_file: int = 10_000) -> list[dict]:
    """
    Load passages from NCSE periodical text files using passage_utils extraction.

    Caps each file at cap_per_file passages before pooling, so that no single
    periodical (e.g. the 1.2 GB Northern Star) dominates the NCSE contribution.
    """
    if not NCSE_DIR.exists():
        print(f"  WARNING: {NCSE_DIR} not found — skipping NCSE")
        return []

    all_passages = []
    for txt_file in sorted(NCSE_DIR.glob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        file_passages = extract_passages_from_text(text)
        if len(file_passages) > cap_per_file:
            file_passages = rng.sample(file_passages, cap_per_file)
        ncse_key = txt_file.stem.lower().strip()
        ncse_pub, ncse_year = NCSE_MAP.get(ncse_key, (txt_file.stem, ""))
        all_passages.extend(
            {
                "text":        p,
                "source":      "NCSE",
                "title":       txt_file.stem,
                "book":        txt_file.stem,
                "author":      "unknown",
                "book_title":  ncse_pub,
                "book_year":   ncse_year,
                "prov_method": "ncse_map",
            }
            for p in file_passages
        )
        print(f"  {'NCSE/' + txt_file.name:<40} {len(file_passages):>7,} candidates")

    return all_passages


# ---------------------------------------------------------------------------
# BL loader — windowed passage extraction from decade JSONL.gz files
# ---------------------------------------------------------------------------

def load_bl_passages(rng: random.Random,
                     max_books: int = 2000,
                     max_passages: int = 60_000) -> list[dict]:
    """
    Sample up to max_books from BL decade JSONL.gz files and extract
    sentence-boundary-aligned passages using passage_utils.
    """
    if not BL_DATA_DIR.exists():
        print(f"  WARNING: {BL_DATA_DIR} not found — skipping BL")
        return []

    decade_files = sorted(BL_DATA_DIR.glob("books_*.jsonl.gz"))
    if not decade_files:
        print(f"  WARNING: no books_*.jsonl.gz in {BL_DATA_DIR} — skipping BL")
        return []

    all_passages = []
    books_per_decade = max(1, max_books // len(decade_files))

    for dfile in decade_files:
        decade_passages = []
        try:
            with gzip.open(str(dfile), "rt", encoding="utf-8", errors="replace") as f:
                books_seen = 0
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = rec.get("text", "")
                    if not text or len(text.split()) < MIN_WORDS:
                        continue
                    title = rec.get("title", "Unknown")
                    author = rec.get("author") or rec.get("creator") or _author_from_title(title)
                    author_str = str(author).strip() if author else "unknown"
                    bl_author = author_str if author_str.lower() not in ("unknown", "") else ""
                    passages = extract_passages_from_text(text)
                    decade_passages.extend(
                        {
                            "text":        p,
                            "source":      "BL",
                            "title":       title,
                            "book":        title,
                            "author":      author_str,
                            "book_title":  title,
                            "book_author": bl_author,
                            "prov_method": "raw",
                        }
                        for p in passages
                    )
                    books_seen += 1
                    if books_seen >= books_per_decade:
                        break
        except Exception as e:
            print(f"  WARNING: error reading {dfile.name}: {e}")
            continue

        print(f"  {'BL/' + dfile.name:<40} {len(decade_passages):>7,} candidates")
        all_passages.extend(decade_passages)

    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)

    return all_passages


# ---------------------------------------------------------------------------
# PG Nonfiction loader — catalog-filtered epubs, no fiction contamination
# ---------------------------------------------------------------------------

def load_pg_nonfiction_passages(rng: random.Random,
                                max_books: int = 400,
                                max_passages: int = 30_000) -> list[dict]:
    """
    Load passages from Project Gutenberg nonfiction epubs identified via pg_catalog.csv.
    Excludes fiction (LoCC classes PS/PR/PZ/PQ/PT/PG and subject-tagged novels/poetry).
    """
    if not PG_DATA_DIR.exists():
        print(f"  WARNING: {PG_DATA_DIR} not found — skipping PG Nonfiction")
        return []
    if not PG_CATALOG.exists():
        print(f"  WARNING: {PG_CATALOG} not found — skipping PG Nonfiction")
        return []

    all_epubs = build_nonfiction_epubs(PG_CATALOG, PG_VICTORIAN_LIST, PG_DATA_DIR)
    if not all_epubs:
        print("  WARNING: no nonfiction epubs found — skipping PG Nonfiction")
        return []

    # Pull author metadata from catalog for stable tagging.
    pg_meta: dict[str, dict[str, str]] = {}
    try:
        with PG_CATALOG.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pg_id = (row.get("Text#") or "").strip()
                if not pg_id:
                    continue
                pg_meta[pg_id] = {
                    "title": (row.get("Title") or "").replace("\n", " ").strip(),
                    "author": (row.get("Authors") or row.get("Author") or "").strip() or "unknown",
                }
    except Exception:
        pg_meta = {}

    sampled = rng.sample(all_epubs, min(max_books, len(all_epubs)))
    all_passages: list[dict] = []

    for epub_path, pg_id, title in sampled:
        try:
            raw = extract_epub_text(epub_path)
            text = clean_epub_text(raw)
            passages = extract_passages_from_text(text)
            meta = pg_meta.get(pg_id, {})
            book = meta.get("title") or title
            author = meta.get("author") or "unknown"
            pg_author = author if author.lower() != "unknown" else ""
            prov = "pg_catalog" if pg_id in pg_meta else "raw"
            all_passages.extend(
                {
                    "text":        p,
                    "source":      "PG_Nonfiction",
                    "title":       title,
                    "book":        book,
                    "author":      author,
                    "book_title":  book,
                    "book_author": pg_author,
                    "pg_id":       pg_id,
                    "prov_method": prov,
                }
                for p in passages
            )
        except Exception:
            continue

    print(f"  {'PG Nonfiction epubs':<40} {len(all_passages):>7,} candidates "
          f"({len(sampled)} books sampled from {len(all_epubs):,} available)")

    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)

    return all_passages


# ---------------------------------------------------------------------------
# IA loader — random sample of files, paragraph-level extraction
# ---------------------------------------------------------------------------

def load_ia_passages(rng: random.Random,
                     max_files: int = 3000,
                     max_passages: int = 40_000) -> list[dict]:
    """
    Sample up to max_files from the IA corpus and extract quality passages.

    Uses ia_text_parser's process_file() for cleaning (column-render detection,
    Google boilerplate stripping, noise filtering), then applies passage_utils
    quality gates (sentence boundaries, word count, coherence).
    """
    if not IA_RAW_DIR.exists():
        print(f"  WARNING: {IA_RAW_DIR} not found — skipping IA")
        return []

    all_files = ia_iter_files()
    if not all_files:
        print("  WARNING: no IA text files found — skipping IA")
        return []

    sampled_files = rng.sample(all_files, min(max_files, len(all_files)))
    all_passages: list[dict] = []

    for f in sampled_files:
        paragraphs, stype = ia_process_file(f)
        if stype in ("skip", "error"):
            continue
        ia_pub_title, ia_year = parse_ia_id(f.parent.name)
        for p in paragraphs:
            if is_good_passage(p) and starts_cleanly(p) and ends_cleanly(p):
                all_passages.append({
                    "text":        p,
                    "source":      "IA",
                    "title":       f.parent.name,
                    "book":        f.parent.name,
                    "author":      "unknown",
                    "book_title":  ia_pub_title,
                    "book_year":   ia_year,
                    "prov_method": "ia_parse",
                })

    print(f"  {'IA files sampled':<40} {len(all_passages):>7,} candidates "
          f"({len(sampled_files):,} files from {len(all_files):,} available)")

    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)

    return all_passages


# ---------------------------------------------------------------------------
# IB loader — Institutional Books parquet shards (HathiTrust / Google Books)
# ---------------------------------------------------------------------------

def load_ib_passages(rng: random.Random,
                     max_shards: int = 60,
                     max_passages: int = 40_000) -> list[dict]:
    """
    Sample from Institutional Books pretraining shards.

    Each shard is a parquet file with a single `text` column; each row is one
    book. No title/author metadata is available at this level — passages are
    tagged with the shard filename as a book proxy.

    The shard texts include library stamps and cataloguing headers at the top
    of each book. These fail the word-count / coherence gates in passage_utils
    and are silently dropped, so no special pre-filter is needed.
    """
    if not IB_SHARDS_DIR.exists():
        print(f"  WARNING: {IB_SHARDS_DIR} not found — skipping IB")
        return []

    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not installed — skipping IB")
        return []

    shard_files = sorted(IB_SHARDS_DIR.glob("*.parquet"))
    if not shard_files:
        print(f"  WARNING: no .parquet files in {IB_SHARDS_DIR} — skipping IB")
        return []

    sampled = rng.sample(shard_files, min(max_shards, len(shard_files)))
    all_passages: list[dict] = []
    books_seen = 0

    for shard in sampled:
        try:
            # Load barcode column if present (new shards); fall back for legacy text-only shards
            try:
                df = pd.read_parquet(shard, columns=["barcode", "text"])
                has_barcode = True
            except Exception:
                df = pd.read_parquet(shard, columns=["text"])
                has_barcode = False
        except Exception as e:
            print(f"  WARNING: error reading {shard.name}: {e}")
            continue
        for row_idx, row in enumerate(df.itertuples(index=False)):
            text = row.text if hasattr(row, "text") else ""
            if not isinstance(text, str) or len(text.split()) < MIN_WORDS:
                continue
            barcode = getattr(row, "barcode", "") if has_barcode else ""
            books_seen += 1
            passages = extract_passages_from_text(text)
            all_passages.extend(
                {
                    "text":      p,
                    "source":    "IB",
                    "title":     shard.stem,
                    "book":      shard.stem,
                    "author":    "unknown",
                    "barcode":   barcode,
                    "shard_row": row_idx,
                }
                for p in passages
            )

    print(f"  {'IB shard books':<40} {len(all_passages):>7,} candidates "
          f"({books_seen} books in {len(sampled)} shards)")

    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)

    return all_passages


# ---------------------------------------------------------------------------
# HATHI / WELLCOME loaders — deduped HathiTrust + Wellcome medical parquet shards.
# Both have columns {text, year, source, pub_place, lang, ocr_score, htid}. HATHI is
# pre-filtered to on-theme (mind / machine / consciousness) BOOKS via the workset CSV
# so a thinking_machines run isn't trawling 200k unrelated volumes; the routing,
# Stage-1 passage gate, and Stage-2 pair judge then do the fine-grained selection.
# ---------------------------------------------------------------------------

# On-theme title/author selection for HATHI books (ported from gen_thinking_machines).
_TM_TITLE_RE = re.compile(
    r"\b(automaton|automata|android\w*|consciousness|psycholog\w+|laws? of thought|"
    r"mental (physiolog\w+|philosoph\w+|science)|physiology of mind|"
    r"physical basis of mind|philosophy of mind|science of mind|phrenolog\w+|"
    r"animal intelligence|the intellectual powers|senses? and the intellect|"
    r"mind and body|body and mind|calculating engine|analytical engine|"
    r"difference engine|thinking machine|symbolic logic|formal logic)\b", re.I)
_TM_MINDWORD_RE = re.compile(r"\b(mind|minds|intellect\w*|cognit\w+|cerebr\w+|reasoning|sensation\w*)\b", re.I)
_TM_CUE_RE = re.compile(r"\b(philosoph\w+|scien\w+|treatise|principles?|theor\w+|essays?|"
                        r"inquir\w+|investigation|physiolog\w+|psycholog\w+|analysis|"
                        r"nature of|study|lectures?|logic)\b", re.I)
_TM_EXCLUDE_RE = re.compile(
    r"\b(surgery|surgical|anatom\w+|dissection|materia medica|therapeutic\w*|nursing|"
    r"midwifery|obstetric\w*|clinical|hospital|operative|diseases of the nervous|"
    r"practice of (medicine|physic)|fevers?|consumption|ophthalmic|venereal|"
    r"vaccination|patents?|machine[- ]gun\w*|steam|corliss|locomotive|boiler|turbine|"
    r"artillery|naval|railway|engineering|"
    # Economics / social-policy / commerce — polymath authors (Jevons, Whewell) wrote
    # widely; without this their non-mind works route into thinking_machines via author.
    r"coal|labou?r|trades?[\s'’-]*(union\w*|societ\w*)|social reform|political economy|"
    r"currency|bimetall\w+|tariff|taxation|free trade|poor law|factory act\w*|"
    r"commerc\w+|finance|financial|wages|the state in relation)\b", re.I)
_TM_AUTHORS = {"babbage", "huxley", "tyndall", "boole", "jevons", "whewell", "maudsley",
               "mercier", "romanes", "lewes", "bain", "spencer", "carpenter", "sully", "ferrier"}

# OCR furniture: running headers (>=2 consecutive ALL-CAPS words, e.g. "SLEEP AND
# DREAMS.", "THE PHYSICAL BASIS OF MIND.") and page numbers get fused into the body.
# Capword requires >=2 capital LETTERS so single initials ("J. S. Mill") survive.
_OCR_RUNHEAD_RE = re.compile(r"\s\d{0,4}\s*(?:[A-Z]{2,}[.'&]?\s+){1,}[A-Z]{2,}[.'&]?(?=\s)")
_OCR_PAGENUM_RE = re.compile(r"(?<=[.;,])\s+\d{1,4}\s+(?=[A-Z])")   # bare page no. between sentences
_OCR_FUSEDNUM_RE = re.compile(r"([a-z])\d{2,4}(?=\s)")             # "pheno406 ..." -> "pheno ..."


def _strip_ocr_furniture(text: str) -> str:
    t = " " + text + " "
    for _ in range(2):                 # headers can sit back-to-back
        t = _OCR_RUNHEAD_RE.sub(" ", t)
    t = _OCR_PAGENUM_RE.sub(" ", t)
    t = _OCR_FUSEDNUM_RE.sub(r"\1", t)
    return re.sub(r"\s{2,}", " ", t).strip()


_HATHI_META_CACHE: dict[str, dict] | None = None

# Collapse volumes/copies of one work to a single key so the per-book cap applies
# across them (Vols 1-3 of "Problems of Life and Mind" are separate htids).
_VOL_RE = re.compile(r"\b(vol\w*|part|book|tome|pt)\.?\s*[ivxlcdm0-9]+\b", re.IGNORECASE)


def _work_key(title: str, author: str) -> str:
    t = _VOL_RE.sub(" ", title.lower())
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()[:50]
    return f"{author.split(',')[0].strip().lower()}|{t}"


def _hathi_title_on_theme(title: str, author: str) -> bool:
    if _TM_EXCLUDE_RE.search(title):
        return False
    if _TM_TITLE_RE.search(title):
        return True
    if _TM_MINDWORD_RE.search(title) and _TM_CUE_RE.search(title):
        return True
    return author.split(",")[0].strip().lower() in _TM_AUTHORS


def _load_hathi_meta() -> dict[str, dict]:
    """htid -> {title, author} for on-theme volumes only (cached)."""
    global _HATHI_META_CACHE
    if _HATHI_META_CACHE is not None:
        return _HATHI_META_CACHE
    meta: dict[str, dict] = {}
    if not HATHI_META.exists():
        print(f"  WARNING: {HATHI_META} not found — HATHI routing will lack titles")
        _HATHI_META_CACHE = meta
        return meta
    with HATHI_META.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            title = (row.get("title") or "").strip()
            author = (row.get("authors") or "").strip()
            if not _hathi_title_on_theme(title, author):
                continue
            hid = (row.get("id") or "").strip()
            if hid:
                meta[hid] = {"title": title, "author": author,
                             "year": str(row.get("year") or "").strip()}
    _HATHI_META_CACHE = meta
    return meta


WELLCOME_META = Path("wellcome_titles.jsonl")  # {id, t (title), y (year)}
_WELLCOME_META_CACHE: dict[str, dict] | None = None


def _wnorm(htid: str) -> str:
    """Wellcome shard htid -> wellcome_titles id form (upper, drop _NNNN volume suffix)."""
    return str(htid).upper().split("_")[0]


def _load_wellcome_meta() -> dict[str, dict]:
    """normalized id -> {title, year} (cached). Local Wellcome metadata has no author."""
    global _WELLCOME_META_CACHE
    if _WELLCOME_META_CACHE is not None:
        return _WELLCOME_META_CACHE
    meta: dict[str, dict] = {}
    if WELLCOME_META.exists():
        with WELLCOME_META.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                meta[_wnorm(r["id"])] = {"title": (r.get("t") or "").strip(),
                                         "year":  str(r.get("y") or "").strip()}
    else:
        print(f"  WARNING: {WELLCOME_META} not found — WELLCOME records will lack titles")
    _WELLCOME_META_CACHE = meta
    return meta


def load_hathi_broad_passages(rng: random.Random,
                              max_shards: int = 150,
                              max_passages: int = 60_000,
                              min_ocr: float = 0.85,
                              per_book: int = 4) -> list[dict]:
    """Sample broadly across ALL of hathi_shards_dedup (1,343 parquet files — the
    largest single source in the pretraining corpus, ~half of it by file count),
    unlike load_hathi_passages() which is deliberately restricted to an on-theme
    (mind/machine/consciousness) workset CSV built for the thinking_machines
    sub-corpus specifically. This loader has no title/author metadata (the workset
    CSV that provides that for the on-theme loader doesn't cover the broad corpus),
    so passages route on lexical content alone via route_passage_to_subcorpus() —
    same situation as the IB loader. lang is filtered to English; htid is used
    directly as the per-book cap key since there's no title to collapse
    multi-volume works onto a shared work key.
    """
    if not HATHI_SHARDS_DIR.exists():
        print(f"  WARNING: {HATHI_SHARDS_DIR} not found — skipping HATHI_BROAD")
        return []
    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not installed — skipping HATHI_BROAD")
        return []

    shard_files = sorted(HATHI_SHARDS_DIR.glob("*.parquet"))
    rng.shuffle(shard_files)
    shard_files = shard_files[:max_shards]
    all_passages: list[dict] = []
    books_seen = 0
    book_count: dict[str, int] = defaultdict(int)

    for shard in shard_files:
        try:
            df = pd.read_parquet(shard, columns=["text", "htid", "year", "lang", "ocr_score"])
        except Exception as e:
            print(f"  WARNING: error reading {shard.name}: {e}")
            continue
        for row in df.itertuples(index=False):
            lang = getattr(row, "lang", None)
            if lang is not None and str(lang).lower() not in ("eng", "en", ""):
                continue
            score = getattr(row, "ocr_score", None)
            if score is not None and score < min_ocr:
                continue
            text = row.text if isinstance(row.text, str) else ""
            if len(text.split()) < MIN_WORDS:
                continue
            hid = getattr(row, "htid", "") or f"{shard.stem}:{books_seen}"
            books_seen += 1
            cleaned = [c for c in (_strip_ocr_furniture(p) for p in extract_passages_from_text(text))
                       if len(c.split()) >= MIN_WORDS]
            remaining = per_book - book_count[hid]
            if remaining <= 0:
                continue
            if len(cleaned) > remaining:
                cleaned = rng.sample(cleaned, remaining)
            book_count[hid] += len(cleaned)
            yr = getattr(row, "year", None)
            all_passages.extend(
                {
                    "text":        p,
                    "source":      "HATHI_BROAD",
                    "title":       hid,
                    "book":        hid,
                    "author":      "unknown",
                    "book_title":  "",
                    "book_author": "",
                    "book_year":   str(int(yr)) if yr else "",
                    "htid":        hid,
                }
                for p in cleaned
            )

    print(f"  {'HATHI_BROAD sampled books':<40} {len(all_passages):>7,} candidates "
          f"({books_seen} books in {len(shard_files)} shards)")
    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


def load_hathi_passages(rng: random.Random,
                        max_shards: int = 200,
                        max_passages: int = 40_000,
                        min_ocr: float = 0.80,
                        per_book: int = 6) -> list[dict]:
    """Sample passages from on-theme HathiTrust books (mind / machine / consciousness).

    `per_book` caps how many passages any single volume contributes, so a few large
    works (e.g. Lewes) can't dominate the pool. Each htid is one row in the deduped
    shards, so the cap is applied per row.
    """
    if not HATHI_SHARDS_DIR.exists():
        print(f"  WARNING: {HATHI_SHARDS_DIR} not found — skipping HATHI")
        return []
    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not installed — skipping HATHI")
        return []

    targets = _load_hathi_meta()
    if not targets:
        print("  WARNING: no on-theme HATHI volumes resolved — skipping HATHI")
        return []
    target_ids = set(targets)

    shard_files = sorted(HATHI_SHARDS_DIR.glob("*.parquet"))
    rng.shuffle(shard_files)
    shard_files = shard_files[:max_shards]
    all_passages: list[dict] = []
    books_seen = 0
    work_count: dict[str, int] = defaultdict(int)   # per-work cap, across volumes/shards

    for shard in shard_files:
        try:
            df = pd.read_parquet(shard, columns=["text", "htid", "ocr_score"])
        except Exception as e:
            print(f"  WARNING: error reading {shard.name}: {e}")
            continue
        for row in df.itertuples(index=False):
            hid = getattr(row, "htid", "")
            if hid not in target_ids:
                continue
            score = getattr(row, "ocr_score", None)
            if score is not None and score < min_ocr:
                continue
            text = row.text if isinstance(row.text, str) else ""
            if len(text.split()) < MIN_WORDS:
                continue
            m = targets[hid]
            wkey = _work_key(m["title"], m["author"])
            remaining = (per_book - work_count[wkey]) if per_book else None
            if per_book and remaining <= 0:
                continue
            books_seen += 1
            cleaned = [c for c in (_strip_ocr_furniture(p) for p in extract_passages_from_text(text))
                       if len(c.split()) >= MIN_WORDS]
            if per_book and len(cleaned) > remaining:
                cleaned = rng.sample(cleaned, remaining)
            work_count[wkey] += len(cleaned)
            all_passages.extend(
                {
                    "text":        p,
                    "source":      "HATHI",
                    "title":       m["title"],
                    "book":        m["title"][:60],
                    "author":      m["author"],
                    "book_title":  m["title"],
                    "book_author": m["author"],
                    "book_year":   m["year"],
                    "htid":        hid,
                }
                for p in cleaned
            )

    print(f"  {'HATHI on-theme books':<40} {len(all_passages):>7,} candidates "
          f"({books_seen} books in {len(shard_files)} shards)")
    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


def load_wellcome_passages(rng: random.Random,
                           max_shards: int = 80,
                           max_passages: int = 30_000,
                           min_ocr: float = 0.80,
                           per_book: int = 6) -> list[dict]:
    """Sample passages from Wellcome medical shards (mental physiology / mind-body).

    No per-book title metadata is available, so routing relies on in-text lexical
    patterns; the Stage-1/Stage-2 judges remain the quality gate.
    """
    if not WELLCOME_SHARDS_DIR.exists():
        print(f"  WARNING: {WELLCOME_SHARDS_DIR} not found — skipping WELLCOME")
        return []
    try:
        import pandas as pd
    except ImportError:
        print("  WARNING: pandas not installed — skipping WELLCOME")
        return []

    wmeta = _load_wellcome_meta()
    shard_files = sorted(WELLCOME_SHARDS_DIR.glob("*.parquet"))
    sampled = rng.sample(shard_files, min(max_shards, len(shard_files)))
    all_passages: list[dict] = []
    books_seen = 0

    for shard in sampled:
        try:
            df = pd.read_parquet(shard, columns=["text", "htid", "ocr_score", "year"])
        except Exception as e:
            print(f"  WARNING: error reading {shard.name}: {e}")
            continue
        for row in df.itertuples(index=False):
            score = getattr(row, "ocr_score", None)
            if score is not None and score < min_ocr:
                continue
            text = row.text if isinstance(row.text, str) else ""
            if len(text.split()) < MIN_WORDS:
                continue
            books_seen += 1
            hid = getattr(row, "htid", "")
            wm = wmeta.get(_wnorm(hid), {})
            yr = wm.get("year") or (str(int(row.year)) if getattr(row, "year", None) else "")
            title = wm.get("title") or hid or shard.stem
            cleaned = [c for c in (_strip_ocr_furniture(p) for p in extract_passages_from_text(text))
                       if len(c.split()) >= MIN_WORDS]
            if per_book and len(cleaned) > per_book:
                cleaned = rng.sample(cleaned, per_book)
            all_passages.extend(
                {
                    "text":        p,
                    "source":      "WELLCOME",
                    "title":       title,
                    "book":        title,
                    "author":      "unknown",
                    "book_title":  wm.get("title", ""),
                    "book_author": "",            # not in local Wellcome metadata
                    "book_year":   yr,
                    "htid":        hid,
                }
                for p in cleaned
            )

    print(f"  {'WELLCOME shard books':<40} {len(all_passages):>7,} candidates "
          f"({books_seen} books in {len(sampled)} shards)")
    if len(all_passages) > max_passages:
        all_passages = rng.sample(all_passages, max_passages)
    return all_passages


# ---------------------------------------------------------------------------
# EB QA loader — pre-formed Q/A pairs, fills encyclopedic_factual directly
# ---------------------------------------------------------------------------

# EB questions are formulaic ("Could you tell me about Aa?", "What is X?").
# Pull the topic noun out so downstream record metadata is informative.
_EB_TOPIC_RE = re.compile(
    r"(?:tell me about|what (?:is|can you tell me about|about)|who (?:is|was)|"
    r"do you know(?: about| of)?|what do you know about|describe)\s+(.+?)\s*[\?\.!]?\s*$",
    re.IGNORECASE,
)


def _eb_topic_from_question(q: str) -> str:
    m = _EB_TOPIC_RE.search(q.strip())
    if not m:
        return "encyclopaedia entry"
    topic = m.group(1).strip().rstrip(".?!,;:")
    topic = re.sub(r"^(?:a|an|the)\s+", "", topic, flags=re.IGNORECASE)
    return topic[:60] or "encyclopaedia entry"


def load_eb_qa_passages(rng: random.Random,
                        max_passages: int = 15_000) -> list[dict]:
    """
    Load Encyclopaedia Britannica QA pairs as pre-formed passages tagged
    encyclopedic_factual. These passages carry a `user_text` field that
    signals the LM-question step should be skipped — the EB pair is used
    as-is as an SFT example.
    """
    if not EB_QA_DIR.exists():
        print(f"  WARNING: {EB_QA_DIR} not found — skipping EB")
        return []

    pairs: list[dict] = []
    for fp in sorted(EB_QA_DIR.glob("*.jsonl")):
        try:
            with fp.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, list) or len(rec) < 2:
                        continue
                    user = next(
                        (m.get("content", "") for m in rec if m.get("role") == "user"),
                        "",
                    ).strip()
                    asst = next(
                        (m.get("content", "") for m in rec if m.get("role") == "assistant"),
                        "",
                    ).strip()
                    if not user or not asst:
                        continue
                    pairs.append({
                        "text":            asst,
                        "source":          "EB",
                        "title":           "Encyclopaedia Britannica",
                        "book":            fp.stem,
                        "author":          "Encyclopaedia Britannica",
                        "book_title":      "Encyclopaedia Britannica",
                        "book_author":     "Encyclopaedia Britannica",
                        "prov_method":     "static",
                        "user_text":       user,
                        "sub_corpus":      "encyclopedic_factual",
                        "sub_score":       100,
                        "sub_corpus_all":  ["encyclopedic_factual"],
                    })
        except Exception as e:
            print(f"  WARNING: error reading {fp.name}: {e}")
            continue

    if len(pairs) > max_passages:
        pairs = rng.sample(pairs, max_passages)

    print(f"  {'EB encyclopedia QA pairs':<40} {len(pairs):>7,} candidates")
    return pairs


# ---------------------------------------------------------------------------
# Corpus sampling (weighted across source types)
# ---------------------------------------------------------------------------

SOURCE_LOADERS = {
    "ncse":        ("Loading NCSE passages...",                load_ncse_passages),
    "bl":          ("Loading BL passages...",                  load_bl_passages),
    "pg":          ("Loading PG Nonfiction passages...",       load_pg_nonfiction_passages),
    "ia":          ("Loading IA passages...",                  load_ia_passages),
    "ib":          ("Loading IB (Institutional Books) passages...", load_ib_passages),
    "hathi":       ("Loading HATHI on-theme passages...",      load_hathi_passages),
    "hathi_broad": ("Loading HATHI_BROAD passages...",         load_hathi_broad_passages),
    "wellcome":    ("Loading WELLCOME passages...",            load_wellcome_passages),
}
DEFAULT_SOURCES = ["ncse", "bl", "pg", "ia", "ib", "hathi_broad"]


def sample_passages(
    n: int,
    rng: random.Random,
    subcorpus_weighting: bool,
    disposition_map: Path,
    mixed_source_ratio: float,
    sources: list[str] | None = None,
    only_subcorpus: str | None = None,
    max_shards: int | None = None,
    per_book: int | None = None,
) -> list[dict]:
    """
    Build a pool from the selected sources, then draw n passages with source-level
    weighting. Returns list of {text, source, title} dicts.
    """
    sources = sources or DEFAULT_SOURCES
    pool: list[dict] = []
    for key in sources:
        msg, loader = SOURCE_LOADERS[key]
        print(msg)
        # Only the shard-based parquet loaders accept max_shards / per_book overrides.
        if key in ("hathi", "hathi_broad", "wellcome"):
            kw = {}
            if max_shards:
                kw["max_shards"] = max_shards
            if per_book:
                kw["per_book"] = per_book
            pool.extend(loader(rng, **kw))
        else:
            pool.extend(loader(rng))

    if not pool:
        print("ERROR: no passages found — check source paths")
        sys.exit(1)

    # Route every candidate into one or more sub-corpus buckets. We keep
    # the primary tag for audit-trail purposes, plus the full match list so
    # the quota sampler can discover a passage from any of its tagged subs.
    # EB passages arrive pre-tagged as encyclopedic_factual — leave them
    # alone; the rest go through the lexical + author/title router.
    for p in pool:
        if p.get("sub_corpus_all"):
            continue
        matches = route_passage_to_subcorpus(
            p["text"], p.get("title", ""), p.get("author", "")
        )
        if matches:
            p["sub_corpus"] = matches[0][0]
            p["sub_score"] = matches[0][1]
            p["sub_corpus_all"] = [m[0] for m in matches]
        else:
            p["sub_corpus"] = "unassigned"
            p["sub_score"] = 0
            p["sub_corpus_all"] = []

    if only_subcorpus:
        before = len(pool)
        pool = [p for p in pool if only_subcorpus in p.get("sub_corpus_all", [])]
        print(f"\n  --only-subcorpus={only_subcorpus}: kept {len(pool):,} of {before:,} routed passages")
        if not pool:
            print(f"ERROR: no passages routed to '{only_subcorpus}' — broaden sources or patterns")
            sys.exit(1)

    src_totals = Counter(p["source"] for p in pool)
    sub_totals = Counter(p["sub_corpus"] for p in pool)
    print(f"\n  Pool totals:")
    for src, cnt in src_totals.most_common():
        print(f"    {src:<15} {cnt:>8,}")
    print(f"\n  Routed sub-corpus totals:")
    for sub, cnt in sub_totals.most_common(14):
        print(f"    {sub:<22} {cnt:>8,}")
    print(f"  Total pool: {len(pool):,}")

    chosen: list[dict]
    targets = {}
    if subcorpus_weighting:
        targets = derive_subcorpus_weights(disposition_map, mixed_source_ratio)
        if targets:
            print("\n  Target sub-corpus weights (A + mixed-F source slice):")
            for sub, wt in sorted(targets.items(), key=lambda kv: kv[1], reverse=True):
                print(f"    {sub:<22} {wt * 100:>6.2f}%")
            chosen = _sample_with_subcorpus_targets(pool, n, rng, targets)
        else:
            print("  WARNING: could not derive sub-corpus targets; using source-only weighting.")
            subcorpus_weighting = False

    if not subcorpus_weighting:
        # Legacy behavior: weighted by source family only, dedup by passage prefix.
        chosen = _weighted_pick_without_replacement(pool, n, rng, seen=set())

    src_counts = Counter(p["source"] for p in chosen)
    sub_counts = Counter(p.get("sub_corpus", "unassigned") for p in chosen)
    print(f"\n  Sampled {len(chosen):,} passages:")
    for src, count in src_counts.most_common():
        print(f"    {src:<15} {count:>5,}")
    print("\n  Sampled sub-corpus mix:")
    for sub, count in sub_counts.most_common(14):
        pct = 100.0 * count / max(1, len(chosen))
        target_pct = (targets.get(sub, 0.0) * 100.0) if targets else 0.0
        if targets:
            print(f"    {sub:<22} {count:>5,} ({pct:>5.1f}%)  target={target_pct:>5.1f}%")
        else:
            print(f"    {sub:<22} {count:>5,} ({pct:>5.1f}%)")

    return chosen


# ---------------------------------------------------------------------------
# Stage 1 — Passage gate + prompt generation
# ---------------------------------------------------------------------------

_USER_CORPUS_EXAMPLES = """\
   "Is it ever right to go to war?"
   "Should the poor be given relief without having to enter the workhouse?"
   "What did you think of the Tichborne trial?"
   "Do you not think Shakespeare the greatest writer England has produced?"
   "What do you think of trade unions?"
   "Do you think the Chartists could ever have won if more people had joined them?"
   "What are your thoughts on the Irish?"
   "What do you think the coming of the railway has done for country towns?"
   "Perchance you have heard of Mr. Babbage and his engine?"
   "What is electricity, really?"
   "How does typhoid fever spread?"
   "Explain how a coal fire was laid and kept going in a Victorian kitchen."
   "What is it like to go from Manchester to Liverpool on the train?"
   "what should a gentleman wear to a dinner party in the country?"
   "How would a merchant's son get himself into Parliament?"
   "Is it proper to keep a parrot in the drawing room?"
   "What happens to a man who is caught poaching?"
   "What is the purpose of education?"
   "What is your opinion of Thomas Carlyle?"
   "What do you make of Mr. Ruskin and his ideas about art?"
   "Explain to me how a steam locomotive works."
   "I have long wondered whether the bat ought to be counted among the birds, since it flies, or among the beasts, since it has fur."
   "I should like to know your view of the opium trade with China."
   "There are boys sent up chimneys to sweep them who are barely seven years old."
   "Tell me what you know of Mr. Brunel and his bridges.\""""

GENERATOR_SYSTEM_PROMPT = f"""\
You are curating training data for a Victorian-era chatbot. You will be given a passage \
from a real 19th-century text.

Your job is to:

1. Assess whether the passage is a self-contained, coherent paragraph. Reject it if:
   - Mid-scene narrative fiction: characters, events, or locations are referenced but \
not defined within the passage itself
   - OCR garbage: the text is dominated by garbled characters, tables, or non-prose content
   - Anaphoric opening: the passage begins with references a listener cannot resolve \
("the signals used", "these sentiments", "the proposed regulations") without prior context
   - Dangling pronouns: "he", "she", or "they" appear as the primary subject with no \
named referent anywhere in the passage

2. If usable, write ONE natural prompt that a curious modern person might send, to which \
this passage would serve as a plausible Victorian gentleman's reply. The prompt should:
   - Sound like a real person talking, not an exam question
   - Be open enough that the passage answers it naturally
   - NOT mention a book, passage, or text -- the person is asking the Victorian directly
   - Be 10-25 words
   - Name specific people, places, diseases, or things -- never use vague demonstratives \
like "this man", "this illness", or "this factory" without naming the referent
   - Take the identified topic into account

For calibration, here are typical inputs that users of this system send -- \
note that they are not always questions:
{_USER_CORPUS_EXAMPLES}

3. Also identify: topic (3-5 plain words, e.g. "steam engine operation"), \
content_type (exactly one of: expository, narrative, instructional, argumentative, descriptive).


SECOND-PERSON ADDRESS IS NOT A DEFECT. This gate was written for expository book passages,
where a reply explains something to no one in particular. Passages drawn from LETTERS and
personal correspondence are DIALOGIC -- they address a reader directly. A passage saying
"you", "your letter", "as you say", "I cannot agree with your view" resolves perfectly in
this application, because that reader IS the person the Victorian is speaking to. Accept
such passages; do not count "you"/"your" as an unresolved reference.
Measured 2026-08-22: rejecting second-person address held the gate pass rate on curated
Victorian correspondence to 4%, which is why the first epistolary rebuild wrote only 25
rows. With this exception the rate is 29%, and the remaining rejections are genuine
third-party anaphora.
Continue to reject THIRD-PARTY anaphora: "he"/"she"/"they" as the main subject with no
named referent, or named characters whose circumstances are never explained.

Respond with ONLY valid JSON, no commentary:
If rejected: {{"passage_ok": false, "reason": "brief reason"}}
If accepted: {{"passage_ok": true, "topic": "...", "content_type": "...", "prompt": "..."}}
"""

# (passage_text, assistant_json) pairs — 2 rejected, 3 accepted.
_GENERATOR_FEW_SHOTS: list[tuple[str, str]] = [
    (
        # Rejected — mid-scene fiction; characters and events undefined.
        'Passage:\n\n'
        "We never should have heard of the letter at all, I believe, unless I had asked "
        "for it myself. Do you call that confidence, my love, towards Doctor Strong? I am "
        "surprised. You ought to know better. The letter was reluctantly produced; and as "
        "I handed it to the old lady, I saw how the unwilling hand from which I took it "
        "trembled. \"Now let us see,\" said Mrs. Markleham, putting her glass to her eye, "
        "\"where the passage is. The remembrance of old times, my dearest Annie -- and so "
        "forth -- it's not there. The amiable old Proctor -- who's he? Dear me, Annie, "
        "how illegibly your cousin Maldon writes, and how stupid I am! Doctor, of course.\"",
        '{"passage_ok": false, "reason": "mid-scene fiction; Mrs. Markleham, Doctor Strong, '
        'and Annie are all undefined for a listener who was not present at the scene"}',
    ),
    (
        # Accepted — self-contained technical exposition.
        'Passage:\n\n'
        "In order to transmit messages along the telegraphic wires, a certain amount of "
        "electric power is required, and this force must be increased in proportion to the "
        "length of the wire. A large wire, however, will transmit more electricity than a "
        "small one, in proportion to the squares of their diameters, or the areas of their "
        "section. According to some accurate experiments made by M. Pouillet, a "
        "distinguished member of the Institute of France, a voltaic battery of ten pairs "
        "of plates will supply electricity sufficient for telegraphic purposes by means of "
        "a copper wire one eighth of an inch in diameter, and 600 miles in length. A force "
        "of much less intensity, however, is amply sufficient, as it is easy to strengthen "
        "the current by batteries at intermediate stations.",
        '{"passage_ok": true, "topic": "electric power in telegraph transmission", '
        '"content_type": "expository", '
        '"prompt": "How does the telegraph manage to send messages over such great distances?"}',
    ),
    (
        # Accepted — self-contained literary argument.
        'Passage:\n\n'
        "Now in no poem of Coleridge will one find either the one or the other kind of "
        "completeness; there is not only no mechanism in it, but no apparent organisation; "
        "his thoughts flow out of one another, rather than grow out of one another, "
        "following some law of association which we cannot analyse though we spontaneously "
        "reply to it -- nay, we cannot suppose the process to have passed above the plane "
        "of consciousness in Coleridge himself. It is this entire absence of space, time, "
        "cause and effect, theory, &c., which seems to distinguish Coleridge from all other "
        "poets, even those that seem most to resemble him. I do not say that all this gives "
        "him an advantage over Wordsworth, but it certainly gives his poetry an air of "
        "inspiration not to be found in any other, not even in that of Shelley, which seems "
        "at first so much more wonderful. Coleridge's thoughts seem to me always to "
        "represent the Ideas of the Pure Reason and the aspirations of a Love too large to "
        "be limited by human relations.",
        '{"passage_ok": true, "topic": "the poetry of Coleridge", '
        '"content_type": "argumentative", '
        '"prompt": "What do you think of Coleridge\'s poetry?"}',
    ),
    (
        # Rejected — anaphoric opening; "the signals used" and "Hook's invention"
        # are never defined within the passage.
        'Passage:\n\n'
        "The signals used were either large letters of the alphabet, or figures of various "
        "shapes to represent them: the latter being the more valuable, as by a change of "
        "key, the nature of the communication might be kept a secret from those actually "
        "employed in making the signals. Amontous tried this method in a small tract of "
        "land before several persons of the highest rank at the Court of France. But though "
        "Hook's invention and Amontous's modification were published all over Europe, and "
        "the former as early as 1684, yet they were not practically applied to any useful "
        "purpose until the time of the French revolution.",
        '{"passage_ok": false, "reason": "anaphoric opening (\'the signals used\') and '
        'undefined referent (\'Hook\'s invention\') -- a listener has no prior context '
        'to resolve these references"}',
    ),
    (
        # Accepted — self-contained descriptive passage, minor OCR noise.
        'Passage:\n\n'
        "The Boston journals contain an account of a frightful collision at sea, which "
        "caused the loss of one hundred and seventy lives. The ship Shanunga, which arrived "
        "at Boston on the 16th August, from Liverpool, reports that on Monday morning, the "
        "9th August, at one o'clock, in lat. 44 25, longitude 58 30, the weather being "
        "foggy, she came in contact with the Swedish barque Iduna, from Hamburg for New "
        "York, with 206 persons on board. The Iduna sank in about half an hour. Immediately "
        "after the collision the Shanunga's boats were put out, and, with one boat from "
        "the barque, picked up 34 persons only. One hundred and seventy-two persons, "
        "including the master, Captain Ernest Andreas Moberg, were lost. 1500 dollars were "
        "subscribed at Boston in a couple of days for the relief of the sufferers.",
        '{"passage_ok": true, "topic": "a ship collision at sea", '
        '"content_type": "descriptive", '
        '"prompt": "Have you heard any news about shipwrecks recently?"}',
    ),
    (
        # Rejected — lots of OCR noise, and aphoristic phrasing.
        'Passage:\n\n'
        "IL] BOTANY. 223 _. , Dr Livingstone believes this to be a fibrous Jouaze. , . plant" 
        "of great value, and that it was before entirely unknown to botanists. It grows about" 
        "Tete, and in large quantities in the country of the Maravi. He submitted some specimens" 
        "of it to Messrs Pye, Brothers, of London, who pronounce it to be suitable as a substitute" 
        "for flax, in comparison with which it is stronger and of finer fibre. There is a drawing" 
        "of this plant at p. 646, Travels. The natives make a thread of it, which is as strong as"
        "catgut. Possibly our manufacturers will find it adapted for sail-cloth, &c. _ , This is" 
        "another new plant, being a species of aloe of fibrous tissue, found by our traveller in" 
        "the same districts as the Buaze. It was suggested to him by the Portuguese as being" 
        "fitted for tbe manufacture of paper. He has met with several of great value. The Medicinal"
        "-jq-u_ yom;ca) producing strychnia, flourishes abundantly on tbe Leeambye. The Cinchona"
        "bark grows in large quantities on the eastern coast. Senna is there growing in whole" 
        "forests, and possibly, like that of Egypt.",
        '{"passage_ok": false, "reason": "passage starts with OCR noise, lots of OCR noise throughout text, '
        'aphoristic phrasing throughout the text"}'
    ),

]


# ---------------------------------------------------------------------------
# Stage 2 — Pair quality scoring (independent judge)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are evaluating (prompt, passage) pairs for a Victorian-era chatbot training dataset.

You will be given a user prompt and a Victorian passage intended to serve as the \
Victorian gentleman's reply. Score the pair from 0-3:

   3  Excellent -- passage is substantive and self-contained; no dangling pronouns \
("he", "she", "the letter") without named referents inside the passage; passage \
answers the prompt naturally as a Victorian gentleman's reply; prompt is specific \
and natural.
   2  Good -- mostly works; minor OCR noise, slightly expository framing, or mildly \
mid-thought opening that is self-recoverable; prompt is a reasonable fit.
   1  Marginal -- passage has subtle anaphoric dependencies the listener cannot fully \
resolve, or prompt fits poorly, or passage is too thin to constitute a satisfying reply.
   0  Unusable -- passage is mid-scene narrative fiction, dominated by OCR garbage, or \
so referential it makes no sense as a standalone reply; or prompt is deeply mismatched.

Respond with ONLY valid JSON, no commentary:
{"pair_score": 0, "pair_note": "one brief sentence"}
"""

# (prompt+passage user turn, assistant_json) pairs — scores 0, 3, 3, 2, 1.
_JUDGE_FEW_SHOTS: list[tuple[str, str]] = [
    (
        # Score 0 — mid-scene fiction; even a reasonable prompt cannot save it.
        "Prompt: Tell me about the letter that Mrs. Markleham received.\n\nPassage:\n"
        "We never should have heard of the letter at all, I believe, unless I had asked "
        "for it myself. Do you call that confidence, my love, towards Doctor Strong? I am "
        "surprised. You ought to know better. The letter was reluctantly produced; and as "
        "I handed it to the old lady, I saw how the unwilling hand from which I took it "
        "trembled. \"Now let us see,\" said Mrs. Markleham, putting her glass to her eye, "
        "\"where the passage is. The remembrance of old times, my dearest Annie -- and so "
        "forth -- it's not there. The amiable old Proctor -- who's he? Dear me, Annie, "
        "how illegibly your cousin Maldon writes, and how stupid I am! Doctor, of course.\"",
        '{"pair_score": 0, "pair_note": "Mid-scene fiction; Mrs. Markleham, Doctor Strong, '
        'and Annie are undefined for any listener who was not present at the scene."}',
    ),
    (
        # Score 3 — clean technical pair.
        "Prompt: How does the telegraph manage to send messages over such great distances?\n\nPassage:\n"
        "In order to transmit messages along the telegraphic wires, a certain amount of "
        "electric power is required, and this force must be increased in proportion to the "
        "length of the wire. A large wire, however, will transmit more electricity than a "
        "small one, in proportion to the squares of their diameters, or the areas of their "
        "section. According to some accurate experiments made by M. Pouillet, a "
        "distinguished member of the Institute of France, a voltaic battery of ten pairs "
        "of plates will supply electricity sufficient for telegraphic purposes by means of "
        "a copper wire one eighth of an inch in diameter, and 600 miles in length. A force "
        "of much less intensity, however, is amply sufficient, as it is easy to strengthen "
        "the current by batteries at intermediate stations.",
        '{"pair_score": 3, "pair_note": "Self-contained technical exposition; answers the '
        'prompt directly as a Victorian explaining telegraphy, no prior context needed."}',
    ),
    (
        # Score 3 — clean argumentative pair.
        "Prompt: What do you think of Coleridge's poetry?\n\nPassage:\n"
        "Now in no poem of Coleridge will one find either the one or the other kind of "
        "completeness; there is not only no mechanism in it, but no apparent organisation; "
        "his thoughts flow out of one another, rather than grow out of one another, "
        "following some law of association which we cannot analyse though we spontaneously "
        "reply to it -- nay, we cannot suppose the process to have passed above the plane "
        "of consciousness in Coleridge himself. It is this entire absence of space, time, "
        "cause and effect, theory, &c., which seems to distinguish Coleridge from all other "
        "poets, even those that seem most to resemble him. I do not say that all this gives "
        "him an advantage over Wordsworth, but it certainly gives his poetry an air of "
        "inspiration not to be found in any other, not even in that of Shelley, which seems "
        "at first so much more wonderful. Coleridge's thoughts seem to me always to "
        "represent the Ideas of the Pure Reason and the aspirations of a Love too large to "
        "be limited by human relations.",
        '{"pair_score": 3, "pair_note": "Self-contained critical opinion; passage delivers '
        'a complete Victorian literary judgement with no external references needed."}',
    ),
    (
        # Score 2 — good pair, minor OCR noise in vessel names.
        "Prompt: Have you heard any news about shipwrecks recently?\n\nPassage:\n"
        "The Boston journals contain an account of a frightful collision at sea, which "
        "caused the loss of one hundred and seventy lives. The ship Shanunga, which arrived "
        "at Boston on the 16th August, from Liverpool, reports that on Monday morning, the "
        "9th August, at one o'clock, in lat. 44 25, longitude 58 30, the weather being "
        "foggy, she came in contact with the Swedish barque Iduna, from Hamburg for New "
        "York, with 206 persons on board. The Iduna sank in about half an hour. Immediately "
        "after the collision the Shanunga's boats were put out, and, with one boat from "
        "the barque, picked up 34 persons only. One hundred and seventy-two persons, "
        "including the master, Captain Ernest Andreas Moberg, were lost. 1500 dollars were "
        "subscribed at Boston in a couple of days for the relief of the sufferers.",
        '{"pair_score": 2, "pair_note": "Good self-contained passage; slight OCR noise in '
        'vessel names but content is clear and answers naturally."}',
    ),
    (
        # Score 1 — coherent topic, anaphoric opening strands the listener.
        "Prompt: Were there ways to send messages quickly before the electric telegraph was invented?\n\nPassage:\n"
        "The signals used were either large letters of the alphabet, or figures of various "
        "shapes to represent them: the latter being the more valuable, as by a change of "
        "key, the nature of the communication might be kept a secret from those actually "
        "employed in making the signals. Amontous tried this method in a small tract of "
        "land before several persons of the highest rank at the Court of France. But though "
        "Hook's invention and Amontous's modification were published all over Europe, and "
        "the former as early as 1684, yet they were not practically applied to any useful "
        "purpose until the time of the French revolution.",
        '{"pair_score": 1, "pair_note": "Coherent topic but anaphoric opening '
        "(\'the signals used\') and undefined referent (\'Hook\'s invention\') "
        'leave a listener without prior context genuinely stranded."}',
    ),
]


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")
_VALID_CONTENT_TYPES = {"expository", "narrative", "instructional", "argumentative", "descriptive"}


def _parse_json_obj(raw: str) -> dict | None:
    """Strip markdown fences and extract the first JSON object."""
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    m = _JSON_OBJ_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_generation_json(raw: str) -> dict | None:
    """
    Parse Stage 1 generator output.

    Returns {"passage_ok": False, "reason": str} if the passage was rejected,
    {"passage_ok": True, "topic": str, "content_type": str, "prompt": str} if accepted,
    or None on parse failure.
    """
    obj = _parse_json_obj(raw)
    if obj is None:
        return None

    passage_ok = bool(obj.get("passage_ok", False))
    if not passage_ok:
        return {"passage_ok": False, "reason": str(obj.get("reason", "")).strip()[:120]}

    topic        = str(obj.get("topic", "")).strip()
    content_type = str(obj.get("content_type", "")).strip().lower()
    prompt       = str(obj.get("prompt", "")).strip()
    if not topic or not prompt:
        return None
    if content_type not in _VALID_CONTENT_TYPES:
        content_type = "expository"

    return {
        "passage_ok":   True,
        "topic":        topic,
        "content_type": content_type,
        "prompt":       prompt,
    }


def parse_judgment_json(raw: str) -> dict | None:
    """
    Parse Stage 2 judge output.

    Returns {"pair_score": int, "pair_note": str} or None on parse failure.
    """
    obj = _parse_json_obj(raw)
    if obj is None:
        return None

    pair_score_raw = obj.get("pair_score")
    try:
        pair_score = int(pair_score_raw) if pair_score_raw is not None else None
        if pair_score not in (0, 1, 2, 3):
            pair_score = None
    except (TypeError, ValueError):
        pair_score = None

    if pair_score is None:
        return None

    return {
        "pair_score": pair_score,
        "pair_note":  str(obj.get("pair_note", "")).strip()[:120],
    }


class MLXMetadataBackend:
    def __init__(self, model_name: str):
        from mlx_lm import load, generate

        self.model_name = model_name
        self._generate = generate
        print(f"loading local model {model_name}...")
        self.model, self.tokenizer = load(model_name)
        print("local model loaded")

    def _call(
        self,
        system: str,
        shots: list[tuple[str, str]],
        user_content: str,
        max_tokens: int,
    ) -> str:
        messages = [{"role": "system", "content": system}]
        for ex_user, ex_asst in shots:
            messages.append({"role": "user",      "content": ex_user})
            messages.append({"role": "assistant", "content": ex_asst})
        messages.append({"role": "user", "content": user_content})
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._generate(self.model, self.tokenizer, prompt=prompt,
                              max_tokens=max_tokens, verbose=False)

    def generate_prompt_raw(self, passage_text: str, max_tokens: int = 200) -> str:
        return self._call(
            GENERATOR_SYSTEM_PROMPT,
            _GENERATOR_FEW_SHOTS,
            f"Passage:\n\n{passage_text}",
            max_tokens,
        )

    def judge_pair_raw(self, prompt: str, passage_text: str, max_tokens: int = 80) -> str:
        return self._call(
            JUDGE_SYSTEM_PROMPT,
            _JUDGE_FEW_SHOTS,
            f"Prompt: {prompt}\n\nPassage:\n{passage_text}",
            max_tokens,
        )


class AnthropicMetadataBackend:
    def __init__(self):
        import anthropic

        self.client = anthropic.AsyncAnthropic()

    async def _call(
        self,
        system: str,
        shots: list[tuple[str, str]],
        user_content: str,
        model: str,
        semaphore: asyncio.Semaphore,
        max_tokens: int,
    ) -> str | None:
        messages: list[dict] = []
        for ex_user, ex_asst in shots:
            messages.append({"role": "user",      "content": ex_user})
            messages.append({"role": "assistant", "content": ex_asst})
        messages.append({"role": "user", "content": user_content})
        async with semaphore:
            try:
                response = await self.client.messages.create(
                    model=model, max_tokens=max_tokens, temperature=0.3,
                    system=system, messages=messages,
                )
                return response.content[0].text.strip()
            except Exception as e:
                print(f"  API error: {e}", file=sys.stderr)
                return None

    async def generate_prompt_raw(
        self,
        passage_text: str,
        model: str,
        semaphore: asyncio.Semaphore,
        max_tokens: int = 200,
    ) -> str | None:
        return await self._call(
            GENERATOR_SYSTEM_PROMPT, _GENERATOR_FEW_SHOTS,
            f"Passage:\n\n{passage_text}",
            model, semaphore, max_tokens,
        )

    async def judge_pair_raw(
        self,
        prompt: str,
        passage_text: str,
        model: str,
        semaphore: asyncio.Semaphore,
        max_tokens: int = 80,
    ) -> str | None:
        return await self._call(
            JUDGE_SYSTEM_PROMPT, _JUDGE_FEW_SHOTS,
            f"Prompt: {prompt}\n\nPassage:\n{passage_text}",
            model, semaphore, max_tokens,
        )


# ---------------------------------------------------------------------------
# Assemble instruction-tuning record
# ---------------------------------------------------------------------------

INSTRUCTION = ""


def assemble_record(passage: dict, gen_meta: dict) -> dict:
    rec = {
        "instruction":   INSTRUCTION.format(topic=gen_meta["topic"]),
        "subject":       gen_meta["topic"],
        "topic":         gen_meta["topic"],
        "content_type":  gen_meta["content_type"],
        "source":        passage["source"],
        "sub_corpus":    passage.get("sub_corpus", "unassigned"),
        "routing_score": passage.get("sub_score", 0),
        "book":          passage.get("book", passage.get("title", "unknown")),
        "author":        passage.get("author", "unknown"),
        "book_title":    passage.get("book_title", ""),
        "book_author":   passage.get("book_author", ""),
        "book_year":     passage.get("book_year", ""),
        "htid":          passage.get("htid", "") or "",
        "prov_method":   passage.get("prov_method", ""),
        "barcode":       passage.get("barcode", ""),
        "shard_row":     passage.get("shard_row", -1),
        "passage":       passage["text"],
        "pair": [
            {"role": "user",      "content": gen_meta["prompt"]},
            {"role": "assistant", "content": passage["text"]},
        ],
    }
    if passage.get("pg_id"):
        rec["pg_id"] = passage["pg_id"]
    return rec


# ---------------------------------------------------------------------------
# Post-generation pair-score filter
# ---------------------------------------------------------------------------

def _filter_by_pair_score(
    records: list[dict],
    target_n: int,
    min_score: int = 2,
) -> list[dict]:
    """
    Filter assembled records by pair_score, preserving per-sub-corpus proportions.

    Records without a pair_score (EB pre-formed pairs) are always kept and
    counted toward target_n before proportional filtering of scored records.

    Balance strategy: within each sub-corpus bucket, keep the highest-scoring
    records proportionally, so no single bucket crowds out others that
    inherently produce more contextual prose (fraught_social, period_commerce).
    Falls back to min_score-1 then 0 per bucket if quota cannot be filled.
    """
    scored   = [r for r in records if r.get("pair_score") is not None]
    unscored = [r for r in records if r.get("pair_score") is None]  # EB pre-formed pairs

    dist = Counter(r["pair_score"] for r in scored)
    print(f"\n  pair_score distribution ({len(scored):,} scored, {len(unscored):,} unscored/EB):")
    for sc in (3, 2, 1, 0):
        print(f"    score={sc}: {dist[sc]:,}")

    # EB pairs always pass through; reduce scored quota accordingly.
    scored_target = max(0, target_n - len(unscored))

    by_sub: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_sub[r.get("sub_corpus", "unassigned")].append(r)

    total_scored = len(scored)
    selected: list[dict] = list(unscored)
    seen: set[str] = {r["passage"][:80] for r in unscored}

    for sub, members in by_sub.items():
        quota = max(1, round(scored_target * len(members) / max(1, total_scored)))
        members.sort(key=lambda x: -(x.get("pair_score") or 0))
        for threshold in (min_score, min_score - 1, 0):
            candidates = [r for r in members if (r.get("pair_score") or 0) >= threshold]
            if candidates:
                break
        for r in candidates[:quota]:
            key = r["passage"][:80]
            if key not in seen:
                seen.add(key)
                selected.append(r)

    if len(selected) < target_n:
        remainder = [r for r in scored if r["passage"][:80] not in seen]
        remainder.sort(key=lambda x: -(x.get("pair_score") or 0))
        for r in remainder:
            if len(selected) >= target_n:
                break
            key = r["passage"][:80]
            if key not in seen:
                seen.add(key)
                selected.append(r)

    final_dist = Counter(r.get("pair_score", "n/a") for r in selected)
    print(f"  Kept {len(selected):,} / {len(records):,} records after pair_score filter.")
    print(f"  Score distribution of kept set: " +
          "  ".join(f"{sc}:{final_dist[sc]}" for sc in (3, 2, 1, 0, "n/a")))
    return selected


# ---------------------------------------------------------------------------
# Audit writer — human-readable passage inspection, no LM calls
# ---------------------------------------------------------------------------

def write_audit(
    passages: list[dict],
    n_per_sub: int,
    out_path: Path,
    rng: random.Random,
) -> None:
    """
    Write a human-readable audit of sampled passages to out_path.

    Groups passages by primary sub-corpus, samples n_per_sub from each,
    and formats each with its metadata header, quality flags, and full text.
    Intended for manual review of chunk quality before committing to a full
    LM-question-generation run.

    Quality flags shown per passage:
      starts_cleanly / ends_cleanly  — from passage_utils gates
      words                          — token count proxy
      [EB Q:]                        — for EB pre-formed pairs, shows the
                                       existing user question
      [WARN: mid-sentence start]     — when starts_cleanly fails
      [WARN: mid-sentence end]       — when ends_cleanly fails
    """
    from collections import defaultdict
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for p in passages:
        by_sub[p.get("sub_corpus", "unassigned")].append(p)

    sep_major = "=" * 80
    sep_minor = "-" * 60

    lines: list[str] = []
    lines.append(sep_major)
    lines.append(f"  gen_teacher_source4 — passage audit")
    lines.append(f"  {len(passages):,} passages sampled total  |  {n_per_sub} shown per sub-corpus")
    lines.append(sep_major)
    lines.append("")

    # Print in disposition-map priority order, then alphabetical remainder.
    priority = [
        "political_contemporary", "fraught_social", "science_technology",
        "religion_morality", "conduct_propriety", "travel_geography",
        "literary_criticism", "encyclopedic_factual", "history_biography",
        "thinking_machines", "period_communication", "period_dictionary",
        "period_commerce", "unassigned",
    ]
    ordered_subs = [s for s in priority if s in by_sub]
    ordered_subs += sorted(s for s in by_sub if s not in priority)

    for sub in ordered_subs:
        pool = by_sub[sub]
        sample = rng.sample(pool, min(n_per_sub, len(pool)))
        lines.append(sep_major)
        lines.append(f"  SUB-CORPUS: {sub}  ({len(pool):,} in sampled pool)")
        lines.append(sep_major)
        lines.append("")
        for i, p in enumerate(sample, 1):
            text  = p.get("text", "")
            words = len(text.split())
            sc    = starts_cleanly(text)
            ec    = ends_cleanly(text)
            subs_all = "|".join(p.get("sub_corpus_all") or [sub])
            lines.append(
                f"[{i}/{len(sample)}]  source={p.get('source','?')}"
                f"  score={p.get('sub_score', 0)}"
                f"  subs={subs_all}"
            )
            lines.append(
                f"  title : {p.get('book') or p.get('title', 'unknown')}"
            )
            lines.append(f"  author: {p.get('author', 'unknown')}")
            flags = []
            flags.append("starts_ok" if sc else "WARN:mid-start")
            flags.append("ends_ok"   if ec else "WARN:mid-end")
            flags.append(f"{words} words")
            lines.append(f"  flags : {' | '.join(flags)}")
            if p.get("user_text"):
                lines.append(f"  EB Q  : {p['user_text']}")
            lines.append("")
            lines.append(text)
            lines.append("")
            lines.append(sep_minor)
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nAudit written → {out_path}  ({len(ordered_subs)} sub-corpora)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher model Source 4 instruction-tuning pairs"
    )
    parser.add_argument("--count",       type=int, default=1000,
                        help="Target number of pairs (default: 1000)")
    parser.add_argument("--concurrency", type=int, default=25,
                        help="Max concurrent API calls (default: 25)")
    parser.add_argument("--backend", choices=["mlx", "anthropic"], default="mlx",
                        help="Backend for Stage 1 passage gate + question generation (default: mlx).")
    parser.add_argument("--model",       type=str, default=DEFAULT_MLX_MODEL)
    parser.add_argument("--judge-backend", choices=["mlx", "anthropic"], default=None,
                        help="Backend for Stage 2 pair scoring (default: same as --backend).")
    parser.add_argument("--judge-model",   type=str, default=None,
                        help="Model for Stage 2 judge (default: haiku if judge-backend=anthropic, else --model).")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma list of sources to load: "
                             f"{','.join(SOURCE_LOADERS)} (default: {','.join(DEFAULT_SOURCES)}). "
                             "e.g. --sources hathi,wellcome")
    parser.add_argument("--only-subcorpus", type=str, default=None,
                        help="Keep only passages routed to this sub-corpus "
                             "(e.g. thinking_machines). Focuses a run on one theme.")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Cap parquet shards read per shard-source (hathi/wellcome). "
                             "Lower = faster/smaller; default uses loader defaults (200/80).")
    parser.add_argument("--per-book", type=int, default=6,
                        help="Max passages any single volume contributes to the pool "
                             "(hathi/wellcome) so a few large works can't dominate. Default 6.")
    parser.add_argument("--subcorpus-weighting", action=argparse.BooleanOptionalAction, default=True,
                        help="Use cluster-derived sub-corpus quotas during passage sampling (default: on).")
    parser.add_argument("--disposition-map", type=Path, default=DEFAULT_DISPOSITION_MAP,
                        help="CSV map with disposition/sub_corpus columns for target weighting.")
    parser.add_argument("--mixed-source-ratio", type=float, default=0.60,
                        help="How much of F-cluster size to count toward source quotas (default: 0.60).")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max generation tokens for metadata model.")
    parser.add_argument("--oversample", type=float, default=1.0,
                        help="Over-generate by this factor, then filter by pair_score "
                             "(e.g. 3.0 = sample 3x target, generate+score all, keep best "
                             "target preserving sub-corpus proportions). Default: 1.0 "
                             "(no oversampling).")
    parser.add_argument("--preview",     action="store_true",
                        help="Sample passages and print stats only — no API calls")
    parser.add_argument("--audit", type=int, nargs="?", const=5, default=0,
                        metavar="N",
                        help="Write a human-readable passage audit (N per sub-corpus, "
                             "default 5). Implies --preview (no LM calls). "
                             "Use: --audit or --audit 10")
    parser.add_argument("--audit-file", type=Path,
                        default=Path("training_data/teacher/source4_audit.txt"),
                        help="Destination for --audit output (default: training_data/teacher/source4_audit.txt)")
    parser.add_argument("--output",      type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    # If --model was not explicitly set, pick a sensible default per backend
    if args.model == DEFAULT_MLX_MODEL and args.backend == "anthropic":
        args.model = "claude-haiku-4-5-20251001"

    # Resolve judge backend/model defaults
    if args.judge_backend is None:
        args.judge_backend = args.backend
    if args.judge_model is None:
        args.judge_model = ("claude-haiku-4-5-20251001"
                            if args.judge_backend == "anthropic"
                            else args.model)

    rng = random.Random(args.seed)

    sources = None
    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        bad = [s for s in sources if s not in SOURCE_LOADERS]
        if bad:
            print(f"ERROR: unknown --sources value(s): {', '.join(bad)}. "
                  f"Valid: {','.join(SOURCE_LOADERS)}")
            sys.exit(1)
    # When focusing on one theme, source-level quota weighting (tuned for the full
    # mix) is counter-productive — fall back to simple source weighting.
    if args.only_subcorpus and args.subcorpus_weighting:
        print(f"  (--only-subcorpus set: disabling --subcorpus-weighting)")
        args.subcorpus_weighting = False

    sample_n = args.count
    if args.oversample > 1.0:
        sample_n = int(args.count * args.oversample)
        print(f"Oversampling {sample_n:,} passages (x{args.oversample}) "
              f"for pair-score-filtered target of {args.count:,}...\n")
    else:
        print(f"Sampling {args.count:,} passages from Victorian corpus...\n")

    passages = sample_passages(
        sample_n,
        rng,
        subcorpus_weighting=args.subcorpus_weighting,
        disposition_map=args.disposition_map,
        mixed_source_ratio=args.mixed_source_ratio,
        sources=sources,
        only_subcorpus=args.only_subcorpus,
        max_shards=args.max_shards,
        per_book=args.per_book,
    )

    if args.audit:
        write_audit(passages, args.audit, args.audit_file, rng)
        return

    if args.preview:
        print(f"\n--- Sample passages ---")
        for p in rng.sample(passages, min(5, len(passages))):
            title = f" [{p.get('book', p.get('title', 'unknown'))[:50]}]"
            print(f"\n  [{p['source']}]{title}  author={p.get('author', 'unknown')}")
            print(f"  routed={p.get('sub_corpus', 'unassigned')}  score={p.get('sub_score', 0)}")
            print(f"  {p['text'][:300]}...")
        return

    # ------------------------------------------------------------------
    # Create backends and semaphore (semaphore always available for async paths)
    # ------------------------------------------------------------------
    semaphore = asyncio.Semaphore(args.concurrency)

    try:
        if args.backend == "mlx":
            gen_backend: MLXMetadataBackend | AnthropicMetadataBackend = MLXMetadataBackend(args.model)
        else:
            gen_backend = AnthropicMetadataBackend()
    except ImportError as exc:
        print(f"Backend not available: {exc}")
        sys.exit(1)

    if args.judge_backend == args.backend:
        judge_backend = gen_backend
    elif args.judge_backend == "mlx":
        try:
            judge_backend: MLXMetadataBackend | AnthropicMetadataBackend = MLXMetadataBackend(args.judge_model)
        except ImportError:
            print("mlx-lm not installed. Run: pip install mlx-lm")
            sys.exit(1)
    else:
        try:
            judge_backend = AnthropicMetadataBackend()
        except ImportError:
            print("anthropic not installed. Run: pip install anthropic")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Stage 1 — passage gate + prompt generation
    # ------------------------------------------------------------------
    print(f"\nStage 1: passage gate + prompt generation "
          f"(backend={args.backend} model={args.model})...")
    if args.judge_backend != args.backend or args.judge_model != args.model:
        print(f"Stage 2: pair scoring will use "
              f"judge-backend={args.judge_backend} model={args.judge_model}")

    generated: list[tuple[dict, dict]] = []  # (passage, gen_meta)
    gen_fail = gate_reject = 0
    t0 = time.time()

    if args.backend == "mlx":
        for i, passage in enumerate(passages, 1):
            if passage.get("user_text"):
                gen_meta = {
                    "passage_ok":   True,
                    "topic":        _eb_topic_from_question(passage["user_text"]),
                    "content_type": "expository",
                    "prompt":       passage["user_text"],
                }
                generated.append((passage, gen_meta))
            else:
                raw      = gen_backend.generate_prompt_raw(passage["text"], max_tokens=args.max_tokens)
                gen_meta = parse_generation_json(raw)
                if gen_meta is None:
                    gen_fail += 1
                elif not gen_meta["passage_ok"]:
                    gate_reject += 1
                else:
                    generated.append((passage, gen_meta))
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"  {i:>5,}/{len(passages):,}  passed={len(generated):,} "
                f"rejected={gate_reject} failed={gen_fail}  {rate:.2f}/s",
                end="\r",
            )
        print()

    else:  # anthropic
        eb_passages  = [p for p in passages if     p.get("user_text")]
        llm_passages = [p for p in passages if not p.get("user_text")]

        tasks    = [gen_backend.generate_prompt_raw(p["text"], model=args.model,
                        semaphore=semaphore, max_tokens=args.max_tokens)
                    for p in llm_passages]
        raw_list = await asyncio.gather(*tasks)

        for passage, raw in zip(llm_passages, raw_list):
            gen_meta = parse_generation_json(raw or "")
            if gen_meta is None:
                gen_fail += 1
            elif not gen_meta["passage_ok"]:
                gate_reject += 1
            else:
                generated.append((passage, gen_meta))

        for passage in eb_passages:
            gen_meta = {
                "passage_ok":   True,
                "topic":        _eb_topic_from_question(passage["user_text"]),
                "content_type": "expository",
                "prompt":       passage["user_text"],
            }
            generated.append((passage, gen_meta))

    print(f"\n  Stage 1 complete: {len(generated):,} passed gate  "
          f"({gate_reject:,} rejected, {gen_fail:,} parse failures)\n")

    # ------------------------------------------------------------------
    # Stage 2 — independent pair scoring
    # ------------------------------------------------------------------
    print(f"Stage 2: pair scoring ({len(generated):,} pairs)...")

    results: list[dict] = []
    judge_fail = 0
    t0 = time.time()

    if args.judge_backend == "mlx":
        for i, (passage, gen_meta) in enumerate(generated, 1):
            rec = assemble_record(passage, gen_meta)
            if passage.get("user_text"):
                results.append(rec)
            else:
                raw      = judge_backend.judge_pair_raw(gen_meta["prompt"], passage["text"], max_tokens=80)
                judgment = parse_judgment_json(raw)
                if judgment:
                    rec["pair_score"] = judgment["pair_score"]
                    rec["pair_note"]  = judgment["pair_note"]
                else:
                    judge_fail += 1
                results.append(rec)
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"  {i:>5,}/{len(generated):,}  ok={len(results):,} "
                f"judge_fail={judge_fail}  {rate:.2f}/s",
                end="\r",
            )
        print()

    else:  # anthropic judge
        llm_generated = [(p, m) for p, m in generated if not p.get("user_text")]
        eb_generated  = [(p, m) for p, m in generated if     p.get("user_text")]

        judge_tasks = [
            judge_backend.judge_pair_raw(m["prompt"], p["text"], model=args.judge_model,
                                         semaphore=semaphore, max_tokens=80)
            for p, m in llm_generated
        ]
        judge_raws = await asyncio.gather(*judge_tasks)

        for (passage, gen_meta), raw in zip(llm_generated, judge_raws):
            rec      = assemble_record(passage, gen_meta)
            judgment = parse_judgment_json(raw or "")
            if judgment:
                rec["pair_score"] = judgment["pair_score"]
                rec["pair_note"]  = judgment["pair_note"]
            else:
                judge_fail += 1
            results.append(rec)

        for passage, gen_meta in eb_generated:
            results.append(assemble_record(passage, gen_meta))

    print(f"\n  Stage 2 complete: {len(results):,} records  "
          f"({judge_fail:,} judge failures)\n")

    if args.oversample > 1.0 and results:
        results = _filter_by_pair_score(results, args.count)

    # Breakdown stats
    ct_dist  = Counter(r["content_type"] for r in results)
    src_dist = Counter(r["source"]       for r in results)
    sub_dist = Counter(r.get("sub_corpus", "unassigned") for r in results)
    print(f"\nCompleted: {len(results):,} pairs  "
          f"(gate_reject={gate_reject} gen_fail={gen_fail} judge_fail={judge_fail})")
    print("Content types:")
    for ct, n in ct_dist.most_common():
        print(f"  {ct:<16} {n:>5,}")
    print("Sources:")
    for src, n in src_dist.most_common():
        print(f"  {src:<16} {n:>5,}")
    print("Sub-corpora:")
    for sub, n in sub_dist.most_common():
        print(f"  {sub:<22} {n:>5,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWritten: {args.output}  ({len(results):,} records)")
    print(f"\nEach record contains:")
    print(f"  .instruction  — teacher model task prompt")
    print(f"  .pair         — [{{'role':'user',...}}, {{'role':'assistant',...}}]")
    print(f"                  (usable directly as victoriana SFT data)")


if __name__ == "__main__":
    asyncio.run(main())
