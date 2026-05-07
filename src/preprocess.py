# MODULI PËR PREPROCESIM TË TEKSTIT
# Ky skedër përmban funksione për pastrimin dhe normalizimin bazë të tekstit
# Para se të aplikohet TF-IDF, teksti duhet të pastrohet nga elementë jo-tematikë

import re  # Moduli për shprehje të rregullta (regex) për gjetje dhe zëvendësim teksti

# 1. KOMPILIMI I SHPREHJEVE TË RREGULLTA (REGEX)
# Kompilimi i regex-ve një herë në fillim është më efikas sesa të ri-kompilohet çdo herë

# Regex për zbulimin dhe heqjen e URL-ve:
# - https?:// → përputhet me "http://" ose "https://" ("s?" bën "s" opsionale)
# - \S+ → përputhet me një ose më shumë karaktere që NUK janë hapësirë (space)
# - | → OSE (OR)
# - www\.\S+ → përputhet me "www." e ndjekur nga karaktere jo-hapësirë
_url = re.compile(r"(https?://\S+|www\.\S+)")

# Regex për zbulimin dhe heqjen e tag-eve HTML:
# - <[^>]+> → përputhet me çdo gjë që fillon me "<", përmban një ose më shumë karaktere që NUK janë ">"
# dhe mbaron me ">"
_html = re.compile(r"<[^>]+>")

# Regex për heqjen e të gjithë karaktereve që NUK janë shkronja të vogla (a-z) ose hapësira:
# - [^a-z\s] → " ^ " brenda kllapave katrore kundërshton përputhjen
# - a-z → shkronja të vogla nga a në z
# - \s → hapësira (space, tab, newline)
_nonalpha = re.compile(r"[^a-z\s]")


# 2. FUNKSIONI KRYESOR PËR PASTRIMIN E TEKSTIT
def basic_clean(text: str) -> str:
    """Lightweight text normalization.

    - Lowercases

    - Strips URLs and HTML tags

    - Removes non-letters

    - Collapses whitespace

    """
   
    # 2.1 VALIDIMI I TIPIT TË HYRJES
    # Kontrollo nëse input është string; nëse jo, kthe string bosh
    # Kjo parandalon gabime nëse i jepet një numër, None, ose objekt tjetër
    if not isinstance(text, str):
        return ""  # Kthe string bosh për input jo-valid

    # 2.2 KONVERTIMI NË SHKRONJA TË VOJGLA (LOWERCASE)
    # Të gjitha shprehjet e rregullta më poshtë punojnë me shkronja të vogla
    # Kjo gjithashtu bën që modeli të jetë case-insensitive
    t = text.lower()

    # 2.3 HEQJA E URL-VE DHE TAG-EVE HTML
    # Zëvendëso çdo përputhje me regex-in _url me një hapësirë
    # _url.sub(" ", t) gjen të gjitha URL-të dhe i zëvendëson me " " (hapësirë)
    t = _url.sub(" ", t)
    
    # Zëvendëso çdo përputhje me regex-in _html me një hapësirë
    # Heq tag-et si <p>, <div>, <br/>, etj.
    t = _html.sub(" ", t)
    
    # 2.4 HEQJA E KARAKTEREVE JO-ALFABETIKE
    # Zëvendëso çdo karakter që NUK është shkronjë e vogël (a-z) ose hapësirë
    # Kjo heq numrat, simbolet, pikëpyetjet, presjet, etj.
    # P.sh.: "hello123!" → "hello "
    t = _nonalpha.sub(" ", t)
    
    # 2.5 ZVOGËLIMI I HAPËSIRAVE TË SHUMTA
    # \s+ përputhet me një ose më shumë hapësira (space, tab, newline)
    # re.sub(r"\s+", " ", t) zëvendëson të gjitha sekuencat e hapësirave me një hapësirë të vetme
    # .strip() heq hapësirat në fillim dhe në fund të stringut
    t = re.sub(r"\s+", " ", t).strip()
    
    return t
