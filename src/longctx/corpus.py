"""Corpus grammar: entity types, the register-entry text format, and the
parser that reads it back.

The generator WRITES entries with `render_entry`; the validator and the
mechanical oracle READ them with `parse_entry`. Keeping both here means the
solvability check is a check on the text, not on the generator's bookkeeping.

Entry grammar (one entry per line inside a chunk):

    <Name> — <Field>: <value>. <Field>: <value> [current since 2023; previously
    <old> from 2019 to 2023]. <Field>: <old> from 2019 to 2023, then <value>
    [current since 2023]. Notes: <free text>

Values never contain '.', '[', ']' or ';'. Names never contain '—'.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Entity types and the chain
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AttrSpec:
    name: str          # field label as written in the entry, e.g. "Annual budget"
    prefix: str        # "PJ-" for codes, "" for numbers
    lo: int
    hi: int
    thousands: bool    # format the number with thousands separators
    near_delta: int    # max |delta| for near-miss values


@dataclass(frozen=True)
class TypeSpec:
    name: str
    section: str
    link_field: str        # the field that names the next entity in the chain
    link_phrase: str       # "the lead of {}" — composes the question
    next_type: str
    attrs: tuple[AttrSpec, ...]
    near_miss_suffixes: tuple[str, ...]
    status_field: str
    status_values: tuple[str, ...]


TYPE_SPECS: dict[str, TypeSpec] = {
    "project": TypeSpec(
        name="project", section="Project Register", link_field="Lead",
        link_phrase="the lead of {}", next_type="person",
        attrs=(AttrSpec("Charge code", "PJ-", 1000, 9999, False, 40),
               AttrSpec("Cost centre", "CC-", 1000, 9999, False, 40)),
        near_miss_suffixes=(" II", " Bay", " Prime"),
        status_field="Status", status_values=("active", "on hold", "closing", "pilot"),
    ),
    "person": TypeSpec(
        name="person", section="Personnel Directory", link_field="Unit",
        link_phrase="the unit of {}", next_type="department",
        attrs=(AttrSpec("Badge number", "", 10000, 99999, False, 400),
               AttrSpec("Desk extension", "", 1000, 9999, False, 40)),
        near_miss_suffixes=(" Jr", "-Oakes", " Sr"),
        status_field="Grade", status_values=("G4", "G5", "G6", "G7"),
    ),
    "department": TypeSpec(
        name="department", section="Department Ledger", link_field="Site",
        link_phrase="the site of {}", next_type="facility",
        attrs=(AttrSpec("Annual budget", "", 1000000, 9999999, True, 60000),
               AttrSpec("Ledger code", "LG-", 1000, 9999, False, 40)),
        near_miss_suffixes=(" Support", " Field Ops", " Reserve"),
        status_field="Review cycle", status_values=("quarterly", "biannual", "annual"),
    ),
    "facility": TypeSpec(
        name="facility", section="Facilities Register", link_field="Region",
        link_phrase="the region of {}", next_type="region",
        attrs=(AttrSpec("Floor area", "", 10000, 99999, True, 800),
               AttrSpec("Asset tag", "FA-", 10000, 99999, False, 400)),
        near_miss_suffixes=(" Annex", " North", " Yard"),
        status_field="Access", status_values=("badge", "escort", "open", "restricted"),
    ),
    "region": TypeSpec(
        name="region", section="Regional Offices", link_field="Primary carrier",
        link_phrase="the primary carrier of {}", next_type="vendor",
        attrs=(AttrSpec("Postal prefix", "RX-", 1000, 9999, False, 40),
               AttrSpec("Tax code", "TX-", 10000, 99999, False, 400)),
        near_miss_suffixes=(" East", " Upper", " Coast"),
        status_field="Tier", status_values=("tier 1", "tier 2", "tier 3"),
    ),
    "vendor": TypeSpec(
        name="vendor", section="Vendor Roster", link_field="Account manager",
        link_phrase="the account manager of {}", next_type="person",
        attrs=(AttrSpec("Contract number", "CN-", 10000, 99999, False, 400),
               AttrSpec("Account number", "AC-", 100000, 999999, False, 4000)),
        near_miss_suffixes=(" Holdings", " Ltd", " Group"),
        status_field="Terms", status_values=("net 30", "net 45", "net 60", "prepaid"),
    ),
}

CHAIN_START = "project"


def chain_types(n_hops: int) -> list[str]:
    """Entity types visited by an n_hops chain, starting at the anchor type."""
    types = [CHAIN_START]
    while len(types) < n_hops:
        types.append(TYPE_SPECS[types[-1]].next_type)
    return types


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
COLORS = ["Amber", "Cobalt", "Crimson", "Ivory", "Jade", "Onyx", "Saffron", "Slate",
          "Umber", "Violet", "Copper", "Indigo", "Coral", "Ochre", "Pewter", "Sable",
          "Teal", "Russet", "Silver", "Maroon", "Azure", "Bronze", "Ebony", "Garnet"]
ANIMALS = ["Falcon", "Heron", "Otter", "Lynx", "Osprey", "Marten", "Kestrel", "Badger",
           "Wren", "Ibis", "Puffin", "Stoat", "Gannet", "Plover", "Merlin", "Vole",
           "Curlew", "Grouse", "Sable", "Tern", "Bittern", "Dunlin", "Ermine", "Fulmar"]
FIRST = ["Dana", "Omar", "Priya", "Lucas", "Mei", "Tomas", "Ingrid", "Kwame", "Sofia",
         "Ravi", "Elena", "Jonas", "Amara", "Felix", "Noor", "Hugo", "Leila", "Marco",
         "Yara", "Piotr", "Sana", "Bram", "Keiko", "Idris", "Greta", "Nikolai", "Aisha",
         "Rafael", "Wanda", "Emil", "Zara", "Callum", "Hana", "Otto", "Lena", "Samir",
         "Beatriz", "Ivo", "Maya", "Kofi"]
LAST = ["Ruiz", "Lind", "Natarajan", "Okafor", "Bergstrom", "Tanaka", "Moreau", "Haddad",
        "Novak", "Silva", "Petrov", "Achebe", "Fischer", "Rossi", "Ibarra", "Kowalski",
        "Duarte", "Mensah", "Larsen", "Quintero", "Vasquez", "Oyelaran", "Brandt", "Castel",
        "Dubois", "Eriksen", "Farouk", "Galanis", "Hoffman", "Iqbal", "Jansen", "Kimura",
        "Lombardi", "Madsen", "Nakamura", "Obi", "Pereira", "Radek", "Salazar", "Tremblay",
        "Ustinov", "Vieira", "Weber", "Xu", "Yilmaz", "Zamora", "Ahmadi", "Beckett", "Coelho",
        "Dimitrov", "Esposito", "Fontaine", "Garza", "Holm", "Ivanova", "Jimenez", "Kaur",
        "Lund", "Mbeki", "Nair"]
DEPT_ADJ = ["Coastal", "Northern", "Central", "Inland", "Western", "Eastern", "Upland",
            "Harbour", "Alpine", "Delta", "Prairie", "Lowland", "Highland", "Metro",
            "Riverside", "Southern", "Frontier", "Lakeside", "Valley", "Summit",
            "Border", "Island", "Basin", "Ridge", "Meadow"]
DEPT_NOUN = ["Logistics", "Payroll", "Compliance", "Procurement", "Analytics", "Facilities",
             "Outreach", "Archives", "Security", "Training", "Maintenance", "Planning",
             "Records", "Fleet", "Catering", "Design", "Licensing", "Surveying", "Transit",
             "Quality", "Inspection", "Dispatch", "Recovery", "Signals", "Warehousing"]
PLACE = ["Ashford", "Brindle", "Calder", "Dorset", "Elmwood", "Fenwick", "Glenmoor", "Hartley",
         "Ironbridge", "Juniper", "Kestwick", "Larkhill", "Marlow", "Norwood", "Oakridge",
         "Penrose", "Quarry", "Redcliff", "Stanmore", "Thornbury", "Underhill", "Verdon",
         "Westmere", "Yarrow", "Zeller", "Bramford", "Crestline", "Dunmore", "Eastvale",
         "Foxglove", "Greyfell", "Holloway", "Ivydale", "Kingsmill", "Longacre", "Millbrook",
         "Northgate", "Orchard", "Pinehurst", "Rookwood"]
FACILITY_KIND = ["Depot", "Plant", "Hub", "Works", "Terminal", "Station", "Mill", "Yardworks"]
TERRAIN = ["Ridge", "Marsh", "Plateau", "Shore", "Fells", "Downs", "Heath", "Moor", "Hollow",
           "Reach", "Strand", "Vale", "Wold", "Cape", "Bluff", "Glade", "Sound", "Tarn",
           "Spur", "Knoll", "Combe", "Firth", "Weald", "Scarp", "Dale"]
VENDOR_SUFFIX = ["Freight", "Haulage", "Carriers", "Transport", "Shipping", "Logistics",
                 "Couriers", "Movers"]

NOISE_SECTIONS = ["Meeting Minutes", "Maintenance Log", "Travel Ledger", "Incident Reports",
                  "Procurement Notes", "Site Circulars", "Safety Bulletins", "Canteen Notices"]

# Filler: no proper nouns, no digits except a 4-digit year, no entity names.
FILLER_TEMPLATES = [
    "The {group} asked for the {item} to be checked again before the {when} review.",
    "A revised {item} schedule was circulated to the {group} {when}.",
    "The {group} noted that the {item} inventory had been reconciled without discrepancy.",
    "Access to the {place} is limited to the {group} until the {item} audit closes.",
    "The {item} handover was completed {when} and signed off by the {group}.",
    "Feedback from the {group} on the {item} pilot remains under consideration.",
    "The {place} will be closed for {item} work during the {when} shift.",
    "Routine {item} checks continue on the usual rota agreed with the {group}.",
    "The {group} reported no outstanding actions from the {when} {item} walkthrough.",
    "A summary of {item} usage at the {place} is attached for the {group}.",
    "The {group} has deferred the {item} decision to the {when} session.",
    "Contractors at the {place} must sign the {item} register {when}.",
    "The {item} allowance will be reviewed alongside the {group} plan {when}.",
    "Minor wear was found during the {when} {item} inspection of the {place}.",
    "The {group} thanked staff at the {place} for supporting the {item} drive.",
    "Two {item} requests from the {group} were approved {when} without changes.",
    "The {place} notice board now lists the {item} contacts for the {group}.",
    "An interim {item} arrangement stays in place until the {group} confirms otherwise.",
]
FILLER_GROUP = ["committee", "night shift", "review board", "works council", "site office",
                "duty team", "planning cell", "steering group", "audit desk", "front desk"]
FILLER_ITEM = ["lighting", "signage", "parking", "recycling", "keycard", "boiler", "pallet",
               "uniform", "ventilation", "drainage", "storage", "catering", "fire door",
               "forklift", "roofing", "visitor", "first aid", "printing", "archive", "fencing"]
FILLER_WHEN = ["on Monday", "last week", "in the spring", "this quarter", "on Thursday",
               "after the holiday", "at month end", "yesterday", "in the autumn", "next week"]
FILLER_PLACE = ["east wing", "loading bay", "north gate", "annex", "car park", "main hall",
                "rear yard", "lower store", "canteen", "workshop", "reception", "boiler room"]


# --------------------------------------------------------------------------- #
# Entity records
# --------------------------------------------------------------------------- #
@dataclass
class Entity:
    type: str
    name: str
    link: str                      # current next-entity name
    attrs: dict[str, str]          # attr name -> current value string
    status: str
    prev_link: str | None = None   # superseded link (a distractor)
    prev_attrs: dict[str, str] = field(default_factory=dict)  # superseded attr values
    notes: list[str] = field(default_factory=list)
    is_near_miss_of: str | None = None


def format_value(spec: AttrSpec, n: int) -> str:
    return f"{spec.prefix}{n:,}" if spec.thousands else f"{spec.prefix}{n}"


def value_number(spec: AttrSpec, value: str) -> int:
    return int(value[len(spec.prefix):].replace(",", ""))


def render_field(field_name: str, value: str, prev: str | None, style: int, y1: int, y2: int) -> str:
    """Render one field. Two superseded styles so no positional rule
    ("the value after 'then'" / "the value before the bracket") is universal."""
    if prev is None:
        return f"{field_name}: {value}."
    if style == 0:
        return f"{field_name}: {value} [current since {y2}; previously {prev} from {y1} to {y2}]."
    return f"{field_name}: {prev} from {y1} to {y2}, then {value} [current since {y2}]."


def render_entry(e: Entity, style: int, y1: int, y2: int) -> str:
    spec = TYPE_SPECS[e.type]
    parts = [render_field(spec.link_field, e.link, e.prev_link, style, y1, y2)]
    for a in spec.attrs:
        parts.append(render_field(a.name, e.attrs[a.name], e.prev_attrs.get(a.name), style, y1, y2))
    parts.append(f"{spec.status_field}: {e.status}.")
    text = f"{e.name} — " + " ".join(parts)
    if e.notes:
        text += " Notes: " + " ".join(e.notes)
    return text


_FIELD_RE = re.compile(
    r"(?P<field>[A-Z][A-Za-z ]+?): (?P<body>[^.\[\]]+?)(?: \[(?P<hist>[^\]]*)\])?\."
)
_STYLE0_HIST = re.compile(r"current since \d{4}; previously (?P<prev>.+?) from \d{4} to \d{4}")
_STYLE1_BODY = re.compile(r"(?P<prev>.+?) from \d{4} to \d{4}, then (?P<cur>.+)")


@dataclass
class ParsedEntry:
    name: str
    fields: dict[str, str]           # field -> current value
    previous: dict[str, str]         # field -> superseded value (if any)


def parse_entry(line: str) -> ParsedEntry | None:
    """Read an entry back from its text. Returns None for non-entry lines."""
    if " — " not in line:
        return None
    name, rest = line.split(" — ", 1)
    body = rest.split(" Notes: ", 1)[0]
    fields: dict[str, str] = {}
    previous: dict[str, str] = {}
    for m in _FIELD_RE.finditer(body):
        f, val, hist = m.group("field"), m.group("body").strip(), m.group("hist")
        if hist is not None:
            if hm := _STYLE0_HIST.search(hist):
                previous[f] = hm.group("prev").strip()
                fields[f] = val
            else:  # style 1: "<prev> from y to y, then <cur>" with "[current since y]"
                bm = _STYLE1_BODY.match(val)
                if bm:
                    previous[f] = bm.group("prev").strip()
                    fields[f] = bm.group("cur").strip()
                else:
                    fields[f] = val
        else:
            fields[f] = val
    if not fields:
        return None
    return ParsedEntry(name=name.strip(), fields=fields, previous=previous)


def parse_chunk_entries(chunk_text: str) -> list[ParsedEntry]:
    out = []
    for line in chunk_text.splitlines():
        p = parse_entry(line)
        if p is not None:
            out.append(p)
    return out


def range_label(first: str, last: str) -> str:
    """Alphabetical range of a chunk, from its first to its last entry. Full
    names, so the map lets a reader decide exactly which chunk holds a name."""
    return f"{first} – {last}" if first != last else first


def compose_question(n_hops: int, anchor: str, answer_attr: str) -> str:
    types = chain_types(n_hops)
    phrase = f"project {anchor}"
    for t in types[:-1]:
        phrase = TYPE_SPECS[t].link_phrase.format(phrase)
    return f"What is the {answer_attr.lower()} of {phrase}?"
