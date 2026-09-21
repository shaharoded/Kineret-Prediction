"""
Compile the Mediator's own event definitions into executable rules.

The Mediator turned `mediator_input.csv` into `mediator_output.csv`. Any
disagreement between the two is therefore not a source conflict — it is this
package failing to reproduce a rule the knowledge base states exactly. So the
rules are not transcribed by hand here; they are **parsed out of the knowledge
base XML**, which is the only copy that cannot drift from what actually ran.

    core/knowledge-base/events/HYPERGLYCEMIA.xml
      derived-from : GLUCOSE_MEASURE (A1), STEADY_GLUCOSE_MEASURE_HIGH (S1)
      rule "or"    : A1 >= 250          -> one extreme reading
                     S1 >= 180          -> sustained: this AND the previous
                                           reading within 24 h were both >= 180

A `parameterized-raw-concept` is a transformation of its parent, so each is read
too and reduced to the operation it performs:

| XML function | Meaning | Executed as |
|---|---|---|
| `id_if_thresh_met` | pass the value through only if the preceding reading within `good-before` satisfies the gate | `sustained` |
| `div` with a static `before` parameter on its own parent | value ÷ the admission's first reading | `ratio_to_baseline` |

Anything else raises rather than being approximated: a rule this package cannot
reproduce exactly must be visible, not silently half-applied.

Each `raw-concept` also declares the ConceptNames it accepts. Those are compiled
too, because they are the knowledge base's own answer to "how is this spelled in
the input" — `KIDNEY_COMPLICATION` accepts an observation code filed under
`KIDNEY_COMPLICATION_OBS`, and `CARDIO-VASCULAR_DISORDER` accepts the ETL's
unhyphenated spelling.

Compile once with `python -m kineret.mediator_rules <kb-path>`; the result ships
as `kineret/config/event_rules.json` so the VM does not need the Mediator repo.
"""

import glob
import json
import os
import xml.etree.ElementTree as ET

# Transformations this module knows how to execute.
SUSTAINED = "sustained"
RATIO_TO_BASELINE = "ratio_to_baseline"


def _constraints(attr_el) -> list:
    """
    Purpose: Read the `<allowed-value>` constraints under one rule attribute.
    Method:  Mirrors the engine's own parser: `equal` is an exact match, `min`
             and `max` are inclusive bounds, and both together form a range.

    Args:
        attr_el (Element): An `<attribute>` element inside `<abstraction-rules>`.

    Returns:
        list[dict]: Constraint dicts, OR-ed together by the engine.
    """
    out = []
    for av in attr_el.findall("allowed-value"):
        equal = av.attrib.get("equal")
        low, high = av.attrib.get("min"), av.attrib.get("max")
        if equal is not None:
            out.append({"type": "equal", "value": equal})
        elif low is not None and high is not None:
            out.append({"type": "range", "min": float(low), "max": float(high)})
        elif low is not None:
            out.append({"type": "min", "value": float(low)})
        elif high is not None:
            out.append({"type": "max", "value": float(high)})
    return out


def _parse_parameterized(path: str) -> dict:
    """
    Purpose: Reduce one parameterized-raw-concept to the operation it performs.
    Method:  Read its parent, its single parameter and its function chain, then
             map the chain onto a transformation this package can execute.

    Args:
        path (str): Path to the concept's XML.

    Returns:
        dict: {'name', 'parent', 'transform': {...}}.

    Raises:
        ValueError: The concept uses a function or shape not supported here.
    """
    root = ET.parse(path).getroot()
    name = root.attrib["name"]
    parent_el = root.find("derived-from")
    parent = parent_el.attrib["name"] if parent_el is not None else None

    parameters = {}
    for param_el in root.findall("./parameters/parameter"):
        parameters[param_el.attrib["ref"]] = {
            "name": param_el.attrib["name"],
            "how": param_el.attrib.get("how", "all"),
            "dynamic": param_el.attrib.get("dynamic", "true").lower() == "true",
            "good_before": param_el.attrib.get("good-before"),
        }

    functions = root.findall("./functions/function")
    if len(functions) != 1:
        raise ValueError(f"{name}: expected exactly one function, got {len(functions)}")
    function = functions[0]
    func_name = function.attrib["name"]
    literals = [lit.attrib["value"] for lit in function.findall("literal")]
    refs = [p.attrib["ref"] for p in function.findall("parameter")]
    param = parameters.get(refs[0]) if refs else None

    if func_name == "id_if_thresh_met":
        # value passes only if the preceding reading satisfies the gate
        threshold = float(literals[0]) if literals else 180.0
        op = literals[1] if len(literals) > 1 else "ge"
        if param is None or not param["dynamic"] or param["how"] != "before":
            raise ValueError(f"{name}: id_if_thresh_met needs a dynamic 'before' parameter")
        return {"name": name, "parent": parent, "transform": {
            "kind": SUSTAINED, "gate_op": op, "gate_value": threshold,
            "window": param["good_before"]}}

    if func_name == "div":
        # value divided by a static 'before' parameter on its own parent = the
        # admission's first reading, which the engine also removes from output.
        if param is None or param["dynamic"] or param["how"] != "before":
            raise ValueError(f"{name}: div is only supported as ratio-to-baseline")
        if param["name"] != parent:
            raise ValueError(f"{name}: div parameter must be the parent concept")
        return {"name": name, "parent": parent,
                "transform": {"kind": RATIO_TO_BASELINE}}

    raise ValueError(
        f"{name}: function {func_name!r} is not supported by kineret.raw_events. "
        f"Add it there rather than approximating the rule.")


def parse_knowledge_base(kb_path: str, events=None) -> dict:
    """
    Purpose: Compile the knowledge base into executable event rules.
    Method:  Read every `raw-concept` for its numeric validity range (the engine
             drops out-of-range readings before any rule sees them), every
             `parameterized-raw-concept` for its transformation, and every
             `event` for its `derived-from` map and abstraction rules. Each rule
             attribute becomes a clause naming a source concept, an optional
             transformation, and the constraints to apply.

    Args:
        kb_path (str):       Path to `core/knowledge-base`.
        events  (list|None): Restrict to these event names.

    Returns:
        dict: {'clippers': {...}, 'events': {...}}.
    """
    # Keyed by the raw CONCEPT, not by the attribute name. Several concepts
    # declare an attribute called `GLUCOSE_MEASURE` -- HIGH_GLUCOSE_IND bounds
    # it at >= 180, LOW_GLUCOSE_IND at <= 70 -- and keying on the attribute lets
    # an indicator's threshold masquerade as the measurement's validity range,
    # silently clipping away every reading a rule was meant to fire on.
    clippers, attributes = {}, {}
    for path in glob.glob(os.path.join(kb_path, "raw-concepts", "*.xml")):
        root = ET.parse(path).getroot()
        concept = root.attrib["name"]

        # A raw concept declares the ConceptNames it accepts. That is how
        # KIDNEY_COMPLICATION admits an observation code under
        # KIDNEY_COMPLICATION_OBS, and how CARDIO-VASCULAR_DISORDER admits the
        # ETL's unhyphenated spelling. A rule looking only for the concept's own
        # name silently misses every row filed under a sibling attribute.
        declared = [a.attrib["name"] for a in root.findall("./attributes/attribute")]
        attributes[concept] = declared or [concept]

        for attr in root.findall("./attributes/attribute"):
            if attr.attrib.get("name") != concept:
                continue                 # secondary attribute, not the range
            for av in attr.findall("./numeric-allowed-values/allowed-value"):
                low, high = av.attrib.get("min"), av.attrib.get("max")
                clippers[concept] = [
                    float(low) if low is not None else None,
                    float(high) if high is not None else None]

    parameterized = {}
    for path in glob.glob(os.path.join(kb_path, "parameterized-raw-concepts", "*.xml")):
        try:
            spec = _parse_parameterized(path)
        except ValueError:
            continue                     # unsupported ones are simply not offered
        parameterized[spec["name"]] = spec

    compiled = {}
    for path in glob.glob(os.path.join(kb_path, "events", "*.xml")):
        root = ET.parse(path).getroot()
        name = root.attrib["name"]
        if events and name not in events:
            continue

        derived = {}
        for attr in root.findall("./derived-from/attribute"):
            ref = attr.attrib.get("ref")
            if ref:
                derived[ref] = attr.attrib["name"]

        rules = []
        for rule_el in root.findall("./abstraction-rules/rule"):
            clauses, unsupported = [], []
            for attr_el in rule_el.findall("attribute"):
                ref = attr_el.attrib.get("ref")
                source = derived.get(ref)
                if source is None:
                    continue
                spec = parameterized.get(source)
                if spec is not None:
                    clause = {"source": spec["parent"], "via": source,
                              "transform": spec["transform"]}
                elif source in parameterized or "_REL_" in source or "STEADY_" in source:
                    unsupported.append(source)
                    continue
                else:
                    clause = {"source": source, "via": None, "transform": None}
                clause["constraints"] = _constraints(attr_el)
                clauses.append(clause)
            if clauses:
                rules.append({"value": rule_el.attrib.get("value", "True"),
                              "operator": rule_el.attrib.get("operator", "or"),
                              "clauses": clauses,
                              "unsupported": unsupported})
        if rules:
            compiled[name] = {"rules": rules}

    return {"clippers": clippers, "attributes": attributes, "events": compiled}


def main():
    """Purpose: Compile a knowledge base to JSON for shipping."""
    import argparse
    from kineret.config import paths

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kb_path", help="path to core/knowledge-base")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(paths.TAK_REPO_PATH), "event_rules.json"))
    args = parser.parse_args()

    compiled = parse_knowledge_base(args.kb_path)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(compiled, handle, indent=1, sort_keys=True)
    print(f"[rules] {len(compiled['events'])} event(s), "
          f"{len(compiled['clippers'])} clipper(s), "
          f"{len(compiled['attributes'])} concept(s) -> {args.out}")
    for name, spec in sorted(compiled["events"].items()):
        for rule in spec["rules"]:
            parts = []
            for clause in rule["clauses"]:
                kind = (clause["transform"] or {}).get("kind", "value")
                bounds = ", ".join(
                    f"{c['type']}={c.get('value', (c.get('min'), c.get('max')))}"
                    for c in clause["constraints"])
                parts.append(f"{clause['source']}[{kind}] {bounds}")
            print(f"  {name:<32} {rule['operator']}( " + " | ".join(parts) + " )")
            if rule["unsupported"]:
                print(f"      UNSUPPORTED clause(s): {rule['unsupported']}")


if __name__ == "__main__":
    main()
