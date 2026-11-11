import csv
import io
import json

from bs4 import BeautifulSoup
import pytest

from nemotron_parse import PostprocessOptions, postprocess_document, render_markdown
from nemotron_parse.postprocessing import (
    extract_classes_bboxes,
    transform_bbox_to_original,
    postprocess_text,
    convert_mmd_to_plain_text_ours,
)
from nemotron_parse import tables


def block(text, cls="Text", box=(0.1, 0.1, 0.9, 0.9)):
    return f"<x_{box[0]}><y_{box[1]}>{text}<x_{box[2]}><y_{box[3]}><class_{cls}>"


def tabular(*rows, cols="cc"):
    return (
        "\\begin{tabular}{"
        + cols
        + "}\n"
        + "\n".join(row + r" \\" for row in rows)
        + "\n\\end{tabular}"
    )


def test_multiline_cell_and_trailing_empty_preserved():
    text = tabular(r"\thead{Name} & \thead{Note}", r"A & first\newline second", "B & ")
    html = postprocess_text(text, "Table", table_format="HTML")
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    assert len(rows) == 3
    assert [len(row.find_all(["th", "td"])) for row in rows] == [2, 2, 2]
    assert rows[1].find_all("td")[1].find("br")
    assert rows[2].find_all("td")[1].get_text() == ""


def test_rich_inline_and_literal_escapes():
    raw = tabular(
        r"\thead{Item} & \thead{Note}",
        r"A & \textstrong{Bold} \textsuperscript{2} \# \textasciitilde{} \textasciicircum{}",
    )
    html = postprocess_text(raw, "Table", table_format="HTML")
    assert "<b>Bold</b>" in html and "<sup>2</sup>" in html
    assert "# ~ ^" in html
    assert r"\thead" not in html


def test_data_rows_not_promoted_to_headers():
    raw = tabular("A & 10", "B & 20")
    html = postprocess_text(raw, "Table", table_format="HTML")
    assert "<th" not in html
    records = json.loads(postprocess_text(raw, "Table", table_format="json"))
    assert [list(row.values()) for row in records] == [["A", "10"], ["B", "20"]]
    assert records[0] == {"column_0": "A", "column_1": "10"}


def test_bold_header_inference_configurable():
    raw = tabular("**Name** & **Value**", "A & 10")
    assert "<th" in postprocess_text(raw, "Table", table_format="HTML")
    assert "<th" not in postprocess_text(
        raw,
        "Table",
        table_format="HTML",
        options=PostprocessOptions(infer_headers=False),
    )


def test_spans_and_hierarchical_export():
    raw = tabular(r"\thead{\multirow{2}{*}{Region}} & \thead{Value}", r" & A", "B & 10")
    # Use the supported outer-span/inner-header native syntax.
    raw = raw.replace(
        r"\thead{\multirow{2}{*}{Region}}", r"\multirow{2}{*}{\thead{Region}}"
    )
    html = postprocess_text(
        raw,
        "Table",
        table_format="HTML",
        options=PostprocessOptions(infer_headers=False),
    )
    soup = BeautifulSoup(html, "html.parser")
    assert soup.find("th")["rowspan"] == "2"
    obj = json.loads(tables.convert_html_table_to_json_hierarchical(html))
    assert obj["headers"][0][0]["rowspan"] == 2
    assert len(obj["data"]) == 2


def test_rowspan_grid_does_not_drop_later_columns():
    html = '<table><tr><td rowspan="2">A</td></tr><tr><td>B</td><td>C</td></tr></table>'
    records = json.loads(tables.convert_html_table_to_json(html))
    assert list(records[1].values()) == ["A", "B", "C"]


@pytest.mark.parametrize(
    "names",
    [["A", "A", "A_1"], ["column", "", "column_1", ""], ["x", "x_1", "x", "x_2", "x"]],
)
def test_collision_free_json_names(names):
    html = "<table><tr>" + "".join("<th>" + n + "</th>" for n in names) + "</tr><tr>"
    values = list(map(str, range(len(names))))
    html += "".join("<td>" + v + "</td>" for v in values) + "</tr></table>"
    record = json.loads(tables.convert_html_table_to_json(html))[0]
    assert len(record) == len(names)
    assert list(record.values()) == values


def test_csv_and_markdown_keep_headerless_first_row():
    raw = tabular("A & 10", "B & 20")
    rows = list(
        csv.reader(io.StringIO(postprocess_text(raw, "Table", table_format="csv")))
    )
    assert rows == [["column_0", "column_1"], ["A", "10"], ["B", "20"]]
    md = postprocess_text(raw, "Table", table_format="markdown")
    assert "| A | 10 |" in md and "| B | 20 |" in md


@pytest.mark.parametrize(
    "source, expected",
    [
        ("<sup>2</sup> and <sub>i</sub>", "^{2} and _{i}"),
        ("C# developer", "C# developer"),
        ("x * y * z", "x * y * z"),
        ("# Heading\nC# developer", "Heading\nC# developer"),
        ("`C# **code**` and $x_1 * y$", "`C# **code**` and $x_1 * y$"),
    ],
)
def test_plain_text(source, expected):
    assert convert_mmd_to_plain_text_ours(source) == expected


def test_prefix_repair_and_exact_duplicate_cleanup():
    raw = "<x_0.01><y_0.01><class_Title>" + block("Hello") * 2
    result = postprocess_document(raw)
    assert result["markdown"] == "Hello" and len(result["blocks"]) == 1
    assert "<class_Title>" not in result["blocks"][0]["text"]


def test_distinct_coordinates_or_code_whitespace_not_deduplicated():
    raw = block("Hello") + block("Hello", box=(0.1, 0.2, 0.9, 1.0))
    assert len(postprocess_document(raw)["blocks"]) == 2
    raw = block("a  b", "Code") + block("a b", "Code")
    assert len(postprocess_document(raw)["blocks"]) == 2


def test_page_number_and_formula_preserved():
    result = postprocess_document(
        block("1", "Page-number") + block("$a+b$", "Inline-formula")
    )
    assert [b["class"] for b in result["blocks"]] == ["Page-number", "Inline-formula"]
    assert "1" in result["markdown"] and "$a+b$" in result["markdown"]


def test_text_cleanup_protects_code_and_math():
    assert postprocess_text("inter\u00adnational<tbc>") == "international"
    assert postprocess_text("• item", "List-item") == "- item"
    assert postprocess_text("a\u00adb", "Code") == "a\u00adb"
    assert postprocess_text("$a\u00adb$") == "$a\u00adb$"


@pytest.mark.parametrize(
    "box", [(float("nan"), 0, 1, 1), (0, 0, float("inf"), 1), (0, 0, 1)]
)
def test_invalid_boxes_rejected(box):
    with pytest.raises(ValueError):
        transform_bbox_to_original(box, 100, 100)


def test_reversed_and_clipped_boxes():
    assert transform_bbox_to_original((0.9, 1.2, -0.1, 0.2), 100, 200) == (
        0.0,
        40.0,
        90.0,
        200.0,
    )


def test_chart_presentation_controls():
    chart = "<chart>\ntitle: Sales\nsubtitle: FY2026\n| Item | Revenue |\n| --- | --- |\n| A | 32 (16.6%) |\n| B | 48 (25%) |\n</chart>"
    raw = block(chart, "Chart")
    ordinary = render_markdown(raw)
    assert "Sales — FY2026" in ordinary and "32 (16.6%)" in ordinary
    assert ">Percent<" not in ordinary
    split = render_markdown(
        raw, options=PostprocessOptions(split_chart_value_percent=True)
    )
    assert ">Percent<" in split and "<td>32</td>" in split and "<td>16.6%</td>" in split
    assert (
        render_markdown(raw, options=PostprocessOptions(render_charts=False)) == chart
    )


def test_empty_chart_title_does_not_consume_metadata():
    chart = (
        "<chart>\ntitle:\nsubtitle: Keep\n| A | B |\n| --- | --- |\n| x | 1 |\n</chart>"
    )
    assert "subtitle: Keep" in render_markdown(block(chart, "Chart"))


def test_headings_and_original_text_exposed():
    document = postprocess_document(
        block("Title", "Title") + block("Section", "Section-header"),
        image_size=(100, 200),
    )
    assert document["markdown"] == "# Title\n\n## Section"
    assert document["blocks"][0]["text"] == "Title"
    assert document["blocks"][0]["bbox_pixels"] == [10.0, 20.0, 90.0, 180.0]


def test_existing_top_level_imports():
    import postprocessing
    import latex2html

    assert postprocessing.postprocess_text("text") == "text"
    assert callable(latex2html.convert_html_table_to_json_hierarchical)


def test_unknown_class_is_not_dropped():
    assert render_markdown(block("Future content", "Future-role")) == "Future content"


def test_presentation_caption_relocation_is_opt_in():
    raw = block("Sales figure", "Caption", (0.1, 0.05, 0.9, 0.09)) + block(
        "<chart>\n| Item | Value |\n| --- | --- |\n| A | 1 |\n</chart>",
        "Chart",
        (0.1, 0.1, 0.9, 0.9),
    )
    default = render_markdown(raw)
    assert default.startswith("Sales figure\n\n")
    moved = render_markdown(
        raw, options=PostprocessOptions(relocate_chart_captions=True)
    )
    assert "<caption>Sales figure</caption>" in moved
    assert moved.count("Sales figure") == 1


def test_body_spans_exposed_in_hierarchical_json():
    html = (
        '<table><tr><th>A</th><th>B</th></tr><tr><td colspan="2">Both</td></tr></table>'
    )
    obj = json.loads(tables.convert_html_table_to_json_hierarchical(html))
    assert obj["body_cells"][0][0]["colspan"] == 2
    assert obj["data"] == [["Both"]]


def test_malformed_table_is_preserved_instead_of_lossy_fallback():
    raw = tabular(r"\multirow{2}{*}{\thead{missing brace} & x", " & y")
    result = render_markdown(block(raw, "Table"))
    assert raw in result


def test_document_preserves_malformed_table_and_reports_warning():
    raw = tabular(r"\multirow{2}{*}{\thead{missing brace} & x", " & y")
    document = postprocess_document(block(raw, "Table"))
    assert document["blocks"][0]["content"] == raw
    assert document["warnings"][0]["block_index"] == 0


@pytest.mark.parametrize("role", ["Code", "Formula"])
def test_literal_tabular_in_code_or_formula_is_not_converted(role):
    text = tabular("A & 10", "B & 20")
    assert render_markdown(block(text, role)) == text


def test_prose_discussing_tabular_is_not_reinterpreted_as_table():
    text = "The LaTeX example uses " + tabular("A & 10", "B & 20") + " in its source."
    assert render_markdown(block(text, "Text")) == text
