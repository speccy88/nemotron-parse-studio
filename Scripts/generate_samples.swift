#!/usr/bin/env swift
import Foundation
import CoreGraphics
import CoreText
import ImageIO
import UniformTypeIdentifiers

let output = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "Samples", isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let width = 612, height = 792
func text(_ string: String, _ x: CGFloat, _ y: CGFloat, _ size: CGFloat, _ c: CGContext) {
    let font = CTFontCreateWithName("Helvetica" as CFString, size, nil)
    let line = CTLineCreateWithAttributedString(NSAttributedString(string: string, attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font, NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0.08, alpha: 1)]))
    c.textPosition = CGPoint(x: x, y: y); CTLineDraw(line, c)
}
func draw(_ c: CGContext, page: Int) {
    c.setFillColor(CGColor(gray: 1, alpha: 1)); c.fill(CGRect(x: 0, y: 0, width: width, height: height))
    text("Nemotron Parse Studio", 48, 728, 28, c)
    text("Document verification fixture · Page \(page)", 48, 697, 13, c)
    text("Multilingual text", 48, 655, 18, c)
    text("English: Accurate structure matters.", 48, 630, 12, c)
    text("Français : Bonjour, Montréal !  Español: información.", 48, 609, 12, c)
    text("日本語：表と文章。 العربية: مرحبا  Ελληνικά: δεδομένα", 48, 588, 12, c)
    text("Merged-cell table", 48, 545, 18, c)
    c.setStrokeColor(CGColor(gray: 0.3, alpha: 1)); c.setLineWidth(1)
    c.stroke(CGRect(x: 48, y: 348, width: 516, height: 168))
    for y in [390, 432, 474] { c.move(to: CGPoint(x: 48, y: y)); c.addLine(to: CGPoint(x: 564, y: y)); c.strokePath() }
    c.move(to: CGPoint(x: 220, y: 348)); c.addLine(to: CGPoint(x: 220, y: 516)); c.strokePath()
    c.move(to: CGPoint(x: 392, y: 348)); c.addLine(to: CGPoint(x: 392, y: 474)); c.strokePath()
    text("Product", 61, 490, 13, c); text("Quarterly revenue (USD)", 237, 490, 13, c)
    text("", 61, 448, 12, c); text("Q1", 237, 448, 12, c); text("Q2", 409, 448, 12, c)
    text("Alpha", 61, 406, 12, c); text("1,250.00", 237, 406, 12, c); text("1,500.25", 409, 406, 12, c)
    text("Beta", 61, 364, 12, c); text("0.00", 237, 364, 12, c)
    text("Chart: Share of responses", 48, 312, 18, c)
    c.setFillColor(CGColor(srgbRed: 0.15, green: 0.45, blue: 0.8, alpha: 1)); c.fill(CGRect(x: 100, y: 195, width: 220, height: 30))
    c.setFillColor(CGColor(srgbRed: 0.15, green: 0.7, blue: 0.5, alpha: 1)); c.fill(CGRect(x: 100, y: 245, width: 330, height: 30))
    text("Yes", 48, 254, 12, c); text("No", 48, 204, 12, c); text("60% (120)", 440, 254, 12, c); text("40% (80)", 330, 204, 12, c)
    text("Figure 1. Percentages and original counts; total n = 200.", 48, 170, 11, c)
    text("Code: for i in range(3): print(i)", 48, 120, 12, c)
    text("Mathematics: E = mc²; ∑ᵢ xᵢ = 1; α + β = γ", 48, 95, 12, c)
    text("Empty cells are intentional. All data is synthetic.", 48, 48, 10, c)
}
func raster(page: Int) -> CGImage {
    let c = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    draw(c, page: page); return c.makeImage()!
}
func imageFile(_ name: String, type: UTType, images: [CGImage]) {
    let d = CGImageDestinationCreateWithURL(output.appendingPathComponent(name) as CFURL, type.identifier as CFString, images.count, nil)!
    for image in images { CGImageDestinationAddImage(d, image, [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary) }
    precondition(CGImageDestinationFinalize(d))
}
func pdf(_ name: String, count: Int, scanned: Bool = false, empty: Bool = false) {
    var rect = CGRect(x: 0, y: 0, width: width, height: height)
    let c = CGContext(output.appendingPathComponent(name) as CFURL, mediaBox: &rect, nil)!
    for i in 1...count {
        c.beginPDFPage(nil)
        if !empty { if scanned { c.draw(raster(page: i), in: rect) } else { draw(c, page: i) } }
        c.endPDFPage()
    }
    c.closePDF()
}
pdf("single-page.pdf", count: 1)
pdf("multipage.pdf", count: 3)
pdf("scanned.pdf", count: 2, scanned: true)
pdf("empty.pdf", count: 1, empty: true)
imageFile("mixed-layout.png", type: .png, images: [raster(page: 1)])
imageFile("mixed-layout.jpg", type: .jpeg, images: [raster(page: 1)])
imageFile("multipage.tiff", type: .tiff, images: [raster(page: 1), raster(page: 2)])
let html = "<!doctype html><html><body><h1>Local HTML sample</h1><p>Bonjour Montréal, 42.50</p><table><tr><th colspan=\"2\">Merged heading</th></tr><tr><td>Alpha</td><td>1,250.00</td></tr><tr><td>Beta</td><td></td></tr></table></body></html>"
try html.write(to: output.appendingPathComponent("simple.html"), atomically: true, encoding: .utf8)
try #"{\rtf1\ansi\deff0 {\fonttbl {\f0 Helvetica;}}\f0\fs36 Local RTF sample\par\fs24 This is a local conversion fixture.\par Value: 42.50}"#.write(to: output.appendingPathComponent("simple.rtf"), atomically: true, encoding: .utf8)
func archive(_ name: String, entries: [String: String]) throws {
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    for (path, contents) in entries {
        let file = folder.appendingPathComponent(path)
        try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try contents.write(to: file, atomically: true, encoding: .utf8)
    }
    let target = output.appendingPathComponent(name)
    try? FileManager.default.removeItem(at: target)
    let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
    process.currentDirectoryURL = folder; process.arguments = ["-q", "-r", target.path, "."]
    try process.run(); process.waitUntilExit(); precondition(process.terminationStatus == 0)
}
let relPrefix = #"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">"#
let contentPrefix = #"<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>"#
try archive("simple.docx", entries: [
    "[Content_Types].xml": contentPrefix + #"<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"#,
    "_rels/.rels": relPrefix + #"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"#,
    "word/document.xml": #"<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Local DOCX conversion sample</w:t></w:r></w:p><w:p><w:r><w:t>Bonjour Montréal. Original amount: 1,250.00</w:t></w:r></w:p><w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/></w:tblPr><w:tblGrid><w:gridCol w:w="3000"/><w:gridCol w:w="3000"/></w:tblGrid><w:tr><w:tc><w:p><w:r><w:t>Product</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Amount</w:t></w:r></w:p></w:tc></w:tr><w:tr><w:tc><w:p><w:r><w:t>Alpha</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>42.50</w:t></w:r></w:p></w:tc></w:tr></w:tbl><w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr></w:body></w:document>"#
])
try archive("simple.xlsx", entries: [
    "[Content_Types].xml": contentPrefix + #"<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>"#,
    "_rels/.rels": relPrefix + #"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>"#,
    "xl/workbook.xml": #"<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Sample" sheetId="1" r:id="rId1"/></sheets></workbook>"#,
    "xl/_rels/workbook.xml.rels": relPrefix + #"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>"#,
    "xl/worksheets/sheet1.xml": #"<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><cols><col min="1" max="2" width="24" customWidth="1"/></cols><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Product</t></is></c><c r="B1" t="inlineStr"><is><t>Amount</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>Alpha</t></is></c><c r="B2"><v>1250.00</v></c></row><row r="3"><c r="A3" t="inlineStr"><is><t>Beta</t></is></c><c r="B3"><v>42.50</v></c></row></sheetData><pageMargins left="0.7" right="0.7" top="0.75" bottom="0.75" header="0.3" footer="0.3"/></worksheet>"#
])
try archive("simple.pptx", entries: [
    "[Content_Types].xml": contentPrefix + #"<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/><Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/></Types>"#,
    "_rels/.rels": relPrefix + #"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/></Relationships>"#,
    "ppt/presentation.xml": #"<?xml version="1.0"?><p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst><p:sldSz cx="9144000" cy="6858000"/><p:notesSz cx="6858000" cy="9144000"/></p:presentation>"#,
    "ppt/_rels/presentation.xml.rels": relPrefix + #"<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/></Relationships>"#,
    "ppt/slides/slide1.xml": #"<?xml version="1.0"?><p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/><p:sp><p:nvSpPr><p:cNvPr id="2" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="914400" y="914400"/><a:ext cx="7315200" cy="3657600"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US" sz="3200"/><a:t>Local PPTX conversion sample</a:t></a:r></a:p><a:p><a:r><a:rPr lang="en-US" sz="2400"/><a:t>Revenue: 1,250.00 · Share: 60%</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld></p:sld>"#
])
print("Generated PDF, scanned PDF, image, TIFF, HTML, and RTF fixtures in \(output.path)")
