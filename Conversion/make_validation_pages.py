#!/usr/bin/env python3
"""Generate nonconfidential synthetic pages for genuine model checks."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
out=Path(__file__).resolve().parent/'evidence'/'pages'
out.mkdir(parents=True,exist_ok=True)
font='/System/Library/Fonts/Supplemental/Arial.ttf'
def f(n): return ImageFont.truetype(font,n)
img=Image.new('RGB',(640,800),'white');d=ImageDraw.Draw(img)
d.text((38,35),'Parse Studio validation',font=f(30),fill='black')
d.text((38,94),'A local document with text and a table.',font=f(21),fill='black')
d.text((38,135),'Bonjour le monde.  2 + 2 = 4.',font=f(21),fill='black')
for y in [210,255,300,345]: d.line((40,y,600,y),fill='black',width=2)
for x in [40,310,450,600]: d.line((x,210,x,345),fill='black',width=2)
for x,y,t in [(50,221,'Item'),(320,221,'Count'),(460,221,'Price'),(50,266,'Apples'),(320,266,'3'),(460,266,'1.50'),(50,311,'Pears'),(320,311,'5'),(460,311,'2.25')]: d.text((x,y),t,font=f(20),fill='black')
d.text((40,400),'Quarterly share',font=f(25),fill='black')
d.rectangle((100,475,210,650),fill='#5070b0');d.rectangle((300,535,410,650),fill='#60a0b0')
d.text((108,447),'60%',font=f(20),fill='black');d.text((310,507),'40%',font=f(20),fill='black')
d.text((133,660),'A',font=f(20),fill='black');d.text((340,660),'B',font=f(20),fill='black')
d.text((40,725),'Figure 1. Example values, 60% and 40%.',font=f(20),fill='black')
img.save(out/'mixed.png')
simple=Image.new('RGB',(512,256),'white');dr=ImageDraw.Draw(simple)
dr.text((32,35),'Local parsing test',font=f(28),fill='black')
dr.text((32,105),'Hello Nemotron.',font=f(24),fill='black')
simple.save(out/'simple.png')
Image.new('RGB',(512,256),'white').save(out/'empty.png')
print(out)
